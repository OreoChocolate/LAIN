from torch.utils.data import Dataset
from datasets.vcoco import VCOCO
from datasets.hicodet import HICODet
import datasets.transform as T
import os
import pocket
from PIL import Image

import numpy as np
import torch
from utils.hico_text_label import hico_unseen_index

def custom_collate(batch):
    images = []
    targets = []
    # images_clip = []

    for im, tar in batch:
        images.append(im)
        targets.append(tar)

        # images_clip.append(im_clip)
    return images, targets


class DataFactory(Dataset):
    def __init__(self, name, partition, data_root, clip_model_name, zero_shot=False, zs_type='rare_first',
                 num_classes=600, args=None):  ##ViT-B/16, ViT-L/14@336px

        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)
        assert clip_model_name in ['ViT-L/14@336px', 'ViT-B/16']
        self.clip_model_name = clip_model_name
        if self.clip_model_name == 'ViT-B/16':
            self.clip_input_resolution = 224
        elif self.clip_model_name == 'ViT-L/14@336px':
            self.clip_input_resolution = 336

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict'),
                args=args
            )
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                                       ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        # add clip normalization
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
            ])
            self.clip_transforms = T.Compose([
                T.IResize([self.clip_input_resolution,self.clip_input_resolution]),
            ])
        else:
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
            ])
            self.clip_transforms = T.Compose([
                T.IResize([self.clip_input_resolution,self.clip_input_resolution]),
            ])

        self.partition = partition
        self.name = name
        self.count=0
        self.zero_shot = zero_shot
        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            self.zs_type = zs_type
            self.filtered_hoi_idx = hico_unseen_index[self.zs_type]

        self.keep = [i for i in range(len(self.dataset))]

        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            self.zs_keep = []
            self.remain_hoi_idx = [i for i in np.arange(600) if i not in self.filtered_hoi_idx]
            if os.path.exists(f'datasets/zs_{args.zs_type}_{self.partition}_idx.pkl'):
                import pickle
                self.zs_keep = pickle.load(open(f'datasets/zs_{args.zs_type}_{self.partition}_idx.pkl', 'rb'))
                print(f'datasets/zs_{args.zs_type}_{self.partition}_idx.pkl is loaded')
            else:
                for i in self.keep:
                    (image, target), filename = self.dataset[i]
                    # if 1 in target['hoi']:
                    #     pdb.set_trace()
                    mutual_hoi = set(self.remain_hoi_idx) & set([_h.item() for _h in target['hoi']])
                    if len(mutual_hoi) != 0:
                        self.zs_keep.append(i)
                import pickle
                pickle.dump(self.zs_keep, open(f'datasets/zs_{args.zs_type}_{self.partition}_idx.pkl', 'wb'))
            self.keep = self.zs_keep

            self.dataset.zs_object_to_target = [[] for _ in range(self.dataset.num_object_cls)]
            if num_classes == 600:
                for corr in self.dataset.class_corr:
                    if corr[0] not in self.filtered_hoi_idx:
                        self.dataset.zs_object_to_target[corr[1]].append(corr[0])
            else:
                for corr in self.dataset.class_corr:
                    if corr[0] not in self.filtered_hoi_idx:
                        self.dataset.zs_object_to_target[corr[1]].append(corr[2])





    def __len__(self):
        return len(self.keep)

    # train detr with roi
    def __getitem__(self, i):
        (image, target), filename = self.dataset[self.keep[i]]
        # (image, target), filename = self.dataset[i]
        if self.name == 'hicodet' and self.zero_shot and self.partition == 'train2015':
            _boxes_h, _boxes_o, _hoi, _object, _verb = [], [], [], [], []
            for j, hoi in enumerate(target['hoi']):
                if hoi in self.filtered_hoi_idx:
                    # pdb.set_trace()
                    continue
                _boxes_h.append(target['boxes_h'][j])
                _boxes_o.append(target['boxes_o'][j])
                _hoi.append(target['hoi'][j])
                _object.append(target['object'][j])
                _verb.append(target['verb'][j])
            target['boxes_h'] = torch.stack(_boxes_h)
            target['boxes_o'] = torch.stack(_boxes_o)
            target['hoi'] = torch.stack(_hoi)
            target['object'] = torch.stack(_object)
            target['verb'] = torch.stack(_verb)
        w,h = image.size
        target['orig_size'] = torch.tensor([h,w])

        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')
            ## TODO add target['hoi']

        image, target = self.transforms(image, target)
        image_clip, target = self.clip_transforms(image, target)
        image, _ = self.normalize(image, None)
        image_clip, target = self.normalize(image_clip, target)
        target['filename'] = filename
        target['filename_num'] = torch.tensor(int(filename.split('.')[0].split('_')[-1]))
        return (image,image_clip), target
