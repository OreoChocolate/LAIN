"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import numpy as np
import scipy.io as sio
import wandb
from tqdm import tqdm
from collections import defaultdict

from utils.hico_text_label import hico_unseen_index
import utils.ddp as ddp
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation
import datetime


class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

from torch.cuda import amp
from pocket.ops import relocate_to_cuda

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117,test_loader=None,args=None, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.net = net
        self.max_norm = max_norm
        self.num_classes = num_classes
        self.train_loader = dataloader
        self.test_loader = test_loader
        self.best_unseen = -1
        self.best_seen = -1
        self.args = args
        if self.args.amp:
            self.scaler = amp.GradScaler(enabled=True)

    def _on_end_iteration(self):
        # Print stats in the master process
        if self._verbal and self._state.iteration % self._print_interval == 0:
            self._print_statistics()

        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()

    def _on_start_iteration(self):
        self._state.iteration += 1
        self._state.inputs = relocate_to_cuda(self._state.inputs,ignore=True, non_blocking=True)
        self._state.targets = relocate_to_cuda(self._state.targets,ignore=True, non_blocking=True)

    def _print_statistics(self):
        running_loss = self._state.running_loss.mean()
        t_data = self._state.t_data.sum() / self._world_size
        t_iter = self._state.t_iteration.sum() / self._world_size

        t_iter_mean = self._state.t_iteration.mean()
        t_data_mean = self._state.t_data.mean()

        it_sec = t_iter_mean + t_data_mean

        # Print stats in the master process
        if self._rank == 0:
            num_iter = len(self._train_loader)
            n_d = len(str(num_iter))
            current_iter = self._state.iteration - num_iter * (self._state.epoch - 1)
            print(
                "Epoch [{}/{}], Iter. [{}/{}], "
                "Loss: {:.4f}, "
                "Time[Data/Iter./Remain.]: [{:.2f}s/{:.2f}s/{}]".format(
                self._state.epoch, self.epochs,
                str(current_iter).zfill(n_d),
                num_iter, running_loss, t_data, t_iter,  datetime.timedelta(seconds=(num_iter-current_iter)*it_sec)
            ))
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def _on_each_iteration(self):
        self._state.net.train()
        with amp.autocast(enabled=self.args.amp):
            loss_dict = self._state.net(
                *self._state.inputs, targets=self._state.targets)
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        if self.args.amp:
            self._state.loss = sum(loss for loss in loss_dict.values())
            self._state.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(self._state.loss).backward()
            self.scaler.step(self._state.optimizer)
            self.scaler.update()
        else:
            self._state.loss = sum(loss for loss in loss_dict.values())
            self._state.optimizer.zero_grad(set_to_none=True)
            self._state.loss.backward()
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
            self._state.optimizer.step()

    def _on_end_epoch(self):
        # if self._rank == 0:
        #     self.save_checkpoint()
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()
        self.net.object_class_to_target_class = self.test_loader.dataset.datset.object_class_to_target_class


        if self.args.dataset == 'vcoco':
            ret = self.cache_vcoco(self.test_loader)
            vsrl_annot_file = 'vcoco/data/vcoco/vcoco_test.json'
            coco_file = 'vcoco/data/instances_vcoco_all_2014.json'
            split_file = 'vcoco/data/splits/vcoco_test.ids'
            vcocoeval = eval_vcoco.VCOCOeval(vsrl_annot_file, coco_file, split_file)
            det_file = 'vcoco_cache/cache.pkl'
            b= vcocoeval._do_eval(ret, ovr_thresh=0.5)
            mAPs = {
                'sc2': b[1]
            }

            wandb.log(mAPs)
            return
            # raise NotImplementedError(f"Evaluation on V-COCO has not been implemented.")
        ap = self.test_hico(self.test_loader, self.args)

        self.net.object_class_to_target_class = self.train_loader.dataset.datset.object_class_to_target_class
        self.net.tp = None

        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(self.train_loader.dataset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        if self._rank == 0:
            mAPs = {'mAP': ap.mean() * 100,
                    'rare': ap[rare].mean() * 100,
                    'non-rare': ap[non_rare].mean() * 100
                    }

            print(
                f"The mAP is {ap.mean() * 100:.2f},"
                f" rare: {ap[rare].mean() * 100:.2f},"
                f" none-rare: {ap[non_rare].mean() * 100:.2f},"
            )

            if self.args.zs:
                zs_hoi_idx = hico_unseen_index[self.args.zs_type]
                print(f'>>> zero-shot setting({self.args.zs_type}!!)')
                ap_unseen = []
                ap_seen = []
                for i, value in enumerate(ap):
                    if i in zs_hoi_idx:
                        ap_unseen.append(value)
                    else:
                        ap_seen.append(value)

                ap_unseen = torch.as_tensor(ap_unseen).mean()
                ap_seen = torch.as_tensor(ap_seen).mean()

                mAPs.update({"unseen": ap_unseen * 100, "seen": ap_seen * 100})
                print(
                    f"full mAP: {ap.mean() * 100:.2f}",
                    f"unseen: {ap_unseen * 100:.2f}",
                    f"seen: {ap_seen * 100:.2f}",
                )

            self.save_checkpoint()
            wandb.log(mAPs)

    @torch.no_grad()
    def test_hico(self, dataloader, args=None):
        net = self._state.net
        net.eval()
        dataset = dataloader.dataset.dataset
        interaction_to_verb = torch.as_tensor(dataset.interaction_to_verb)

        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        tgt_num_classes = 600
        
        num_gt = dataset.anno_interaction if args.dataset == "hicodet" else None
        meter = DetectionAPMeter(
            tgt_num_classes, nproc=1,
            num_gt=num_gt,
            algorithm='11P'
        )

        gt_set = []
        pred_list = []

        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            outputs = net(inputs,batch[1])
            # Skip images without detections
            if outputs is None or len(outputs) == 0:
                continue
            # # Batch size is fixed as 1 for inference
            # assert len(output) == 1, f"Batch size is not 1 but {len(outputs)}."
            for output, target in zip(outputs, batch[-1]):
                output = pocket.ops.relocate_to_cpu(output, ignore=True)


                gt_set.append(target['hoi'])

                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
                objects = output['objects']
                scores = output['scores']
                verbs = output['labels']

                if net.module.num_classes==117 or net.module.num_classes==407:
                    interactions = conversion[objects, verbs]
                else:
                    interactions = verbs

                # Recover target box scale
                gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])
                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                unique_hoi = interactions.unique()
                
                for hoi_idx in unique_hoi:
                    gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                    det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                            gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                            boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )
                        # all_det_idxs.append(det_idx)
                # meter.append(scores, interactions, labels)   # scores human*object*verb, interaction（600), labels
                results = (scores, interactions, labels)
                pred_list.append(results)

        gathered_pred_list = []
        for preds in ddp.all_gather(pred_list):
            gathered_pred_list.extend(preds)
        for pred in gathered_pred_list:
            meter.append(*pred)


        ap = meter.eval()
        return ap

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        return all_results