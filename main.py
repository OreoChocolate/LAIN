"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import os
import torch
import random
import warnings
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import wandb

from models.LAIN import build_detector
from utils.args import get_args

from engine import CustomisedDLE
from datasets import DataFactory, custom_collate
from utils.hico_text_label import hico_unseen_index

warnings.filterwarnings("ignore")

def main(rank, args):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.set_device(rank)


    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16'
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'


    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root,
                           clip_model_name=args.clip_model_name, zero_shot=args.zs, zs_type=args.zs_type,
                           num_classes=args.num_classes, args=args)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root,
                          clip_model_name=args.clip_model_name, args=args)

    # if args.dataset == 'vcoco':
    #     object_n_verb_to_interaction = vcoco_object_n_verb_to_interaction(num_object_cls=len(trainset.dataset.objects), num_action_cls=len(trainset.dataset.actions), class_corr=class_corr)
    #     trainset.dataset.object_n_verb_to_interaction = object_n_verb_to_interaction
    #     testset.dataset.object_n_verb_to_interaction = object_n_verb_to_interaction

        
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank
        )
    )

    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(
        testset, shuffle=False, drop_last=False
        )
    )

    args.human_idx = 0
    object_n_verb_to_interaction = trainset.dataset.object_n_verb_to_interaction
    object_to_target = trainset.dataset.object_class_to_target_class


    print('[INFO]: num_classes', args.num_classes)
    lain = build_detector(args, object_to_target, object_n_verb_to_interaction=object_n_verb_to_interaction, clip_model_path=args.clip_dir_vit)

    if args.dataset == 'hicodet' and args.eval:  ## after building model, manually change obj_to_target
        lain.object_class_to_target_class = testset.dataset.object_class_to_target_class

    if os.path.exists(args.resume):
        print(f"===>>> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        lain.load_state_dict(checkpoint['model_state_dict'],strict=False)
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")

    engine = CustomisedDLE(
        lain, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
        test_loader=test_loader,
        args=args
    )

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return
    
    if args.eval:
        lain.eval()
        if args.dataset == 'vcoco':
            import eval_vcoco
            ret = engine.cache_vcoco(test_loader)
            vsrl_annot_file = 'vcoco/data/vcoco/vcoco_test.json'
            coco_file = 'vcoco/data/instances_vcoco_all_2014.json'
            split_file = 'vcoco/data/splits/vcoco_test.ids'
            vcocoeval = eval_vcoco.VCOCOeval(vsrl_annot_file, coco_file, split_file)
            det_file = 'vcoco_cache/cache.pkl'
            ap = vcocoeval._do_eval(ret, ovr_thresh=0.5)
            print(ap)
            return
        ap = engine.test_hico(test_loader, args)
        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        print(
            f"The mAP is {ap.mean()*100:.2f},"
            f" rare: {ap[rare].mean()*100:.2f},"
            f" none-rare: {ap[non_rare].mean()*100:.2f},"
        )
        if args.zs:
            zs_hoi_idx = hico_unseen_index[args.zs_type]
            print(f'>>> zero-shot setting({args.zs_type}!!)')
            ap_unseen = []
            ap_seen = []
            for i, value in enumerate(ap):
                if i in zs_hoi_idx: 
                    ap_unseen.append(value)
                else: 
                    ap_seen.append(value)

            ap_unseen = torch.as_tensor(ap_unseen).mean()
            ap_seen = torch.as_tensor(ap_seen).mean()
            print(
                f"full mAP: {ap.mean()*100:.2f}",
                f"unseen: {ap_unseen*100:.2f}",
                f"seen: {ap_seen*100:.2f}",
            )
            
        return

    for p in lain.detector.parameters():
        p.requires_grad = False

    for n, p in lain.clip_head.named_parameters():
        if n.startswith('visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj'): 
            p.requires_grad = True
        elif 'adaptermlp' in n or "prompt_learner" in n:
            p.requires_grad = True
        elif 'visual_prompt' in n:
            p.requires_grad = True
        else: p.requires_grad = False

    param_dicts = [
        {
            "params": [p for n, p in lain.clip_head.named_parameters()
                    if p.requires_grad]
        },
        { ## others
            "params": [p for n, p in lain.named_parameters()
                    if p.requires_grad and 'clip_head' not in n],
            "lr": args.lr_head,
        },
    ]

    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_vit,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)

    if args.resume:
        # optim.load_state_dict(checkpoint['optim_state_dict'])
        # lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch=checkpoint['epoch']
        iteration = checkpoint['iteration']
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler, epoch=epoch,iteration=iteration, scaler=scaler)
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    import json
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()

    engine(args.epochs)

if __name__ == '__main__':
    args = get_args()
    print(args)

    if args.debug:
        os.environ['WANDB_MODE'] ='disabled'
        os.environ["MASTER_PORT"] = args.port
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    print('WORLD_SIZE ' + str(os.environ.get("WORLD_SIZE",1)))


    os.environ["MASTER_ADDR"] = "localhost"
    local_rank = int(os.environ.get("LOCAL_RANK",0))
    args.local_rank = local_rank
    if local_rank == 0:
        wandb.init(project='LAIN', name=args.output_dir)

    args.world_size = int(os.environ.get("WORLD_SIZE",1))
    main(local_rank,args)
