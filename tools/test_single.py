"""简化的单样本测试脚本，用于生成可视化结果"""
import argparse
import os
import os.path as osp
import numpy as np
import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.model import SoftGroup
from softgroup.util import get_root_logger, load_checkpoint
from tqdm import tqdm


def save_npy(root, name, scan_id, arr):
    """保存单个npy文件"""
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    path = osp.join(root, f'{scan_id}.npy')
    np.save(path, arr)


def save_single_instance(root, scan_id, insts):
    """保存单个实例预测结果"""
    f = open(osp.join(root, f'{scan_id}.txt'), 'w')
    os.makedirs(osp.join(root, 'predicted_masks'), exist_ok=True)
    for i, inst in enumerate(insts):
        assert scan_id == inst['scan_id']
        label_id = inst['label_id']
        conf = inst['conf']
        f.write(f'predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f}\n')
        mask_path = osp.join(root, 'predicted_masks', f'{scan_id}_{i:03d}.txt')
        mask = inst['pred_mask']
        if isinstance(mask, np.ndarray):
            np.savetxt(mask_path, mask, fmt='%d')
        else:
            # 如果是RLE编码，需要解码
            from softgroup.util import rle_decode
            mask = rle_decode(mask)
            np.savetxt(mask_path, mask, fmt='%d')
    f.close()


def main():
    parser = argparse.ArgumentParser('SoftGroup Single Sample Test')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', type=str, default='visualization_results', help='directory for output results')
    parser.add_argument('--sample-idx', type=int, default=0, help='sample index to test (default: 0)')
    args = parser.parse_args()
    
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    logger = get_root_logger()
    
    model = SoftGroup(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    load_checkpoint(args.checkpoint, logger, model)
    
    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=False, **cfg.dataloader.test)
    
    eval_tasks = cfg.model.test_cfg.eval_tasks
    os.makedirs(args.out, exist_ok=True)
    
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            if i != args.sample_idx:
                continue
            logger.info(f'Processing sample {i}/{len(dataloader)}')
            result = model(batch)
            
            scan_id = result['scan_id']
            logger.info(f'Scan ID: {scan_id}')
            
            # 保存坐标和颜色
            if 'semantic' in eval_tasks:
                coords = result['coords_float']
                colors = result['color_feats']
                save_npy(args.out, 'coords', scan_id, coords)
                save_npy(args.out, 'colors', scan_id, colors)
                
                # 保存语义预测和标签
                sem_pred = result['semantic_preds']
                sem_label = result['semantic_labels']
                save_npy(args.out, 'semantic_pred', scan_id, sem_pred)
                save_npy(args.out, 'semantic_label', scan_id, sem_label)
                
                # 保存offset预测和标签
                if 'offset_preds' in result:
                    offset_pred = result['offset_preds']
                    offset_label = result['offset_labels']
                    save_npy(args.out, 'offset_pred', scan_id, offset_pred)
                    save_npy(args.out, 'offset_label', scan_id, offset_label)
            
            # 保存实例预测
            if 'instance' in eval_tasks:
                if 'pred_instances' in result:
                    pred_insts = result['pred_instances']
                    logger.info(f'Found {len(pred_insts)} predicted instances')
                    if len(pred_insts) > 0:
                        # 打印前10个实例的详细信息
                        for idx, inst in enumerate(pred_insts[:10]):
                            logger.info(f'  Instance {idx}: label_id={inst["label_id"]}, conf={inst["conf"]:.4f}')
                        save_single_instance(args.out, scan_id, pred_insts)
                        logger.info(f'Saved {len(pred_insts)} predicted instances')
                    else:
                        logger.warning(f'pred_instances list is empty for {scan_id}')
                else:
                    logger.warning(f'No pred_instances key in result for {scan_id}')
                
                # 保存GT实例
                if 'gt_instances' in result:
                    gt_inst = result['gt_instances']
                    if isinstance(gt_inst, np.ndarray):
                        gt_path = osp.join(args.out, 'gt_instance', f'{scan_id}.txt')
                        os.makedirs(osp.dirname(gt_path), exist_ok=True)
                        np.savetxt(gt_path, gt_inst, fmt='%d')
                        logger.info(f'Saved GT instances for {scan_id}')
            
            logger.info(f'Results saved to {args.out}')
            break  # 只处理一个样本


if __name__ == '__main__':
    main()

