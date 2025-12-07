"""FOR-instance 数据集可视化脚本"""
import argparse
import os
import os.path as osp
import numpy as np

# FOR-instance 语义类别颜色映射
FORINSTANCE_COLORS = {
    0: [100, 200, 100],    # low_vegetation - 浅绿色
    1: [139, 69, 19],       # terrain - 棕色
    2: [34, 139, 34],       # tree - 深绿色
    -100: [128, 128, 128]   # ignore - 灰色
}

# 实例颜色（使用 Detectron2 调色板）
COLOR_DETECTRON2 = np.array([
    0.000, 0.447, 0.741,
    0.850, 0.325, 0.098,
    0.929, 0.694, 0.125,
    0.494, 0.184, 0.556,
    0.466, 0.674, 0.188,
    0.301, 0.745, 0.933,
    0.635, 0.078, 0.184,
    0.600, 0.600, 0.600,
    1.000, 0.000, 0.000,
    1.000, 0.500, 0.000,
    0.749, 0.749, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 1.000,
    0.667, 0.000, 1.000,
    0.333, 0.333, 0.000,
    0.333, 0.667, 0.000,
    0.333, 1.000, 0.000,
    0.667, 0.333, 0.000,
    0.667, 0.667, 0.000,
    0.667, 1.000, 0.000,
    1.000, 0.333, 0.000,
    1.000, 0.667, 0.000,
    1.000, 1.000, 0.000,
]).astype(np.float32).reshape(-1, 3) * 255


def get_coords_color(prediction_path, room_name, task):
    """获取坐标和颜色"""
    coord_file = osp.join(prediction_path, 'coords', room_name + '.npy')
    color_file = osp.join(prediction_path, 'colors', room_name + '.npy')
    label_file = osp.join(prediction_path, 'semantic_label', room_name + '.npy')
    
    xyz = np.load(coord_file)
    rgb = np.load(color_file)
    label = np.load(label_file)
    
    # 处理RGB颜色（如果被归一化到[-1, 1]）
    if rgb.min() < 0:
        rgb = (rgb + 1) * 127.5
    rgb = rgb.clip(0, 255)
    
    if task == 'input':
        # 原始输入点云
        pass
        
    elif task == 'semantic_gt':
        # GT语义标签
        label = label.astype(int)
        label_rgb = np.zeros_like(rgb)
        for cls_id, color in FORINSTANCE_COLORS.items():
            mask = label == cls_id
            label_rgb[mask] = color
        rgb = label_rgb
        
    elif task == 'semantic_pred':
        # 预测语义标签
        semantic_file = osp.join(prediction_path, 'semantic_pred', room_name + '.npy')
        assert os.path.isfile(semantic_file), f'No semantic result: {semantic_file}'
        label_pred = np.load(semantic_file).astype(int)
        label_rgb = np.zeros_like(rgb)
        for cls_id, color in FORINSTANCE_COLORS.items():
            mask = label_pred == cls_id
            label_rgb[mask] = color
        rgb = label_rgb
        
    elif task == 'offset_semantic_pred':
        # 带offset的语义预测
        semantic_file = osp.join(prediction_path, 'semantic_pred', room_name + '.npy')
        assert os.path.isfile(semantic_file), f'No semantic result: {semantic_file}'
        label_pred = np.load(semantic_file).astype(int)
        label_rgb = np.zeros_like(rgb)
        for cls_id, color in FORINSTANCE_COLORS.items():
            mask = label_pred == cls_id
            label_rgb[mask] = color
        rgb = label_rgb
        
        # 应用offset
        offset_file = osp.join(prediction_path, 'offset_pred', room_name + '.npy')
        if os.path.isfile(offset_file):
            offset_coords = np.load(offset_file)
            xyz = xyz + offset_coords
            print(f'Applied offset to coordinates')
        
    elif task == 'instance_gt':
        # GT实例标签
        inst_label_file = osp.join(prediction_path, 'gt_instance', room_name + '.txt')
        assert os.path.isfile(inst_label_file), f'No GT instance file: {inst_label_file}'
        inst_label = np.array(open(inst_label_file).read().splitlines(), dtype=int)
        
        # FOR-instance格式: class_id * 1000 + instance_id
        # 提取instance_id (取模1000)
        inst_label = inst_label % 1000
        # 将-100转换为-1（背景）
        inst_label[inst_label == 900] = -1  # 如果ignore_label是-100，在%1000后可能变成900
        
        print(f'GT Instance number: {inst_label.max() + 1 if inst_label.max() >= 0 else 0}')
        inst_label_rgb = np.zeros_like(rgb)
        ins_num = inst_label.max() + 1 if inst_label.max() >= 0 else 0
        if ins_num > 0:
            ins_pointnum = np.zeros(ins_num)
            for _ins_id in range(ins_num):
                ins_pointnum[_ins_id] = (inst_label == _ins_id).sum()
            sort_idx = np.argsort(ins_pointnum)[::-1]
            for _sort_id in range(ins_num):
                inst_label_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[
                    _sort_id % len(COLOR_DETECTRON2)]
        rgb = inst_label_rgb
        
    elif task == 'instance_pred':
        # 预测实例
        instance_file = osp.join(prediction_path, 'pred_instance', room_name + '.txt')
        if not os.path.isfile(instance_file):
            print(f'Warning: No instance prediction file: {instance_file}')
            print('Using semantic prediction instead')
            semantic_file = osp.join(prediction_path, 'semantic_pred', room_name + '.npy')
            if os.path.isfile(semantic_file):
                label_pred = np.load(semantic_file).astype(int)
                label_rgb = np.zeros_like(rgb)
                for cls_id, color in FORINSTANCE_COLORS.items():
                    mask = label_pred == cls_id
                    label_rgb[mask] = color
                rgb = label_rgb
            return xyz, rgb
            
        f = open(instance_file, 'r')
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        
        # 初始化颜色为语义预测颜色（作为背景）
        semantic_file = osp.join(prediction_path, 'semantic_pred', room_name + '.npy')
        if os.path.isfile(semantic_file):
            label_pred = np.load(semantic_file).astype(int)
            inst_label_pred_rgb = np.zeros_like(rgb)
            for cls_id, color in FORINSTANCE_COLORS.items():
                mask = label_pred == cls_id
                inst_label_pred_rgb[mask] = color
        else:
            # 如果没有语义预测，使用灰色背景
            inst_label_pred_rgb = np.ones_like(rgb) * 128
        
        ins_num = len(masks)
        ins_pointnum = np.zeros(ins_num)
        inst_label = -1 * np.ones(rgb.shape[0]).astype(int)  # -1表示背景
        
        # 按分数排序，高分数的优先
        scores = np.array([float(x[-1]) for x in masks])
        sort_inds = np.argsort(scores)[::-1]
        
        valid_instances = []
        for i_ in range(len(masks)):
            i = sort_inds[i_]
            mask_path = osp.join(prediction_path, 'pred_instance', masks[i][0])
            if not os.path.isfile(mask_path):
                continue
            if float(masks[i][2]) < 0.0001:  # 使用极低阈值以显示更多实例
                continue
            mask = np.array(open(mask_path).read().splitlines(), dtype=int)
            
            # 检查mask长度是否匹配
            if mask.shape[0] != rgb.shape[0]:
                print(f'Warning: mask length {mask.shape[0]} != point cloud length {rgb.shape[0]}, skipping')
                continue
                
            pointnum = mask.sum()
            if pointnum == 0:
                continue
                
            print(f'{i} {masks[i]}: pointnum: {pointnum}')
            ins_pointnum[i] = pointnum
            # 只给还没有被赋值的点赋值（高分数的优先）
            inst_label[(mask == 1) & (inst_label == -1)] = i
            valid_instances.append(i)
        
        # 按点数排序，给每个实例分配颜色
        if len(valid_instances) > 0:
            valid_instances = np.array(valid_instances)
            valid_pointnum = ins_pointnum[valid_instances]
            sort_idx = np.argsort(valid_pointnum)[::-1]
            
            for color_idx, inst_idx in enumerate(sort_idx):
                inst_id = valid_instances[inst_idx]
                color = COLOR_DETECTRON2[color_idx % len(COLOR_DETECTRON2)]
                inst_label_pred_rgb[inst_label == inst_id] = color
                print(f'Instance {inst_id} -> color {color_idx}: {color}')
        
        # 确保颜色值在0-255范围内
        rgb = np.clip(inst_label_pred_rgb, 0, 255).astype(np.uint8)
    
    # 过滤无效点
    sem_valid = (label != -100)
    xyz = xyz[sem_valid]
    rgb = rgb[sem_valid]
    
    return xyz, rgb


def write_ply(verts, colors, output_file):
    """写入PLY文件"""
    file = open(output_file, 'w')
    file.write('ply\n')
    file.write('format ascii 1.0\n')
    file.write(f'element vertex {len(verts)}\n')
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write(f'{vert[0]:f} {vert[1]:f} {vert[2]:f} {int(color[0])} {int(color[1])} {int(color[2])}\n')
    file.close()
    print(f'Saved PLY file: {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_path', default='./visualization_results', help='path to prediction results')
    parser.add_argument('--room_name', default='plot_1_annotated', help='room/scan name')
    parser.add_argument('--task', 
                       choices=['input', 'semantic_gt', 'semantic_pred', 'offset_semantic_pred', 'instance_gt', 'instance_pred'],
                       default='semantic_pred', help='visualization task')
    parser.add_argument('--out', help='output PLY file path')
    opt = parser.parse_args()
    
    xyz, rgb = get_coords_color(opt.prediction_path, opt.room_name, opt.task)
    points = xyz[:, :3]
    colors = rgb
    
    if opt.out:
        assert '.ply' in opt.out, 'output file should be in .ply format'
        write_ply(points, colors, opt.out)
    else:
        # 默认输出文件名
        out_file = f'{opt.room_name}_{opt.task}.ply'
        write_ply(points, colors, out_file)

