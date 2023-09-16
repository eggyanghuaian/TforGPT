# Copyright (c) OpenMMLab. All rights reserved.
#完成角度显示，还需要进一步优化水下的竿尖轨迹预测
import copy
import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np

from mmpose.apis import (collect_multi_frames, extract_pose_sequence,
                         get_track_id, inference_pose_lifter_model,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_3d_pose_result)
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo
from mmpose.models import PoseLifter, TopDown

#计算向量角度
import geomeas as gm

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# 全局变量，用于存储关键点历史
"'我希望在原本传入的keypoints中增加4个点，你可以将其编号为17,18,19.20。其中17,18点是由3点与6点连成的线段计算得到，以3点6点连成的线段为中心，将线段长度延长到单位长度2，17，18为这个延长后的线段的两个端点。19与20 关键点由13与16两点以同样的方式计算得到。我希望能通过在keleton中添加[17,18],[19,20]，并在pose_kpt_color，pose_link_color，为其定义颜色后能实现连接与追踪的可视化。'"
past_keypoints = []

def vis_3d_pose_result2(model,
                       result,
                       img=None,
                       dataset='Body3DH36MDataset',
                       dataset_info=None,
                       kpt_score_thr=0.3,
                       radius=16,    #点的半径
                       thickness=12,    #厚度
                       vis_height=800,
                       num_instances=-1,
                       axis_azimuth=70,
                    #    xis_azimuth=axis_azimuth,
                       axis_limit=1.7,
                       axis_dist=10.0,
                       axis_elev=15.0,
                       show=False,
                       out_file=None):
    """Visualize the 3D pose estimation results along with trajectory.
    """
    global past_keypoints
    print("result==",result)
    # 计算新的关键点坐标
    #如果有多个实例，删除后面的实例
    if len(result)>1:
        result.pop()
    # 提取关键点的3D坐标
    keypoints_3d = result[0]['keypoints_3d']  # 假设结果中只有一个实例
    kpt_3 = keypoints_3d[3]
    kpt_6 = keypoints_3d[6]
    kpt_13 = keypoints_3d[13]
    kpt_16 = keypoints_3d[16]

    # 计算点17和点18的坐标
    line_36_3_6 = kpt_6 - kpt_3
    line_36_3_6_normalized = line_36_3_6 / np.linalg.norm(line_36_3_6)
    line_36_3_6_extended = line_36_3_6_normalized * 2  # 延长线段到单位长度2
    kpt_17 = kpt_6 + line_36_3_6_extended
    kpt_18 = kpt_3 - line_36_3_6_extended

    # 计算点19和点20的坐标
    line_13_16 = kpt_16 - kpt_13
    line_13_16_normalized = line_13_16 / np.linalg.norm(line_13_16)
    line_13_16_extended = line_13_16_normalized * 2  # 延长线段到单位长度2
    kpt_19 = kpt_16 + line_13_16_extended
    kpt_20 = kpt_13 - line_13_16_extended

    # 将新的关键点添加到关键点列表中
    keypoints_3d = np.vstack([keypoints_3d, kpt_17, kpt_18, kpt_19, kpt_20])
    result[0]['keypoints_3d'] = keypoints_3d
    
    # # 提取关键点的2D坐标
    # keypoints = result[0]['keypoints']
    # if keypoints is not None and keypoints.any():
    #     keypoints_2d = keypoints[:, :2]

    # # 获取4个附加关键点的坐标
    # kpt_17 = keypoints_2d[3] + (keypoints_2d[6] - keypoints_2d[3]) * 3
    # kpt_18 = keypoints_2d[3] + (keypoints_2d[6] - keypoints_2d[3]) * 6
    # kpt_19 = keypoints_2d[3] + (keypoints_2d[13] - keypoints_2d[3]) * 13
    # kpt_20 = keypoints_2d[3] + (keypoints_2d[16] - keypoints_2d[3]) * 16

    # # 将4个关键点坐标添加到keypoints_2d
    # additional_keypoints = np.array([kpt_17, kpt_18, kpt_19, kpt_20])
    # keypoints_2d = np.vstack((keypoints_2d, additional_keypoints))
    # keypoints = np.vstack((keypoints_2d, additional_keypoints))

    result_keypoints = result[0]['keypoints']
    if result_keypoints is not None and result_keypoints.any():
        keypoints = result_keypoints[:, :2]

    # 计算附加关键点的坐标
    kpt_17 = keypoints[3] + (keypoints[6] - keypoints[3]) * 3
    kpt_18 = keypoints[3] + (keypoints[6] - keypoints[3]) * 6
    kpt_19 = keypoints[3] + (keypoints[13] - keypoints[3]) * 13
    kpt_20 = keypoints[3] + (keypoints[16] - keypoints[3]) * 16
    # 获取原始关键点的 kpt_score
    kpt_scores = result_keypoints[:, 2]
    # 将附加关键点添加到 keypoints 中
    # 创建新的关键点列表，并为新生成的关键点设置相应的 kpt_score
    new_keypoints = np.concatenate((keypoints, [kpt_17, kpt_18, kpt_19, kpt_20]), axis=0)
    new_kpt_scores = np.concatenate((kpt_scores, [kpt_scores[3], kpt_scores[6], kpt_scores[13], kpt_scores[16]]), axis=0)

    # 将新的关键点和 kpt_score 组合为完整的关键点数据
    keypoints_with_scores = np.column_stack((new_keypoints, new_kpt_scores))
    result[0]['keypoints']=keypoints_with_scores
    if dataset_info is not None:
        print("dataset_info is not None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # skeleton = dataset_info.skeleton
        # pose_kpt_color = dataset_info.pose_kpt_color
        # pose_link_color = dataset_info.pose_link_color
        print("NEW color!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])
        skeleton = [
            [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9],
            [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16],
            [17, 18], [19, 20],  # 新增的两条线段
        ]


        pose_kpt_color = palette[
            [9, 0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0, 17, 17, 18, 18]
        ]
        pose_link_color = palette[
            [0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0, 17, 18]
        ]

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEW color")
    else:
        print("dataset_info !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # TODO: These will be removed in the later versions.
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])

        if dataset == 'Body3DH36MDataset':
            skeleton = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                        [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                        [8, 14], [14, 15], [15, 16], [17,18], [19,20]]

            pose_kpt_color = palette[[
                9, 0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0, 9, 9, 0, 0
            ]]
            print("dataset == 'Body3DH36MDataset'",len(pose_kpt_color))
            pose_link_color = palette[[
                0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0,
            ]]

        elif dataset == 'InterHand3DDataset':
            print("3！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
            skeleton = [[0, 1], [1, 2], [2, 3], [3, 20], [4, 5], [5, 6],
                        [6, 7], [7, 20], [8, 9], [9, 10], [10, 11], [11, 20],
                        [12, 13], [13, 14], [14, 15], [15, 20], [16, 17],
                        [17, 18], [18, 19], [19, 20], [21, 22], [22, 23],
                        [23, 24], [24, 41], [25, 26], [26, 27], [27, 28],
                        [28, 41], [29, 30], [30, 31], [31, 32], [32, 41],
                        [33, 34], [34, 35], [35, 36], [36, 41], [37, 38],
                        [38, 39], [39, 40], [40, 41]]

            pose_kpt_color = [[14, 128, 250], [14, 128, 250], [14, 128, 250],
                              [14, 128, 250], [80, 127, 255], [80, 127, 255],
                              [80, 127, 255], [80, 127, 255], [71, 99, 255],
                              [71, 99, 255], [71, 99, 255], [71, 99, 255],
                              [0, 36, 255], [0, 36, 255], [0, 36, 255],
                              [0, 36, 255], [0, 0, 230], [0, 0, 230],
                              [0, 0, 230], [0, 0, 230], [0, 0, 139],
                              [237, 149, 100], [237, 149, 100],
                              [237, 149, 100], [237, 149, 100], [230, 128, 77],
                              [230, 128, 77], [230, 128, 77], [230, 128, 77],
                              [255, 144, 30], [255, 144, 30], [255, 144, 30],
                              [255, 144, 30], [153, 51, 0], [153, 51, 0],
                              [153, 51, 0], [153, 51, 0], [255, 51, 13],
                              [255, 51, 13], [255, 51, 13], [255, 51, 13],
                              [103, 37, 8]]

            pose_link_color = [[14, 128, 250], [14, 128, 250], [14, 128, 250],
                               [14, 128, 250], [80, 127, 255], [80, 127, 255],
                               [80, 127, 255], [80, 127, 255], [71, 99, 255],
                               [71, 99, 255], [71, 99, 255], [71, 99, 255],
                               [0, 36, 255], [0, 36, 255], [0, 36, 255],
                               [0, 36, 255], [0, 0, 230], [0, 0, 230],
                               [0, 0, 230], [0, 0, 230], [237, 149, 100],
                               [237, 149, 100], [237, 149, 100],
                               [237, 149, 100], [230, 128, 77], [230, 128, 77],
                               [230, 128, 77], [230, 128, 77], [255, 144, 30],
                               [255, 144, 30], [255, 144, 30], [255, 144, 30],
                               [153, 51, 0], [153, 51, 0], [153, 51, 0],
                               [153, 51, 0], [255, 51, 13], [255, 51, 13],
                               [255, 51, 13], [255, 51, 13]]
        else:
            print("2！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
            raise NotImplementedError

    if hasattr(model, 'module'):
        model = model.module

    current_frame_keypoints = []
    for res in result:
        keypoints = res.get('keypoints')
        if keypoints is not None and keypoints.any():
            keypoints = keypoints[:, :2]
            if not np.all(keypoints == 0):
                current_frame_keypoints.append(keypoints)
    past_keypoints.append(current_frame_keypoints)
    
    # 保留最近的30帧
    if len(past_keypoints) > 30:
        past_keypoints.pop(0)
    
    # 在图像上绘制关键点轨迹
    for i in range(1, len(past_keypoints)):
        prev_frame_keypoints = past_keypoints[i - 1]
        curr_frame_keypoints = past_keypoints[i]
        for instance_idx in range(len(curr_frame_keypoints)):
            if instance_idx < len(prev_frame_keypoints):
                for kpt_idx, kpt in enumerate(curr_frame_keypoints[instance_idx]):
                    x, y = int(kpt[0]), int(kpt[1])
                    prev_x, prev_y = int(prev_frame_keypoints[instance_idx][kpt_idx][0]), int(prev_frame_keypoints[instance_idx][kpt_idx][1])
                    
                    # 用pose_link_color指定的颜色绘制轨迹
                    for link in skeleton:
                        if kpt_idx == link[0]:
                            line_color = tuple(map(int, pose_kpt_color[kpt_idx]))
                            line_thickness = max(3 - i // 10, 1)
                            if prev_x != 0 and prev_y != 0 and x != 0 and y != 0:
                                cv2.line(img, (prev_x, prev_y), (x, y), line_color, line_thickness)
    
    # 使用原始颜色绘制当前帧的关键点
        '''
    这段代码用于在2D图像上绘制当前帧的关键点。
    通过遍历 current_frame_keypoints 列表中的关键点信息，
    其中 instance_keypoints 是当前帧的关键点信息。然后，
    对于每个关键点索引 kpt_idx 和关键点坐标 kpt，提取其x和y
    坐标值，并将其转换为整数类型。只有当关键点的坐标不为零时，
    才使用预定义的颜色 pose_link_color 和指定的半径 radius 来绘制关键点的圆形标记。
    '''
    print("current_frame_keypoints",current_frame_keypoints)
    for instance_keypoints in current_frame_keypoints:
        for kpt_idx, kpt in enumerate(instance_keypoints):
            x, y = int(kpt[0]), int(kpt[1])
            if x != 0 and y != 0:
                for link in skeleton:
                    if kpt_idx == link[0]:
                        print("skeleton",skeleton)
                        print("len_skeleton",len(skeleton))
                        # print("pose_link_color==",pose_link_color)
                        # print("pose_link_color,kpt_idx",pose_link_color[kpt_idx],kpt_idx)
                        point_color = tuple(map(int, pose_kpt_color[kpt_idx]))
                        cv2.circle(img, (x, y), radius, point_color, -1)
    # print("!!pose_kpt_color=",pose_kpt_color)
    print("dataset=",dataset)
    img_3d = model.show_result(
        result,
        img,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        vis_height=vis_height,
        num_instances=num_instances,
        axis_azimuth=axis_azimuth,
        show=show,
        out_file=out_file)
    
    # 显示图像
    if show:
        cv2.imshow('2D and 3D Poses', img_3d)
        cv2.waitKey(1)
    return img_3d

def convert_keypoint_definition(keypoints, pose_det_dataset,
                                pose_lift_dataset):
    """Convert pose det dataset keypoints definition to pose lifter dataset
    keypoints definition, so that they are compatible with the definitions
    required for 3D pose lifting.
    将姿态检测数据集的关键点定义转换为姿态提升数据集的关键点定义，以便其与3D姿态提升所需的定义兼容。



    Args:
        keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.要转换的2D关键点
        pose_det_dataset, (str): Name of the dataset for 2D pose detector.2D姿态检测器的数据集名称。
        pose_lift_dataset (str): Name of the dataset for pose lifter model.姿态提升模型的数据集名称。

    Returns:
        ndarray[K, 2 or 3]: the transformed 2D keypoints.转换后的2D关键点。
    """
    assert pose_lift_dataset in [
        'Body3DH36MDataset', 'Body3DMpiInf3dhpDataset'
        ], '`pose_lift_dataset` should be `Body3DH36MDataset` ' \
        f'or `Body3DMpiInf3dhpDataset`, but got {pose_lift_dataset}.'

    coco_style_datasets = [
        'TopDownCocoDataset', 'TopDownPoseTrack18Dataset',
        'TopDownPoseTrack18VideoDataset'
    ]
    # 初始化新的关键点数组
    keypoints_new = np.zeros((17, keypoints.shape[1]), dtype=keypoints.dtype)
    if pose_lift_dataset == 'Body3DH36MDataset':
        if pose_det_dataset in ['TopDownH36MDataset']:
            keypoints_new = keypoints
        elif pose_det_dataset in coco_style_datasets:
            # pelvis (root) is in the middle of l_hip and r_hip# 骨盆(根)在左髋和右髋的中间
            keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder胸部在左肩和右肩中间
            keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
            # spine is in the middle of thorax and pelvis脊柱在胸部和骨盆的中间
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # in COCO, head is in the middle of l_eye and r_eye 在COOCO中，头部在左眼和右眼之间
            # in PoseTrack18, head is in the middle of head_bottom and head_top在PoseTrack18中，头部在head_bottom和head_top之间
            keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
            # rearrange other keypoints重新排列其他关键点
            keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        elif pose_det_dataset in ['TopDownAicDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip骨盆(根)在左髋和右髋的中间
            keypoints_new[0] = (keypoints[9] + keypoints[6]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder胸部在左肩和右肩中间
            keypoints_new[8] = (keypoints[3] + keypoints[0]) / 2
            # spine is in the middle of thorax and pelvis脊柱在胸部和骨盆的中间
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # neck base (top end of neck) is 1/4 the way from颈部基底(颈部下端)在颈部(颈部下端)和头顶的1/4处
            # neck (bottom end of neck) to head top
            keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
            # head (spherical centre of head) is 7/12 the way from头(头部球心)在颈部(颈部下端)和头顶的7/12处
            # neck (bottom end of neck) to head top
            keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

            keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[6, 7, 8, 9, 10, 11, 3, 4, 5, 0, 1, 2]]
        elif pose_det_dataset in ['TopDownCrowdPoseDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[6] + keypoints[7]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[0] + keypoints[1]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

            keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[7, 9, 11, 6, 8, 10, 0, 2, 4, 1, 3, 5]]
        else:
            raise NotImplementedError(
                f'unsupported conversion between {pose_lift_dataset} and '
                f'{pose_det_dataset}')

    elif pose_lift_dataset == 'Body3DMpiInf3dhpDataset':
        if pose_det_dataset in coco_style_datasets:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[11] + keypoints[12]) / 2
            # neck (bottom end of neck) is in the middle of
            # l_shoulder and r_shoulder
            keypoints_new[1] = (keypoints[5] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2

            # in COCO, head is in the middle of l_eye and r_eye
            # in PoseTrack18, head is in the middle of head_bottom and head_top
            keypoints_new[16] = (keypoints[1] + keypoints[2]) / 2

            if 'PoseTrack18' in pose_det_dataset:
                keypoints_new[0] = keypoints[1]
                # don't extrapolate the head top confidence score
                keypoints_new[16, 2] = keypoints_new[0, 2]
            else:
                # head top is extrapolated from neck and head
                keypoints_new[0] = (4 * keypoints_new[16] -
                                    keypoints_new[1]) / 3
                # don't extrapolate the head top confidence score
                keypoints_new[0, 2] = keypoints_new[16, 2]
            # arms and legs
            keypoints_new[2:14] = keypoints[[
                6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15
            ]]
        elif pose_det_dataset in ['TopDownAicDataset']:
            # head top is head top
            keypoints_new[0] = keypoints[12]
            # neck (bottom end of neck) is neck
            keypoints_new[1] = keypoints[13]
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[9] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
            # arms and legs
            keypoints_new[2:14] = keypoints[0:12]
        elif pose_det_dataset in ['TopDownCrowdPoseDataset']:
            # head top is top_head
            keypoints_new[0] = keypoints[12]
            # neck (bottom end of neck) is in the middle of
            # l_shoulder and r_shoulder
            keypoints_new[1] = (keypoints[0] + keypoints[1]) / 2
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[7] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
            # arms and legs
            keypoints_new[2:14] = keypoints[[
                1, 3, 5, 0, 2, 4, 7, 9, 11, 6, 8, 10
            ]]

        else:
            raise NotImplementedError(
                f'unsupported conversion between {pose_lift_dataset} and '
                f'{pose_det_dataset}')

    return keypoints_new


def get_Kpoint(vis_3d_pose):
    try:
        # 检查vis_3d_pose是否非空且包含至少一个元素
        if not vis_3d_pose or "keypoints_3d" not in vis_3d_pose[0]:
            print("Error: vis_3d_pose is empty or does not contain keypoints_3d")
            return None

        keypoints_3d = vis_3d_pose[0]["keypoints_3d"]

        # 检查keypoints_3d数组的长度
        if len(keypoints_3d) < 17:
            print("Error: keypoints_3d array is too short")
            return None

        # 提取关键点
        kp2 = keypoints_3d[2]
        kp15 = keypoints_3d[15]
        kp3 = keypoints_3d[3]
        kp6 = keypoints_3d[6]
        kp13 = keypoints_3d[13]
        kp16 = keypoints_3d[16]

        L1 = gm.Vector().calVectorFrom2Points(kp3, kp6)
        L2 = gm.Vector().calVectorFrom2Points(kp13, kp16)
        L3 = gm.Vector().calVectorFrom2Points(kp15, kp16)
        L4 = gm.Vector().calVectorFrom2Points(kp2, kp3)

        # 计算模
        l_x = np.sqrt(L1.dot(L1))
        l_y = np.sqrt(L2.dot(L2))
        L_ARM = np.sqrt(L3.dot(L3))
        L_LEG = np.sqrt(L4.dot(L4))

        # 计算点积
        dian = L1.dot(L2)

        # 计算夹角的cos值
        cos_ = dian / (l_x * l_y)

        # 求得夹角（弧度制）
        angle_hu = np.arccos(cos_)

        # 转换为角度值
        angle_d = angle_hu * 180 / np.pi

        return angle_d
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def main():
    global kpt_history
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        'pose_detector_config',
        type=str,
        default=None,
        help='Config file for the 1st stage 2D pose detector')
    parser.add_argument(
        'pose_detector_checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for the 1st stage 2D pose detector')
    parser.add_argument(
        'pose_lifter_config',
        help='Config file for the 2nd stage pose lifter model')
    parser.add_argument(
        'pose_lifter_checkpoint',
        help='Checkpoint file for the 2nd stage pose lifter model')
    parser.add_argument(
        '--video-path', type=str, default='', help='Video path')
    parser.add_argument(
        '--rebase-keypoint-height',
        action='store_true',
        help='Rebase the predicted 3D pose so its lowest keypoint has a '
        'height of 0 (landing on the ground). This is useful for '
        'visualization when the model do not predict the global position '
        'of the 3D pose.')
    parser.add_argument(
        '--norm-pose-2d',
        action='store_true',
        help='Scale the bbox (along with the 2D pose) to the average bbox '
        'scale of the dataset, and move the bbox (along with the 2D pose) to '
        'the average bbox center of the dataset. This is useful when bbox '
        'is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=-1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        type=str,
        default='vis_results',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.9,
        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=8,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Link thickness for visualization')
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Apply a temporal filter to smooth the 2D pose estimation '
        'results. See also --smooth-filter-cfg.')
    parser.add_argument(
        '--smooth-filter-cfg',
        type=str,
        default='configs/_base_/filters/one_euro.py',
        help='Config file of the filter to smooth the pose estimation '
        'results. See also --smooth.')
    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the 2D pose'
        'detection stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the 2D pose'
        'detection stage. Default: False.')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()
    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Failed to load video file {args.video_path}'

    # First stage: 2D pose detection
    print('Stage 1: 2D pose detection.')

    print('Initializing model...')
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    pose_det_model = init_pose_model(
        args.pose_detector_config,
        args.pose_detector_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_det_model, TopDown), 'Only "TopDown"' \
        'model is supported for the 1st stage (2D pose detection)'

    # frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert 'frame_indices_test' in pose_det_model.cfg.data.test.data_cfg
        indices = pose_det_model.cfg.data.test.data_cfg['frame_indices_test']

    pose_det_dataset = pose_det_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_det_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    pose_det_results_list = []
    next_id = 0
    pose_det_results = []

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    print('Running 2D pose detection inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        pose_det_results_last = pose_det_results
        # pose_det_results_list.extend(copy.deepcopy(pose_det_results))

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(person_det_model, cur_frame)

        # keep the person class bounding boxes.
        person_det_results = process_mmdet_results(mmdet_results,
                                                   args.det_cat_id)

        if args.use_multi_frames:
            frames = collect_multi_frames(video, frame_id, indices,
                                          args.online)

        # make person results for current image
        pose_det_results, _ = inference_top_down_pose_model(
            pose_det_model,
            frames if args.use_multi_frames else cur_frame,
            person_det_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=pose_det_dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_det_results, next_id = get_track_id(
            pose_det_results,
            pose_det_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr)

        pose_det_results_list.append(copy.deepcopy(pose_det_results))

    # Second stage: Pose lifting
    print('Stage 2: 2D-to-3D pose lifting.')

    print('Initializing model...')
    pose_lift_model = init_pose_model(
        args.pose_lifter_config,
        args.pose_lifter_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_lift_model, PoseLifter), \
        'Only "PoseLifter" model is supported for the 2nd stage ' \
        '(2D-to-3D lifting)'
    pose_lift_dataset = pose_lift_model.cfg.data['test']['type']

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.fps
        writer = None

    # convert keypoint definition
    for pose_det_results in pose_det_results_list:
        for res in pose_det_results:
            keypoints = res['keypoints']
            res['keypoints'] = convert_keypoint_definition(
                keypoints, pose_det_dataset, pose_lift_dataset)

    # load temporal padding config from model.data_cfg
    if hasattr(pose_lift_model.cfg, 'test_data_cfg'):
        data_cfg = pose_lift_model.cfg.test_data_cfg
    else:
        data_cfg = pose_lift_model.cfg.data_cfg

    # build pose smoother for temporal refinement
    if args.smooth:
        smoother = Smoother(
            filter_cfg=args.smooth_filter_cfg,
            keypoint_key='keypoints',
            keypoint_dim=2)
    else:
        smoother = None

    num_instances = args.num_instances
    pose_lift_dataset_info = pose_lift_model.cfg.data['test'].get(
        'dataset_info', None)
    if pose_lift_dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        pose_lift_dataset_info = DatasetInfo(pose_lift_dataset_info)

    print('Running 2D-to-3D pose lifting inference...')
    # print('Running 2D-to-3D pose lifting inference...')
    # num_frames = len(pose_det_results_list)
    # for batch_start in range(0, num_frames, batch_size):
    #     batch_end = min(batch_start + batch_size, num_frames)
    #     for i in range(batch_start, batch_end):
    #         pose_det_results = pose_det_results_list[i]
    for i, pose_det_results in enumerate(mmcv.track_iter_progress(pose_det_results_list)):
    
        # extract and pad input pose2d sequence
        pose_results_2d = extract_pose_sequence(
            pose_det_results_list,
            frame_idx=i,
            causal=data_cfg.causal,
            seq_len=data_cfg.seq_len,
            step=data_cfg.seq_frame_interval)

        # smooth 2d results
        if smoother:
            pose_results_2d = smoother.smooth(pose_results_2d)

        # 2D-to-3D pose lifting
        pose_lift_results = inference_pose_lifter_model(
            pose_lift_model,
            pose_results_2d=pose_results_2d,
            dataset=pose_lift_dataset,
            dataset_info=pose_lift_dataset_info,
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d)

        # Pose processing
        pose_lift_results_vis = []
        for idx, res in enumerate(pose_lift_results):
            keypoints_3d = res['keypoints_3d']
            # exchange y,z-axis, and then reverse the direction of x,z-axis
            keypoints_3d = keypoints_3d[..., [0, 2, 1]]
            keypoints_3d[..., 0] = -keypoints_3d[..., 0]
            keypoints_3d[..., 2] = -keypoints_3d[..., 2]
            # rebase height (z-axis)
            if args.rebase_keypoint_height:
                keypoints_3d[..., 2] -= np.min(
                    keypoints_3d[..., 2], axis=-1, keepdims=True)
            res['keypoints_3d'] = keypoints_3d
            # add title
            det_res = pose_det_results[idx]
            instance_id = det_res['track_id']
            res['title'] = f'Prediction_3D  (Azimuth_angle=70 Elev_angle=15.0), ({instance_id})'
            # only visualize the target frame
            res['keypoints'] = det_res['keypoints']
            res['bbox'] = det_res['bbox']
            res['track_id'] = instance_id
            pose_lift_results_vis.append(res)

        # Visualization
        if num_instances < 0:
            num_instances = len(pose_lift_results_vis)
        # print("pose_lift_results_vis==",pose_lift_results_vis.key())
        # print("pose_lift_results_vis",pose_lift_results_vis)
        # print("pose_lift_results_visshape",np.array(pose_lift_results_vis).shape)
        img_vis = vis_3d_pose_result2(
            pose_lift_model,
            result=pose_lift_results_vis,
            img=video[i],
            dataset=pose_lift_dataset,
            # dataset_info=None,
            dataset_info=pose_lift_dataset_info,
            out_file=None,
            radius=args.radius,
            thickness=args.thickness,
            num_instances=num_instances,
            show=args.show)

        angle_d=get_Kpoint(pose_lift_results_vis)
        # ax.text(0.8,0.8,0, 'angle = %f°'%angle_d, style='italic',bbox = {'facecolor': 'yellow'},fontsize=15)
        # font = ImageFont.truetype(font_path, font_size)  # 加载字体
        # cv2.putText(img_vis,"Arm unit length:"+str("{:.2f}".format(L_ARM)), (60, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (200, 0, 0), 2)
        # cv2.putText(img_vis,"Leg unit length:"+str("{:.2f}".format(L_LEG)), (60, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (200, 0, 0), 2)
        # cv2.putText(img_vis,"投影于水面，手杆与站立杆夹角（弧度制）:"+str(angle_hu), (60, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 200, 0), 3)
        cv2.putText(img_vis,"Angle between the handrail and the vertical pole:"+str(angle_d), (60, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200), 2)
        # cv2.putText(img_vis,"Angle between the handrail and the vertical pole:"+str("{:.2f}".format(angle_d)), (60, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200), 2)
        # angle_d=None

        if save_out_video:
            if writer is None:
                writer = cv2.VideoWriter(
                    osp.join(args.out_video_root,
                             f'vis_reslut_{osp.basename(args.video_path)}'), fourcc,
                    fps, (img_vis.shape[1], img_vis.shape[0]))
            writer.write(img_vis)

    if save_out_video:
        writer.release()


if __name__ == '__main__':
    # np.set_printoptions(suppress=True)
    main()



'''
原始输出代码
python demo/body3d_two_stage_video_demoDZP5.py \
        demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
        https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
        configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
        https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
        configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py \
        https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth \
        --video-path data/datasets/dzpvideo/Ce_Mian/Ce_Femail_fortest.mp4 \
        --out-video-root data/datasets/dzpvideo/Ce_Mian/output/ \
        --rebase-keypoint-height
'''
