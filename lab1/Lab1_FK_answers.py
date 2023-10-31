import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    # file read
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()

        joint_name = []
        joint_parent = []
        joint_offset = []
        stack = []

        for i in range(len(lines)):
            if lines[i].startswith("ROOT"):
                joint_name.append(lines[i].split()[1])
                joint_parent.append(-1)
            elif lines[i].startswith("MOTION"):
                break
            else:
                tmp_line = lines[i].split()
                if tmp_line[0] == '{':
                    stack.append(len(joint_name) - 1)
                elif tmp_line[0] == '}':
                    stack.pop()
                elif tmp_line[0] == 'JOINT':
                    joint_name.append(tmp_line[1])
                    joint_parent.append(stack[-1])
                elif tmp_line[0] == 'End':
                    joint_name.append(joint_name[stack[-1]] + '_end')
                    joint_parent.append(stack[-1])
                elif tmp_line[0] == "OFFSET":
                    joint_offset.append(np.array([float(x) for x in tmp_line[1:4]]).reshape(1, -1))
                else:
                    continue

    joint_offset = np.concatenate(joint_offset, axis=0)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    frame_data = motion_data[frame_id]
    joint_positions = []
    joint_orientations = []
    joint_localrotations = []

    count = 0
    for i in range(len(joint_name)):
        if '_end' in joint_name[i]:
            joint_localrotations.append([0., 0., 0.])
        else:
            joint_localrotations.append(frame_data[3 * count + 3: 3 * count + 6])
            count += 1

    joint_positions.append(np.array(frame_data[0: 3]).reshape(1, -1))
    joint_orientations.append(R.from_euler('XYZ', joint_localrotations[0], degrees=True).as_quat().reshape(1, -1))
    for i in range(1, len(joint_name)):
        parent_index = joint_parent[i]
        joint_ort = R.from_quat(joint_orientations[parent_index]) * R.from_euler('XYZ', joint_localrotations[i],
                                                                                 degrees=True)
        joint_pos = joint_positions[parent_index] + joint_offset[i] * np.matrix(
            R.from_quat(joint_orientations[parent_index]).as_matrix()).transpose()
        joint_orientations.append(joint_ort.as_quat().reshape(1, -1))
        joint_positions.append(np.array(joint_pos[0]))

    joint_orientations = np.concatenate(joint_orientations, axis=0)
    joint_positions = np.concatenate(joint_positions, axis=0)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    a_motion_data = load_motion_data(A_pose_bvh_path)
    joint_name_t, _, _ = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_a, _, _ = part1_calculate_T_pose(A_pose_bvh_path)

    A_joint_map = {}
    count = 0
    for i in range(len(joint_name_a)):
        if '_end' in joint_name_a[i]:
            count += 1
        A_joint_map[joint_name_a[i]] = i - count

    motion_data = []
    for i in range(a_motion_data.shape[0]):
        data = []
        for joint in joint_name_t:
            index = A_joint_map[joint]
            if joint == 'RootJoint':
                data += list(a_motion_data[i][0: 6])
            elif joint == 'lShoulder':
                data += list((R.from_euler('XYZ', a_motion_data[i][index * 3 + 3: index * 3 + 6], degrees=True)
                              * R.from_euler('XYZ', [0, 0, -45], degrees=True)).as_euler('XYZ', True))
            elif joint == 'rShoulder':
                data += list((R.from_euler('XYZ', a_motion_data[i][index * 3 + 3: index * 3 + 6], degrees=True)
                              * R.from_euler('XYZ', [0, 0, 45], degrees=True)).as_euler('XYZ', True))
            elif '_end' in joint:
                continue
            else:
                data += list(a_motion_data[i][index * 3 + 3: index * 3 + 6])
        motion_data.append(np.array(data).reshape(1, -1))

    motion_data = np.concatenate(motion_data,axis=0)
    return motion_data
