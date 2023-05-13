import numpy as np
import time


class QuadrupedRobot:

    def __init__(self, param):
        self.lenth_of_Hip = param[0]
        self.lenth_of_Thigh = param[1]
        self.lenth_of_Calf = param[2]
        self.length_of_body = param[3]
        self.width_of_body = param[4]

    def ik_of_leg(self, xyz_pos):
        l1 = self.lenth_of_Hip
        l2 = self.lenth_of_Thigh
        l3 = self.lenth_of_Calf
        x = xyz_pos[:, 0]
        y = xyz_pos[:, 1]
        z = xyz_pos[:, 2]
        # 解析法逆运动学求解关节角度
        l0 = np.sqrt(np.square(y) + np.square(z))
        lyz = np.sqrt(np.square(l0) - np.square(l1))
        lxz = np.sqrt(np.square(x) + np.square(lyz))
        n = (np.square(lxz)-np.square(l2)-np.square(l3))/(2*l2)
        theta1 = np.arctan2(-x, lyz)
        theta2 = np.arccos((n+l2)/lxz)
        joint1 = np.zeros(4)
        for i in range(4):
            if i == 1 or i == 3:
                joint1[i] = np.arctan2(y[i], -z[i])-np.arctan2(l1, lyz[i])  # 左侧的单腿逆运动学和右侧的单腿逆运动学这里差一个负号
            else:
                joint1[i] = np.arctan2(y[i], -z[i])+np.arctan2(l1, lyz[i])  # 左侧的单腿逆运动学和右侧的单腿逆运动学这里差一个负号
        joint2 = theta1 + theta2
        joint3 = -np.arccos(n/l2)
        joints = np.concatenate(np.transpose([joint1, joint2, joint3]))
        return joints

    def ik_of_body(self, body_pose):
        lob = self.length_of_body
        wob = self.width_of_body
        x = body_pose[0]
        y = body_pose[1]
        z = body_pose[2]
        rx = body_pose[3]
        ry = body_pose[4]
        rz = body_pose[5]
        lof = body_pose[6]
        wof = body_pose[7]

        # 旋转矩阵
        rotx = np.array([[1, 0, 0],
                         [0, np.cos(rx), -np.sin(rx)],
                         [0, np.sin(rx),  np.cos(rx)]])

        roty = np.array([[np.cos(ry), 0, np.sin(ry)],
                         [0, 1, 0],
                         [-np.sin(ry), 0, np.cos(ry)]])

        rotz = np.array([[np.cos(rz), -np.sin(rz), 0],
                         [np.sin(rz),  np.cos(rz), 0],
                         [0, 0, 1]])

        rot = np.dot(np.dot(rotx, roty), rotz)
        rot = rot.T

        # 躯干顶点相对于躯干坐标系坐标  顺序为右前，左前，右后，左后
        oa = np.array([[0.5*lob, -0.5*wob, 0],
                       [0.5*lob, 0.5*wob, 0],
                       [-0.5*lob, -0.5*wob, 0],
                       [-0.5*lob, 0.5*wob, 0]])

        # 机器人足点相对于地面坐标系坐标
        ob = np.array([[0.5*lof, -0.5*wof, 0],
                       [0.5*lof, 0.5*wof, 0],
                       [-0.5*lof, -0.5*wof, 0],
                       [-0.5*lof, 0.5*wof, 0]])
        # 世界坐标系下的oa
        oa = np.dot(oa, rot)
        # 最终要输出的足端位置
        xyz_pose = np.array([[x, y, z], [x, y, z], [x, y, z], [x, y, z]])
        xyz_pose = -oa - xyz_pose + ob
        # 因为求出来的位置要送给单腿逆运动学求解，而单腿逆运动学需要的是相对肩部坐标系 这里求得的是世界坐标系下的xyz 因此需要进一步转换
        rot = rot.T
        return xyz_pose,rot

    def get_joint_angle(self,gait_type,body_pose,A,T):
        if gait_type == 'trot':
            xyz_from_foots = self.trot_move(A[0], A[1], A[2], T)
        elif gait_type == 'walk':
            xyz_from_foots = self.walk_move(A[0], A[1], A[2], T)
        else:
            xyz_from_foots = self.jump(A[0], A[1], A[2], T)
        xyz_form_body , rot= self.ik_of_body(body_pose)
        sum_xyz = xyz_form_body + xyz_from_foots
        sum_xyz = np.dot(sum_xyz, rot)
        joint_angles = self.ik_of_leg(sum_xyz)


        return joint_angles


    def walk_move(self, a_x, a_y, a_z, t):
        x_rf, y_rf, z_rf = self.cycloid(a_x, a_y, a_z, t, phase=0, gait_type='walk')
        x_lf, y_lf, z_lf = self.cycloid(a_x, a_y, a_z, t, phase=t/2, gait_type='walk')
        x_rb, y_rb, z_rb = self.cycloid(a_x, a_y, a_z, t, phase=t/4, gait_type='walk')
        x_lb, y_lb, z_lb = self.cycloid(a_x, a_y, a_z, t, phase=3*t/4, gait_type='walk')
        walk_xyz = [[x_rf, y_rf, z_rf],
                    [x_lf, y_lf, z_lf],
                    [x_rb, y_rb, z_rb],
                    [x_lb, y_lb, z_lb]]
        return walk_xyz

    def trot_move(self, A_x , A_y ,A_z, T):
        x_lf, y_lf, z_lf = self.cycloid(A_x, A_y ,A_z,T,phase = 0,gait_type = 'trot')
        x_rf, y_rf, z_rf = self.cycloid(A_x, A_y, A_z, T, phase=T / 2,gait_type = 'trot')
        trot_xyz = [[x_rf, y_rf, z_rf],
                    [x_lf, y_lf, z_lf],
                    [x_lf, y_lf, z_lf],
                    [x_rf, y_rf, z_rf]]
        return trot_xyz

    def jump(self, A_x , A_y ,A_z, T):
        x, y, z = self.cycloid(A_x, A_y, A_z, T, phase=0, gait_type='jump')
        jump_xyz = [[x, y, z],
                    [x, y, z],
                    [x, y, z],
                    [x, y, z]]
        return jump_xyz

    def walk_turn(self,A_x, A_y ,A_z,T):
        pass

    def trot_turn(self,A_x, A_y ,A_z,T):
        pass

    def cycloid(self, A_x, A_y ,A_z,T,phase,gait_type):  #幅值为A 周期为T 摆线
        if gait_type == 'trot':
            t = self.time_wive(T = T,proportion=0.5,phase=phase)
            judge = (time.time() + phase) % T < T / 2
            x = 0.5 * A_x * (2 * np.pi * t - np.sin(t * 2 * np.pi)) - 0.5 * A_x * np.pi
            y = 0.5 * A_y * (2 * np.pi * t - np.sin(t * 2 * np.pi)) - 0.5 * A_y * np.pi
        elif gait_type == 'walk':
            t = self.time_wive(T,proportion=0.25,phase=phase)
            judge = (time.time() + phase) % T < T / 4
            if judge:
                x = 0.5 * A_x * (2 * np.pi * t - np.sin(t * 2 * np.pi)) - 0.5 * A_x * np.pi
                y = 0.5 * A_y * (2 * np.pi * t - np.sin(t * 2 * np.pi)) - 0.5 * A_y * np.pi
            else:
                x = 0.5 * A_x * 2 * np.pi * t - 0.5 * A_x * np.pi
                y = 0.5 * A_y * 2 * np.pi * t - 0.5 * A_y * np.pi
        elif gait_type == 'jump':
            t = self.time_wive(T, proportion=0.02, phase=phase)
            judge = (time.time() + phase) % T < 0.1*T
            x = 0.5 * A_x * (2 * np.pi * t - np.sin(t * 2 * np.pi)) - 0.5 * A_x * np.pi
            y = 0.5 * A_y * (2 * np.pi * t - np.sin(t * 2 * np.pi)) - 0.5 * A_y * np.pi


        if judge:
            z = 0.5 * A_z * (1 - np.cos(t * 2 * np.pi ))
        else:
            z = 0
        return x,y,z

    def time_wive(self,T = 2,proportion=0.5,phase=0):
        t = (time.time() + phase) % T
        if t < proportion * T:
            value = t / (T * proportion)
        else:
            value = 1 / (1-proportion) - (t / (1-proportion)) / T
        return value
