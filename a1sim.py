import pybullet as p
import time
import numpy as np
from param import param
from controler import QuadrupedRobot as Quad

# class Dog:
p.connect(p.GUI)                     #连接GUI界面
plane = p.loadURDF("plane.urdf")     #加载URDF
p.setGravity(0,0,-9.8)               #设置重力
p.setTimeStep(1./500)                #设置时间步长
#p.setDefaultContactERP(0)
#urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

#cube_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.2, 0.5])
#cube_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cube_collision_shape)
#p.resetBasePositionAndOrientation(cube_body, [0, 0, 0.5], [0, 0, 0, 1])

vertices = np.array([[-1, -1, 0], [1, -1, 0], [0, 0, 0.2],[-1,1,0],[1,1,0]], dtype=np.float32)
mesh = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, meshScale=[1, 1, 1])
slope_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=mesh)
p.resetBasePositionAndOrientation(slope_body, [0, 2, 0], [0, 0, 0, 1])

block = p.loadURDF("block.urdf",[0,0.9,0.2],[0,0,0,1])



urdfFlags = p.URDF_USE_SELF_COLLISION
quadruped = p.loadURDF("a1/urdf/a1.urdf",[0,0,0.48],[0,0,0,1], flags = urdfFlags,useFixedBase=False)



lower_legs = [2,5,8,11]#小腿关节的参数
for l0 in lower_legs:
    for l1 in lower_legs:#遍历小腿
        if (l1>l0):#设置小腿之间的碰撞
            enableCollision = 1
            p.setCollisionFilterPair(quadruped, quadruped, l0 ,l1 ,enableCollision)

jointIds=[]
paramIds=[]
#拖动滑块控制关节最大力矩 身体站立位姿
maxForceId = p.addUserDebugParameter("  maxForce",0,100,30) #用户可以拖动滑块来调整maxFirceID

body_pose0 = p.addUserDebugParameter("  forward_or_backword ", -0.3,0.30,0.0075)
body_pose1 = p.addUserDebugParameter("  left or right ", -0.14,0.14,-0.001)
body_pose2 = p.addUserDebugParameter("  high or low ",  0.1,0.36,0.25)
body_pose3 = p.addUserDebugParameter("  r_x",-3,3,0)
body_pose4 = p.addUserDebugParameter("  r_y ",-1,1,0)
body_pose5 = p.addUserDebugParameter("  r_z ",-0.3,0.3,0)
body_pose6 = p.addUserDebugParameter("  length of legs point ",0,0.9,0.366)
body_pose7 = p.addUserDebugParameter("  width of legs point ",0,0.735,0.2641)

body_pose_id  = [body_pose0,body_pose1,body_pose2,body_pose3,body_pose4,body_pose5,body_pose6,body_pose7]

A_x = p.addUserDebugParameter("  A_leg_move_fb ",-0.1,0.1,0.0)
A_y = p.addUserDebugParameter("  A_leg_move_lr ",-0.07,0.07,0)
A_z = p.addUserDebugParameter("  A_leg_high ",-0.1,0.1,0)
A_xyz = [A_x,A_y,A_z]

T_id = p.addUserDebugParameter("  T ", 0, 5, 0.3)

gait_type = 'walk'

ctrl = Quad(param)

for j in range (p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped,j)
    jointName = info[1]
    jointType = info[2]
    if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):#判断是否为移动环节或转动关节
        jointIds.append(j)

p.getCameraImage(1480,1320)#GUI界面左边的三个图像 摄像机
p.setRealTimeSimulation(0)#是否开启实时仿真  如果入参为1 为开启 为0 则为关闭
body_pose = np.zeros(8)
A = np.zeros(3)
while(1):
    maxForce = p.readUserDebugParameter(maxForceId)  # 读取滑块设置的力的大小
    for i in range(8):
        body_pose[i] = p.readUserDebugParameter(body_pose_id[i])
    for i in range(3):
        A[i] = p.readUserDebugParameter(A_xyz[i])
    T = p.readUserDebugParameter(T_id)
    joint_angle = ctrl.get_joint_angle(gait_type,body_pose,A,T)

    for j in range (12):                            #遍历关节
        targetPos = float(joint_angle[j])                #目标关节位置
        p.setJointMotorControl2(quadruped, jointIds[j], p.POSITION_CONTROL, targetPos, force=maxForce)
                             #机器人的ID    #关节id        #位置控制           #目标位置    #最大力矩
    p.stepSimulation()                              #设置好力矩了仿真一步
    time.sleep(1./1000.)

