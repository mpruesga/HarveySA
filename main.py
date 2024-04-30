import pybullet as p
from time import sleep
import pybullet_data as pd
import math
import os

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pd.getDataPath())

p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.resetDebugVisualizerCamera(3,-420,-30,[0.3,-0.3,0.5])
p.setGravity(0, 0, -9.81)

tex = p.loadTexture("brain_texture.jpg")
planeId = p.loadURDF("plane.urdf", [0,0,-2])

startOrientation = p.getQuaternionFromEuler([0,0,math.pi])
boxId = p.loadURDF("skull.urdf", [0,0,0],globalScaling=3.5, baseOrientation=startOrientation , useMaximalCoordinates = False, useFixedBase=1)

RobotId = p.loadURDF("GEN3-6DOF_VISION_URDF_ARM_V01.urdf",[1,0,-2], globalScaling=3.0,useFixedBase=1)

bunnyId2 = p.loadSoftBody("torus/lh.obj", simFileName="lh_simplified.vtk", basePosition = [-0.5,1,-0.3], mass = 1, useNeoHookean = 1, NeoHookeanMu = 180, NeoHookeanLambda = 600, NeoHookeanDamping = 0.1, collisionMargin = 0, useSelfCollision = 1, frictionCoeff = 0.5, repulsionStiffness = 800)
#bunnyId2 = p.loadSoftBody("torus/lh.obj", simFileName="lh_simplified.vtk", basePosition = [2,2,0], mass = 1, useNeoHookean = 1, NeoHookeanMu = 180, NeoHookeanLambda = 600, NeoHookeanDamping = 0.1, collisionMargin = 0.006, useSelfCollision = 1, frictionCoeff = 0.5, repulsionStiffness = 800)
#bunnyId = p.loadSoftBody("lh_simplified.vtk", basePosition = [0,2,0], scale = 1, mass = 1, useNeoHookean = 1, NeoHookeanMu = 180, NeoHookeanLambda = 600, NeoHookeanDamping = 0.1, collisionMargin = 0.006, useSelfCollision = 1, frictionCoeff = 0.5, repulsionStiffness = 800)
p.changeVisualShape(bunnyId2, -1, rgbaColor=[1,1,1,1], textureUniqueId=tex, flags=0)

p.setTimeStep(0.01)
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
p.setRealTimeSimulation(0)
slider = [0,0,0,0,0,0]
slider_value = [0,0,0,0,0,0]

numJoints = p.getNumJoints(RobotId)
for joint in range(numJoints-2):
  print(p.getJointInfo(RobotId, joint))
  slider[joint] = p.addUserDebugParameter("Joint "+str(joint), -math.pi, math.pi, 0)
#targetVel = 1  #rad/s
maxForce = 100  #Newton

while p.isConnected():
  p.stepSimulation()
  p.getCameraImage(320,200)
  for joint in range(numJoints-2):
    slider_value[joint] = p.readUserDebugParameter(slider[joint])
    p.setJointMotorControl(RobotId, joint, p.POSITION_CONTROL, slider_value[joint], maxForce)
  """Returns num mesh vertices and vertex positions.
  kwargs = {}
  if hasattr(p, "MESH_DATA_SIMULATION_MESH"):
    kwargs["flags"] = p.MESH_DATA_SIMULATION_MESH
  num_verts, mesh_vert_positions = p.getMeshData(bunnyId2, **kwargs)
  normal_force = p.getContactPoints(bunnyId2)
  for i in range(len(normal_force)):
    print(normal_force[i][9])
  """