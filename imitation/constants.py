from __future__ import unicode_literals

IMG_HEIGHT = 88
IMG_WIDTH = 200

FEATKEY_KEY = 'key'
FEATKEY_IMG = 'image'

SPEED_BRANCH_INDEX = 4

# data values specified in https://github.com/carla-simulator/imitation-learning#dataset
TGT_STEER = 'Steer'
TGT_GAS = 'Gas'
TGT_BRAKE = 'Brake'
TGT_HAND_BRAKE = 'Hand Brake'  # bool
TGT_REVERSE_GEAR = 'Reverse Gear'  # bool
TGT_STEER_NOISE = 'Steer Noise'
TGT_GAS_NOISE = 'Gas Noise'
TGT_BRAKE_NOISE = 'Brake Noise'
TGT_POS_X = 'Position X'
TGT_POS_Y = 'Position Y'
TGT_SPEED = 'Speed'
TGT_CLSN_OTHER = 'Collision Other'
TGT_CLSN_PED = 'Collision Pedestrian'
TGT_CLSN_CAR = 'Collision Car'
TGT_OPP_LN_INTER = 'Opposite Lane Inter'
TGT_SIDEWALK_INTER = 'Sidewalk Intersect'
TGT_ACC_X = 'Acceleration X,float'
TGT_ACC_Y = 'Acceleration Y'
TGT_ACC_Z = 'Acceleration Z'
TGT_PLATFORM_TIME = 'Platform time'
TGT_GAME_TIME = 'Game Time'
TGT_ORN_X = 'Orientation X'
TGT_ORN_Y = 'Orientation Y'
TGT_ORN_Z = 'Orientation Z'
TGT_HIGH_LVL_CMD = 'High level command'  # (2 Follow lane, 3 Left, 4 Right, 5 Straight)
TGT_NOISE = 'Noise'
TGT_CAMERA = 'Camera'
TGT_ANGLE = 'Angle'
TGT_KEYS = [TGT_STEER, TGT_GAS, TGT_BRAKE, TGT_HAND_BRAKE, TGT_REVERSE_GEAR, TGT_STEER_NOISE, TGT_GAS_NOISE,
            TGT_BRAKE_NOISE, TGT_POS_X, TGT_POS_Y, TGT_SPEED, TGT_CLSN_OTHER, TGT_CLSN_PED, TGT_CLSN_CAR,
            TGT_OPP_LN_INTER, TGT_SIDEWALK_INTER, TGT_ACC_X, TGT_ACC_Y, TGT_ACC_Z, TGT_PLATFORM_TIME, TGT_GAME_TIME,
            TGT_ORN_X, TGT_ORN_Y, TGT_ORN_Z, TGT_HIGH_LVL_CMD, TGT_NOISE, TGT_CAMERA, TGT_ANGLE]
OUTPUT_KEYS = [TGT_STEER, TGT_GAS, TGT_BRAKE]
OUTPUT_KEYS_AND_SPEED = OUTPUT_KEYS + [TGT_SPEED]

OUTPUT_BRANCHES = 'Branches'
OUTPUT_BRANCH_SPEED = 'Speed_Branch'
