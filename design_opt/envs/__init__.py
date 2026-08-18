from .hopper import HopperEnv
from .swimmer import SwimmerEnv
from .ant import AntEnv
from .gap import GapEnv
from .walker import WalkerEnv
from .ant_box import AntPushEnv
from .walker_box import WalkerPushEnv
from .swimmer_box import SwimmerPushEnv
from .ant_box_flip import AntFlipEnv
from .ant_box_lift import AntLiftEnv
from .ant_ import AntEnv_
from .dex_gripper import DexGripperEnv_
from .dex_gripper_stacking import DexGripperStackingEnv
from .dex_gripper_holding import DexGripperHoldingEnv
from .dex_gripper_multi_grasping import DexGripperMultiGraspingEnv
from .dex_gripper_multi_stacking import DexGripperMultiStackingEnv
from .dex_gripper_adversarial_push import DexGripperAdversarialPushEnv
from .dex_gripper_multi_push import DexGripperMultiPushEnv

env_dict = {
    'hopper': HopperEnv,
    'swimmer': SwimmerEnv,
    'ant': AntEnv,
    'gap': GapEnv,
    'walker': WalkerEnv,
    'ant_box': AntPushEnv,
    'walker_box': WalkerPushEnv,
    'swimmer_box': SwimmerPushEnv,
    'ant_box_flip': AntFlipEnv,
    'ant_box_lift': AntLiftEnv,
    'ant_': AntEnv_
}