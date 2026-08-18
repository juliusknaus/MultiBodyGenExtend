from khrylib.utils.math import *

"""
def get_single_body_qposaddr(model, body):
    i = model.body_names.index(body)
    start_joint = model.body_jntadr[i]
    # assert start_joint >= 0
    end_joint = start_joint + model.body_jntnum[i]
    start_qposaddr = model.jnt_qposadr[start_joint]
    if end_joint < len(model.jnt_qposadr):
        end_qposaddr = model.jnt_qposadr[end_joint]
    else:
        end_qposaddr = model.nq
    return start_qposaddr, end_qposaddr
"""
import mujoco
def get_single_body_qposaddr(model, body_name):
    import mujoco

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name
    )

    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found")

    start_joint = model.body_jntadr[body_id]
    num_joints = model.body_jntnum[body_id]

    if num_joints == 0:
        return -1, -1   # ✅ safer than (0,0)

    end_joint = start_joint + num_joints

    start_qposaddr = model.jnt_qposadr[start_joint]

    if end_joint < model.njnt:
        end_qposaddr = model.jnt_qposadr[end_joint]
    else:
        end_qposaddr = model.nq

    return start_qposaddr, end_qposaddr