import numpy as np
from gym import utils
from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv
from khrylib.robot.xml_robot_multi import Robot
from khrylib.utils import get_single_body_qposaddr, get_graph_fc_edges
from khrylib.utils.transformation import quaternion_matrix
from copy import deepcopy
import mujoco
import time
from gym.spaces import Box
import os
import re
from scipy.spatial import ConvexHull, Delaunay, QhullError
from design_opt.utils.rand import rand_coord


class DexGripperMultiStackingEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, cfg, agent):
        self.cur_t = 0
        self.cfg = cfg
        self.env_specs = cfg.env_specs
        self.task_specs = cfg.task_specs
        self.obj_name = self.task_specs.get("obj_name", "box")
        self.agent = agent
        if self.cfg.xml_name == "default":
            self.model_xml_file = os.path.join(cfg.project_path, "assets", "mujoco_envs", "dex_gripper_multi_stacking.xml")
        else:
            self.model_xml_file = os.path.join(cfg.project_path, "assets", "mujoco_envs", f"{self.cfg.xml_name}.xml")
        # robot xml
        self.robot = Robot(cfg.robot_cfg, xml=self.model_xml_file)
        self.init_xml_str = self.robot.export_xml_string()
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        # design options
        self.clip_qvel = cfg.obs_specs.get('clip_qvel', False)
        self.use_projected_params = cfg.obs_specs.get('use_projected_params', True)
        self.abs_design = cfg.obs_specs.get('abs_design', False)
        self.use_body_ind = cfg.obs_specs.get('use_body_ind', False)
        self.use_body_depth_height = cfg.obs_specs.get('use_body_depth_height', False)
        self.use_shortest_distance = cfg.obs_specs.get('use_shortest_distance', False)
        self.use_position_encoding = cfg.obs_specs.get('use_position_encoding', False)
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()
        self.design_param_names = self.robot.get_params(get_name=True)
        self.attr_design_dim = self.design_ref_params.shape[-1]
        self.index_base = 5
        self.stage = 'skeleton_transform'    # transform or execute
        self.control_nsteps = 0
        self.sim_specs = set(cfg.obs_specs.get('sim', []))
        self.attr_specs = set(cfg.obs_specs.get('attr', []))
        # task attr
        self.box_pos = np.array(self.task_specs.get('box_pos'))
        self.random_boxes_pos = bool(self.task_specs.get('random_boxes_pos', False))
        self.box_random_margin = float(self.task_specs.get('box_random_margin', 0.02))
        self.box_spawn_range = self.task_specs.get('box_spawn_range', None)
        self.box_pair_clearance = float(self.task_specs.get('box_pair_clearance', 0.02))
        self.box_spawn_max_tries = int(self.task_specs.get('box_spawn_max_tries', 200))
        self.rob_box_dist = np.array([0.0, 0.0, 0.0])
        if self.task_specs.get('mov_goal', False):
            self.goal_pos = np.array(self.task_specs.get('goal_pos'))
            self.box_goal_dist = self.box_pos - self.goal_pos
        self.num_agents = getattr(cfg, "num_agents", 2)
        self.agent_stages = ['skeleton_transform'] * self.num_agents
        self.agent_stage_steps = [0] * self.num_agents
        self.agent_control_nsteps = [0] * self.num_agents
        self.agent_names = [
            f"0_{i+1}" for i in range(self.num_agents)
        ]
        self.active_agent_id = 0
        self.stage = self.agent_stages[0]
        self.cur_t = self.agent_stage_steps[0]
        self.control_nsteps = self.agent_control_nsteps[0]
        self.design_cur_params = self._tile_design_params(self.design_ref_params)
        MujocoEnv.__init__(self, self.model_xml_file, 4)
        utils.EzPickle.__init__(self)
        self.agent_names = self._resolve_agent_root_names()
        # Per-agent execution controls are 3D (root x/y/z intent for this agent).
        # Using 3 * num_agents here causes a control layout mismatch in per-agent step.
        self.control_action_dim = 3
        self.skel_num_action = 3 if cfg.enable_remove else 2

        self.agent_body_ids = []
        self.agent_geom_ids = []
        self.agent_torso_geom_ids = []

        for name in self.agent_names:

            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                name
            )

            self.agent_body_ids.append(body_id)

            # --- torso geom ---
            geom_start = self.model.body_geomadr[body_id]
            geom_num = self.model.body_geomnum[body_id]

            if geom_num > 0:
                self.agent_torso_geom_ids.append(geom_start)


        self._cache_box_addrs()
        self._configure_gripper_collision_masks()
        self.box_id = self.box1_geom_id

        # Keep aliases for existing reward/helper code that expects single-box names.
        self.obj_body_id = self.box1_body_id
        self.target_body_id = self.box2_body_id
        self.obj_geom_id = self.box1_geom_id
        self.target_geom_id = self.box2_geom_id
        self.box_body_id = self.box1_body_id
        self.box_geom_id = self.box1_geom_id
        self.box_qpos_adr = self.box1_qpos_adr
        self.box_qvel_adr = self.box1_qvel_adr
        self.box_init_height = self.model.geom_size[self.box1_geom_id][2]
        self.target_pos = self.data.xpos[self.box2_body_id].copy()

        self.design_cur_params = self._tile_design_params(self.design_ref_params)
        self.sim_obs_dim = self.get_sim_obs().shape[-1]
        self.attr_fixed_dim = self.get_attr_fixed().shape[-1]
        self.weights = cfg.task_specs.get('weights', {})

    def split_agent_action(self, action):
        return action.reshape(self.num_agents, -1)

    def _sync_debug_stage_state(self, agent_id):
        self.active_agent_id = agent_id
        self.stage = self.agent_stages[agent_id]
        if self.stage == 'execution':
            self.cur_t = self.agent_control_nsteps[agent_id]
        else:
            self.cur_t = self.agent_stage_steps[agent_id]
        self.control_nsteps = self.agent_control_nsteps[agent_id]

    def _all_agents_in_execution(self):
        return all(stage == 'execution' for stage in self.agent_stages)

    def _agent_reward_gates(self, rob_pos_bef, rob_pos_aft, contact_count, force_mag):
        movement_displacement_threshold = float(self.cfg.reward_specs.get('movement_displacement_threshold', 1e-4))
        contact_force_threshold = float(self.cfg.reward_specs.get('contact_force_threshold', 0.0))

        root_displacement = float(np.linalg.norm(rob_pos_aft - rob_pos_bef))
        agent_moved = root_displacement > movement_displacement_threshold
        has_box_contact = (contact_count > 0) and (force_mag >= contact_force_threshold)
        return agent_moved, has_box_contact

    def _combine_agent_controls(self, agent_actions):
        if not agent_actions:
            return np.zeros((0,), dtype=np.float32)

        if not hasattr(self, "model") or self.model is None:
            return np.asarray(agent_actions[0], dtype=np.float32)

        combined_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for agent_id, agent_action in enumerate(agent_actions):
            agent_ctrl = np.asarray(agent_action, dtype=np.float32)
            actuator_ids = self._get_agent_actuator_ids(agent_id)

            if not actuator_ids:
                continue

            if agent_ctrl.shape[0] == self.model.nu:
                combined_ctrl[actuator_ids] = agent_ctrl[actuator_ids]
            elif agent_ctrl.size == len(actuator_ids):
                combined_ctrl[actuator_ids] = agent_ctrl
            else:
                combined_ctrl[actuator_ids] = np.asarray(agent_ctrl.reshape(-1)[:len(actuator_ids)], dtype=np.float32)

        return combined_ctrl

    def _tile_design_params(self, design_params):
        return np.repeat(design_params[None, ...], self.num_agents, axis=0)

    def _align_action_rows(self, action, target_rows):
        action = np.asarray(action)
        if action.shape[0] == target_rows:
            return action
        if action.shape[0] > target_rows:
            return action[:target_rows]

        pad_rows = target_rows - action.shape[0]
        pad_shape = (pad_rows,) + action.shape[1:]
        pad = np.zeros(pad_shape, dtype=action.dtype)
        return np.concatenate([action, pad], axis=0)

    def _update_design_param_tensor(self, agent_id, design_params, reset_all=False):
        design_params = np.asarray(design_params)
        if reset_all or not hasattr(self, "design_cur_params") or self.design_cur_params.ndim != 3:
            self.design_cur_params = self._tile_design_params(design_params)
            return

        new_tensor = self._tile_design_params(design_params)
        old_tensor = self.design_cur_params
        row_count = min(old_tensor.shape[1], new_tensor.shape[1])
        col_count = min(old_tensor.shape[2], new_tensor.shape[2])
        new_tensor[:, :row_count, :col_count] = old_tensor[:, :row_count, :col_count]
        new_tensor[agent_id] = design_params.copy()
        self.design_cur_params = new_tensor

    def get_agent_obs(self, agent_id=0):
        return self._get_obs(agent_id)

    def _resolve_agent_root_names(self):
        root_names = []
        for i in range(self.num_agents):
            suffix = i + 1

            joint_name = f"root_x_{suffix}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                body_id = int(self.model.jnt_bodyid[joint_id])
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name is not None:
                    root_names.append(body_name)
                    continue

            for candidate in (f"0_{suffix}", f"agent_{suffix}_{suffix}", f"agent_{suffix}"):
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, candidate)
                if body_id != -1:
                    root_names.append(candidate)
                    break
            else:
                raise ValueError(f"Could not resolve root body name for agent {i}")

        return root_names

    def _resolve_model_body_name(self, body_name, agent_id=0):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id != -1:
            return body_name

        # Body names in xml_robot_multi may be canonicalized like "0_2".
        # Map those root names back to the model root name for that agent.
        if isinstance(body_name, str) and body_name.startswith("0_"):
            parts = body_name.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                suffix_agent_idx = int(parts[1]) - 1
                if 0 <= suffix_agent_idx < len(self.agent_names):
                    mapped_root = self.agent_names[suffix_agent_idx]
                    mapped_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mapped_root)
                    if mapped_id != -1:
                        return mapped_root

        if not hasattr(self, "agent_names") or len(self.agent_names) == 0:
            return body_name

        if body_name == "0":
            return self.agent_names[agent_id]
        


        suffix = f"_{agent_id + 1}"

        # xml_robot_multi can generate condensed names like 10_1 while XML model may
        # still have 1_1 before first morphology reload.
        condensed = re.sub(r"0_(\d+)$", r"_\1", body_name)
        if condensed != body_name:
            condensed_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, condensed)
            if condensed_id != -1:
                return condensed

        has_agent_suffix = body_name.rsplit("_", 1)[-1].isdigit() if "_" in body_name else False
        if not has_agent_suffix:
            candidate = f"{body_name}{suffix}"
            candidate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if candidate_id != -1:
                return candidate

        for i in range(self.model.nbody):
            candidate_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if candidate_name is None:
                continue
            if candidate_name.endswith(suffix) and candidate_name.startswith(f"{body_name}_"):
                return candidate_name
            if condensed != body_name and candidate_name.endswith(suffix) and candidate_name.startswith(f"{condensed}_"):
                return candidate_name
        

        raise ValueError(f"Body '{body_name}' not found for agent {agent_id}")

    def _get_agent_root_body_id(self, agent_id):
        suffix = agent_id + 1

        # Primary path: resolve the body attached to root_x_{agent} joint.
        joint_name = f"root_x_{suffix}"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            return int(self.model.jnt_bodyid[joint_id])

        fallback_names = []
        if hasattr(self, "agent_names") and 0 <= agent_id < len(self.agent_names):
            fallback_names.append(self.agent_names[agent_id])
        fallback_names.extend([
            f"agent_{suffix}_{suffix}",
            f"agent_{suffix}",
        ])

        for candidate in fallback_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if body_id != -1:
                return body_id

        raise ValueError(f"Could not resolve root body for agent_id={agent_id}")
            

    def allow_add_body(self, body):
        add_body_condition = self.cfg.add_body_condition
        max_nchild = add_body_condition.get('max_nchild', 3)
        min_nchild = add_body_condition.get('min_nchild', 0)
        return body.depth >= self.cfg.min_body_depth and body.depth < self.cfg.max_body_depth - 1 and len(body.child) < max_nchild and len(body.child) >= min_nchild
    
    def allow_remove_body(self, body):
        if body.depth >= self.cfg.min_body_depth + 1 and len(body.child) == 0:
            if body.depth == 1:
                return body.parent.child.index(body) > 0
            else:
                return True
        return False

    def _body_belongs_to_agent(self, body, agent_id):
        agent_suffix = f"_{agent_id + 1}"

        body_suffix = getattr(body, "agent_suffix", None)
        if body_suffix is not None:
            return body_suffix == agent_suffix

        if getattr(body, "name", "").endswith(agent_suffix):
            return True

        for joint in getattr(body, "joints", []):
            joint_name = getattr(joint, "name", "")
            if joint_name.endswith(agent_suffix):
                return True

        return False

    def _get_agent_body_indices(self, agent_id=0):
        return [
            i for i, body in enumerate(self.robot.bodies)
            if self._body_belongs_to_agent(body, agent_id)
        ]

    def _get_agent_bodies(self, agent_id=0):
        indices = self._get_agent_body_indices(agent_id)
        bodies = [self.robot.bodies[i] for i in indices]
        return bodies, indices

    def apply_skel_action(self, skel_action, agent_id=0):
        agent_suffix = f"_{agent_id + 1}"
        bodies = [
            body for body in self.robot.bodies
            if self._body_belongs_to_agent(body, agent_id)
        ]

        for body, a in zip(bodies, skel_action):
            if a == 1 and self.allow_add_body(body):
                self.robot.add_child_to_body(body, agent_id=agent_suffix)
            if a == 2 and self.allow_remove_body(body):
                self.robot.remove_body(body, agent_id=agent_suffix)

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            xml_str_fixed = self.cur_xml_str.replace(' center="0 0 0"', '')
            self.reload_sim_model(xml_str_fixed)
            self.cur_xml_str = xml_str_fixed
            self._cache_box_addrs()
            self._configure_gripper_collision_masks()
            #self.reload_sim_model(xml_str.decode('utf-8'))
        except:
            print(self.cur_xml_str)
            return False      
        self._update_design_param_tensor(agent_id, self.get_attr_design(), reset_all=True)
        return True

    def set_design_params(self, in_design_params, agent_id=0):
        agent_bodies, _ = self._get_agent_bodies(agent_id)
        design_params = in_design_params
        for params, body in zip(design_params, agent_bodies):
            body.set_params(params, pad_zeros=True, map_params=True)
            body.sync_node()

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            xml_str_fixed = self.cur_xml_str.replace(' center="0 0 0"', '')
            self.reload_sim_model(xml_str_fixed)
            self.cur_xml_str = xml_str_fixed
            self._cache_box_addrs()
            self._configure_gripper_collision_masks()
            #self.reload_sim_model(xml_str.decode('utf-8'))
        except:
            print(self.cur_xml_str)
            return False
        if self.use_projected_params:
            self._update_design_param_tensor(agent_id, self.get_attr_design())
        else:
            self._update_design_param_tensor(agent_id, in_design_params.copy())
        return True

    
    """
    def action_to_control(self, a):
        # IMPORTANT: use model.nu (number of actuators), NOT data.ctrl view
        ctrl = np.zeros(self.model.nu, dtype=np.float32)

        assert a.shape[0] == len(self.robot.bodies)

        print("Action to control - robot bodies and actions:", a.shape[0], "and ", len(self.robot.bodies), " and ", self.model.nu)

        # skip root body (same as your original code)
        for body, body_a in zip(self.robot.bodies[1:], a[1:]):
        #for body, body_a in zip(self.robot.bodies, a):
            aname = body.get_actuator_name()
            if aname is None:
                aname = "0_joint"
            print("aname: ", aname, " body: ", body, " and body_a: ", body_a)

            # map actuator name → actuator id in MuJoCo
            act_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                aname
            )

            # only assign if valid actuator exists
            if act_id != -1:
                print("Mapping action to control - actuator id: ", act_id, " for body: ", body.name)
                ctrl[act_id] = body_a
            print("Here we have the act_id and the ctrl: ", act_id, " and ", ctrl, " and ", self.model.nu)

        return ctrl 
    """
    
    def is_gripper_contact(self, contact, limb_geom_ids):
        return (
            (contact.geom1 in limb_geom_ids and contact.geom2 == self.box_geom_id)
            or
            (contact.geom2 in limb_geom_ids and contact.geom1 == self.box_geom_id)
        )

    def capsule_endpoints(self, model, data, geom_id):
            """
            Returns the two world-space endpoints of a capsule geom.
            """
            center = data.geom_xpos[geom_id].copy()
            R = data.geom_xmat[geom_id].reshape(3, 3).copy()
            half_length = model.geom_size[geom_id][1]  # capsule half-length

            axis = R[:, 2]  # capsule axis in world frame
            p1 = center + half_length * axis
            p2 = center - half_length * axis
            return p1, p2


    def gripper_point_cloud(self, model, data, limb_geom_ids):
        pts = []

        for gid in limb_geom_ids:
            p1, p2 = self.capsule_endpoints(model, data, gid)
            pts.append(p1)
            pts.append(p2)

        return np.array(pts)



    def compute_gripper_hull(self, points):
        points = np.asarray(points)

        # Convex hull in 3D needs at least 4 points with shape (N, 3).
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
            return None, None

        try:
            hull = ConvexHull(points)
            tri = Delaunay(points[hull.vertices])
        except (QhullError, ValueError, IndexError):
            return None, None

        return hull, tri

    def point_in_gripper(self, point, tri):
        return tri.find_simplex(point) >= 0

    def sample_points_in_box(self, center, size, n_samples=500):
        """
        center: (3,)
        size: (3,) full box size (MuJoCo geom size = half-extents)
        """
        half = size

        samples = np.random.uniform(-1, 1, size=(n_samples, 3)) * half
        return samples + center

    def occupancy_in_gripper(self, box_center, box_size, tri, n_samples=500):
        if tri is None:
            return 0.0

        samples = self.sample_points_in_box(box_center, box_size, n_samples)

        inside = tri.find_simplex(samples) >= 0

        return np.mean(inside)  # fraction

    def gripper_box_overlap(self, box_pos, box_size, tri, n_samples=500):
        """
        Returns fraction of sampled box volume inside gripper convex hull.
        """

        if tri is None:
            return 0.0

        samples = self.sample_points_in_box(box_pos, box_size, n_samples)

        inside = tri.find_simplex(samples) >= 0

        return float(np.mean(inside))

    def sample_points_in_hull_bbox(self, points, n_samples=5000):
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)

        samples = np.random.uniform(
            low=mins,
            high=maxs,
            size=(n_samples, 3)
        )

        return samples

    def points_inside_box(self, points, box_center, box_size):
        """
        box_size = half extents
        """

        local = points - box_center

        inside = np.all(
            np.abs(local) <= box_size,
            axis=1
        )

        return inside

    def gripper_compactness_score(
        self,
        hull_points,
        tri,
        box_center,
        box_size,
        n_samples=5000
    ):
        """
        Measures how much of the gripper hull
        volume is occupied by the box.
        """

        if tri is None:
            return 0.0

        # Sample points around hull
        samples = self.sample_points_in_hull_bbox(
            hull_points,
            n_samples
        )

        # Keep only points inside hull
        inside_hull = tri.find_simplex(samples) >= 0

        hull_samples = samples[inside_hull]

        if len(hull_samples) == 0:
            return 0.0

        # Check how many hull points lie inside box
        inside_box = self.points_inside_box(
            hull_samples,
            box_center,
            box_size
        )

        return float(np.mean(inside_box))

    def compute_force_closure_reward(
        self,
        model,
        data,
        limb_geom_ids,
        box_geom_id
    ):

        total_force = np.zeros(3)

        for i in range(data.ncon):

            c = data.contact[i]

            is_contact = (
                (c.geom1 in limb_geom_ids and c.geom2 == box_geom_id)
                or
                (c.geom2 in limb_geom_ids and c.geom1 == box_geom_id)
            )

            if not is_contact:
                continue

            force6 = np.zeros(6)

            mujoco.mj_contactForce(
                model,
                data,
                i,
                force6
            )

            f_contact = force6[:3]

            R = c.frame.reshape(3, 3)

            f_world = R @ f_contact

            # ensure consistent sign
            if c.geom2 == box_geom_id:
                total_force += f_world
            else:
                total_force -= f_world

        return -np.linalg.norm(total_force)

    def compute_force_closure_reward(
        self,
        model,
        data,
        limb_geom_ids,
        box_geom_id
    ):

        total_force = np.zeros(3)

        for i in range(data.ncon):

            c = data.contact[i]

            is_contact = (
                (c.geom1 in limb_geom_ids and c.geom2 == box_geom_id)
                or
                (c.geom2 in limb_geom_ids and c.geom1 == box_geom_id)
            )

            if not is_contact:
                continue

            force6 = np.zeros(6)

            mujoco.mj_contactForce(
                model,
                data,
                i,
                force6
            )

            f_contact = force6[:3]

            R = c.frame.reshape(3, 3)

            f_world = R @ f_contact

            # ensure consistent sign
            if c.geom2 == box_geom_id:
                total_force += f_world
            else:
                total_force -= f_world

        return -np.linalg.norm(total_force)

    def compute_total_contact_force_magnitude(
        self,
        model,
        data,
        limb_geom_ids,
        box_geom_id
    ):
        """
        Sum magnitudes of all gripper-box contact forces.
        """

        total_force_magnitude = 0.0

        for i in range(data.ncon):

            c = data.contact[i]

            is_contact = (
                (c.geom1 in limb_geom_ids and c.geom2 == box_geom_id)
                or
                (c.geom2 in limb_geom_ids and c.geom1 == box_geom_id)
            )

            if not is_contact:
                continue

            force6 = np.zeros(6)

            mujoco.mj_contactForce(
                model,
                data,
                i,
                force6
            )

            f_contact = force6[:3]

            mag = np.linalg.norm(f_contact)

            total_force_magnitude += mag

        return total_force_magnitude

    def compute_box_gripper_contacts(
        self,
        model,
        data,
        limb_geom_ids,
        box_geom_id
    ):
        """
        Counts contacts between gripper geoms and box geom.
        """

        n_contacts = 0

        for i in range(data.ncon):

            c = data.contact[i]

            is_contact = (
                (c.geom1 in limb_geom_ids and c.geom2 == box_geom_id)
                or
                (c.geom2 in limb_geom_ids and c.geom1 == box_geom_id)
            )

            if is_contact:
                n_contacts += 1

        return n_contacts

    def compute_box_lowest_point(
        self,
        model,
        data,
        geom_id
    ):
        """
        Computes world-space lowest point of a box geom.
        """

        # Box center
        center = data.geom_xpos[geom_id]

        # World rotation matrix
        R = data.geom_xmat[geom_id].reshape(3, 3)

        # Half extents
        half = model.geom_size[geom_id]

        # 8 local corners
        corners_local = np.array([
            [ sx, sy, sz ]
            for sx in (-half[0], half[0])
            for sy in (-half[1], half[1])
            for sz in (-half[2], half[2])
        ])

        # Transform to world frame
        corners_world = (
            corners_local @ R.T
        ) + center

        # Lowest z
        lowest_z = np.min(corners_world[:, 2])

        return lowest_z
    
    def compute_stability_reward(self, data, box_body_id, gripper_body_id):
        """
        Rewards the box moving together with the gripper.

        High reward:
            box rigidly follows gripper

        Low reward:
            box slips, rotates, shakes
        """

        # MuJoCo cvel format:
        # [wx wy wz vx vy vz]

        # Box velocities
        box_vel6 = data.cvel[box_body_id]

        box_ang_vel = box_vel6[:3]
        box_lin_vel = box_vel6[3:]

        # Gripper velocities
        grip_vel6 = data.cvel[gripper_body_id]

        grip_ang_vel = grip_vel6[:3]
        grip_lin_vel = grip_vel6[3:]

        # Relative motion
        rel_lin = box_lin_vel - grip_lin_vel
        rel_ang = box_ang_vel - grip_ang_vel

        # Combined relative motion magnitude
        rel_motion = (
            np.linalg.norm(rel_lin)
            +
            np.linalg.norm(rel_ang)
        )

        # Reward low relative motion
        reward = np.exp(-rel_motion)

        return reward

    def compute_holding_reward(
        self,
        model,
        data,
        box_geom_id,
        initial_box_height,
        root_body_name=None,
        contact_threshold=2,
        min_lift_height=0.05,
        slip_vel_threshold=0.2
    ):
        """
        Rewards stable long-duration grasp holding.

        Components:
        -----------
        + Lift reward
        + Persistent contact reward
        + Stability reward
        + Low-slip reward
        + Time-alive holding bonus
        - Drop penalty
        """

        total_reward = 0.0

        # ---------------------------------------------------
        # BOX POSITION
        # ---------------------------------------------------

        box_pos = data.geom_xpos[box_geom_id]

        # geom_size = half extents
        box_half_height = model.geom_size[box_geom_id][2]

        # lowest point of box
        bottom_z = box_pos[2] - box_half_height

        lifted = bottom_z > min_lift_height

        # ---------------------------------------------------
        # CONTACT COUNT
        # ---------------------------------------------------

        # Keep contact counting agent-specific in multi-agent setups.
        limb_geom_ids = self.get_limb_geom_ids(model, root_body_name=root_body_name)

        contact_count = self.compute_box_gripper_contacts(model, data, limb_geom_ids, box_geom_id)

        sufficient_contacts = contact_count >= contact_threshold

        # ---------------------------------------------------
        # BOX VELOCITY
        # ---------------------------------------------------

        box_body_id = model.geom_bodyid[box_geom_id]

        linear_vel = data.cvel[box_body_id][:3]

        vel_mag = np.linalg.norm(linear_vel)

        stable_motion = vel_mag < slip_vel_threshold

        # ---------------------------------------------------
        # HOLDING CONDITION
        # ---------------------------------------------------

        successful_hold = (
            lifted
            and sufficient_contacts
            and stable_motion
        )

        # ---------------------------------------------------
        # REWARD TERMS
        # ---------------------------------------------------

        # 1. Lift amount reward
        if lifted:
            lift_height = bottom_z - min_lift_height
            total_reward += 5.0 * lift_height

        # 2. Contact reward
        total_reward += 0.5 * contact_count

        # 3. Stable holding bonus
        if successful_hold:
            total_reward += 10.0

        # 4. Low velocity reward
        total_reward += np.exp(-5.0 * vel_mag)

        # 5. Drop penalty
        if not lifted and vel_mag > 0.5:
            total_reward -= 20.0

        return total_reward

    def get_limb_geom_ids(self, model, root_body_name=None, exclude_geom_type="ellipsoid"):
        root_body_id = -1
        if root_body_name is not None:
            root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)

        suffix = None
        if root_body_name is not None and "_" in root_body_name:
            suffix = root_body_name.split("_")[-1]

        limb_geom_ids = []

        for body_id in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            if body_name is None:
                continue

            # skip non-gripper bodies
            if body_name in {"box_1", "box_2", "box_walls"}:
                continue

            if root_body_name is not None:
                if body_id == root_body_id:
                    continue
                if suffix is not None:
                    body_has_suffix = body_name.endswith(f"_{suffix}")
                    if not body_has_suffix:
                        jnt_start = model.body_jntadr[body_id]
                        jnt_count = model.body_jntnum[body_id]
                        body_has_suffix = False
                        for j in range(jnt_count):
                            jnt_id = jnt_start + j
                            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
                            if jnt_name is not None and jnt_name.endswith(f"_{suffix}"):
                                body_has_suffix = True
                                break
                    if not body_has_suffix:
                        continue

            geom_start = model.body_geomadr[body_id]
            geom_count = model.body_geomnum[body_id]

            for i in range(geom_count):
                geom_id = geom_start + i
                geom_type = model.geom_type[geom_id]

                # only finger capsules
                if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    limb_geom_ids.append(geom_id)

        return limb_geom_ids

    def _get_agent_actuator_ids(self, agent_id):
        if not hasattr(self, "model") or self.model is None:
            return []

        suffix = f"_{agent_id + 1}"
        actuator_ids = []

        try:
            root_body_id = self._get_agent_root_body_id(agent_id)
        except Exception:
            root_body_id = -1

        for act_id in range(self.model.nu):
            trn_type = int(self.model.actuator_trntype[act_id])
            trn_id = int(self.model.actuator_trnid[act_id][0])

            if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT) and trn_id >= 0:
                if trn_id >= self.model.njnt:
                    continue

                body_id = int(self.model.jnt_bodyid[trn_id])
                cur = body_id
                while cur != -1:
                    if cur == root_body_id:
                        actuator_ids.append(act_id)
                        break

                    parent = int(self.model.body_parentid[cur])
                    if parent == cur:
                        break
                    cur = parent
                continue

            act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
            if act_name is not None and act_name.endswith(suffix):
                actuator_ids.append(act_id)

        if not actuator_ids:
            for act_id in range(self.model.nu):
                act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
                if act_name is not None and act_name.endswith(suffix):
                    actuator_ids.append(act_id)

        return sorted(set(actuator_ids))

    def _split_agent_actuator_ids(self, agent_id):
        actuator_ids = self._get_agent_actuator_ids(agent_id)
        root_ids = []
        other_ids = []
        for act_id in actuator_ids:
            act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
            if act_name is not None and act_name.startswith("root_"):
                root_ids.append(act_id)
            else:
                other_ids.append(act_id)

        root_order = ["root_x_motor", "root_y_motor", "root_z_motor"]
        root_ids.sort(
            key=lambda idx: next(
                (i for i, prefix in enumerate(root_order)
                 if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx).startswith(prefix)),
                len(root_order)
            )
        )

    
        return root_ids, other_ids

    def _get_joint_actuator_id(self, joint_id):
        for act_id in range(self.model.nu):
            trn_type = int(self.model.actuator_trntype[act_id])
            trn_id = int(self.model.actuator_trnid[act_id][0])
            if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT) and trn_id == int(joint_id):
                return act_id
        return -1

    def _get_agent_ordered_body_actuator_ids(self, agent_id):
        agent_bodies, _ = self._get_agent_bodies(agent_id)
        ordered_ids = []
        used_ids = set()

        for body in agent_bodies[1:]:
            try:
                model_body_name = self._resolve_model_body_name(body.name, agent_id=agent_id)
            except Exception:
                continue

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, model_body_name)
            if body_id == -1:
                continue

            jnt_start = int(self.model.body_jntadr[body_id])
            jnt_count = int(self.model.body_jntnum[body_id])
            joint_ids = [jnt_start + j for j in range(jnt_count)]
            joint_ids.sort(key=lambda jid: int(self.model.jnt_qposadr[jid]))

            for joint_id in joint_ids:
                if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    continue

                act_id = self._get_joint_actuator_id(joint_id)
                if act_id == -1 or act_id in used_ids:
                    continue

                ordered_ids.append(act_id)
                used_ids.add(act_id)
                break

        return ordered_ids
 
    def action_to_control(self, a, agent_id):
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        a = np.asarray(a)
        root_ids, other_ids = self._split_agent_actuator_ids(agent_id)
        ordered_other_ids = self._get_agent_ordered_body_actuator_ids(agent_id)
        if not ordered_other_ids:
            ordered_other_ids = other_ids

        if a.ndim >= 2:
            flat_actions = np.asarray(a[:, 0]).reshape(-1)
        else:
            flat_actions = a.reshape(-1)

        if len(root_ids) == 0:
            body_actions = flat_actions[1:]
            root_actions = np.zeros((0,), dtype=flat_actions.dtype)
        else:
            root_actions = flat_actions[:len(root_ids)]
            body_actions = flat_actions[len(root_ids):]

        for act_id, val in zip(root_ids, root_actions):
            ctrl[act_id] = val

        for act_id, val in zip(ordered_other_ids, body_actions[:len(ordered_other_ids)]):
            ctrl[act_id] = val
            
        return ctrl

    def _get_agent_root_obs_window(self, agent_id, qpos_len=6, qvel_len=7):
        joint_ids = []
        for act_id in self._get_agent_actuator_ids(agent_id):
            joint_id = int(self.model.actuator_trnid[act_id][0])
            if joint_id >= 0:
                joint_ids.append(joint_id)

        # Keep deterministic ordering by generalized coordinate addresses.
        joint_ids = sorted(set(joint_ids), key=lambda jid: int(self.model.jnt_qposadr[jid]))

        qpos_vals = []
        qvel_vals = []
        for joint_id in joint_ids:
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            qvel_adr = int(self.model.jnt_dofadr[joint_id])
            qpos_dim = 7 if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE else 1
            qvel_dim = 6 if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE else 1
            qpos_vals.extend(self.data.qpos[qpos_adr:qpos_adr + qpos_dim].tolist())
            qvel_vals.extend(self.data.qvel[qvel_adr:qvel_adr + qvel_dim].tolist())

        qpos_vals = np.asarray(qpos_vals, dtype=np.float64)
        qvel_vals = np.asarray(qvel_vals, dtype=np.float64)

        if qpos_vals.shape[0] < qpos_len:
            qpos_vals = np.pad(qpos_vals, (0, qpos_len - qpos_vals.shape[0]), mode='constant')
        if qvel_vals.shape[0] < qvel_len:
            qvel_vals = np.pad(qvel_vals, (0, qvel_len - qvel_vals.shape[0]), mode='constant')

        return qpos_vals[:qpos_len], qvel_vals[:qvel_len]

    
    def multi_step(self, joint_action, agent_id=None):
        if agent_id is not None:
            # Per-agent training path: policy already outputs one full action tensor.
            return self.step(joint_action, agent_id=agent_id)

        if isinstance(joint_action, (list, tuple)):
            agent_actions = list(joint_action)
        else:
            joint_action = np.asarray(joint_action)
            if joint_action.ndim == 3:
                if joint_action.shape[0] != self.num_agents:
                    raise ValueError(
                        f"Expected joint_action first dimension to be num_agents={self.num_agents}, got {joint_action.shape[0]}"
                    )
                agent_actions = [joint_action[idx] for idx in range(self.num_agents)]
            elif joint_action.dtype == object and joint_action.ndim == 1 and joint_action.shape[0] == self.num_agents:
                agent_actions = list(joint_action)
            else:
                if joint_action.ndim < 2:
                    raise ValueError(
                        f"Unsupported joint_action shape {joint_action.shape} for multi-agent stepping"
                    )
                agent_actions = np.split(joint_action, self.num_agents, axis=1)

        if len(agent_actions) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} agent actions, got {len(agent_actions)}"
            )

        if self._all_agents_in_execution():
            base_qpos = self.data.qpos.copy()
            base_qvel = self.data.qvel.copy()
            self.stage = 'execution'
            base_control_nsteps = list(self.agent_control_nsteps)

            per_agent_controls = []
            for idx in range(self.num_agents):
                agent_bodies_i, _ = self._get_agent_bodies(idx)
                action_i = self._align_action_rows(np.asarray(agent_actions[idx]), len(agent_bodies_i))
                control_i = action_i[:, :self.control_action_dim]
                per_agent_controls.append(self.action_to_control(control_i, idx))

            combined_ctrl = self._combine_agent_controls(per_agent_controls)

            try:
                self.do_simulation(combined_ctrl, self.frame_skip)
            except:
                print(self.cur_xml_str)
                return [self._get_obs(i) for i in range(self.num_agents)], 0.0, True, False, {
                    'use_transform_action': False,
                    'stage': 'execution'
                }

            aft_qpos = self.data.qpos.copy()
            aft_qvel = self.data.qvel.copy()

            rewards = 0.0
            termination = False
            truncation = False
            info = {}
            obs = [None] * self.num_agents
            agent_rewards = [0.0] * self.num_agents
            agent_infos = [None] * self.num_agents
            processed_agent_ids = []

            pre_rob_box_dists = []
            pre_rob_positions = []
            self.set_state(base_qpos.copy(), base_qvel.copy())
            mujoco.mj_forward(self.model, self.data)
            box_pos_bef = self.get_body_com("box_1")[0:3].copy()
            target_dist_bef = np.linalg.norm(self.data.xpos[self.box1_body_id] - self.target_pos)
            if self.task_specs.get('mov_goal', False):
                box_goal_dist_bef = np.linalg.norm(box_pos_bef - self.goal_pos)
            else:
                box_goal_dist_bef = None
            for idx in range(self.num_agents):
                agent_body_name = self.agent_names[idx]
                rob_pos_bef = self.get_body_com(agent_body_name)[0:3].copy()
                pre_rob_positions.append(rob_pos_bef)
                pre_rob_box_dists.append(np.linalg.norm(rob_pos_bef - box_pos_bef))

            self.set_state(aft_qpos.copy(), aft_qvel.copy())
            mujoco.mj_forward(self.model, self.data)
            for idx in range(self.num_agents):
                self.agent_control_nsteps[idx] = base_control_nsteps[idx] + 1
            self.control_nsteps = max(self.agent_control_nsteps) if len(self.agent_control_nsteps) > 0 else 0
            self.cur_t = self.control_nsteps

            box_pos_aft = self.get_body_com("box_1")[0:3].copy()
            box_state_aft = self.data.qpos[self.box1_qpos_adr:self.box1_qpos_adr + 7].copy()
            target_dist_aft = np.linalg.norm(self.data.xpos[self.box1_body_id] - self.target_pos)
            self.box_pos = box_pos_aft.copy()

            s = self.state_vector()
            zdir = quaternion_matrix(s[3:7])[:3, 2]
            ang = np.arccos(zdir[2])
            done_condition = self.cfg.done_condition
            min_height = done_condition.get('min_height', 0.0)
            max_height = done_condition.get('max_height', 7.5)
            max_ang = 180
            max_nsteps = done_condition.get('max_nsteps', 1000)

            box_id = self.box1_body_id
            box_geom_id = self.obj_geom_id
            box_size = self.model.geom_size[box_geom_id]

            for idx in range(self.num_agents):
                self.active_agent_id = idx
                agent_body_name = self.agent_names[idx]

                rob_pos_aft = self.get_body_com(agent_body_name)[0:3].copy()
                rob_box_dist_aft = np.linalg.norm(rob_pos_aft - box_pos_aft)
                self.rob_box_dist = rob_pos_aft - box_pos_aft

                limb_geom_ids = self.get_limb_geom_ids(self.model, root_body_name=agent_body_name)
                box_pos = self.data.xpos[box_id]
                points = self.gripper_point_cloud(self.model, self.data, limb_geom_ids)
                _, tri = self.compute_gripper_hull(points)
                grasp_score_aft = self.gripper_box_overlap(box_pos, box_size, tri)
                compactness_score_aft = self.gripper_compactness_score(points, tri, box_pos, box_size)

                force_mag = self.compute_total_contact_force_magnitude(
                    self.model,
                    self.data,
                    limb_geom_ids,
                    box_geom_id
                )
                contact_count = self.compute_box_gripper_contacts(
                    self.model,
                    self.data,
                    limb_geom_ids,
                    box_geom_id
                )
                binary_reward = 1.0 if force_mag >= 5.0 else 0.0
                lift_reward_raw = (box_state_aft[2] - self.box_init_height) / self.dt

                agent_moved, has_box_contact = self._agent_reward_gates(
                    pre_rob_positions[idx],
                    rob_pos_aft,
                    contact_count,
                    force_mag,
                )

                box1_height = float(self.data.xpos[self.box1_body_id][2])
                target_height = 0.5
                if box1_height >= target_height:
                    target_dist_component = (target_dist_bef - target_dist_aft) / self.dt
                else:
                    target_dist_component = 0.0

                distance_component_raw = 0.1 * ((pre_rob_box_dists[idx] - rob_box_dist_aft) / self.dt)
                distance_component = distance_component_raw if agent_moved else 0.0
                grasp_component = 1.0 * grasp_score_aft
                compactness_component = 1.0 * compactness_score_aft
                binary_component = 1.0 * binary_reward
                lift_reward = lift_reward_raw if has_box_contact else 0.0
                lift_component = 5.0 * lift_reward
                reward_components = {
                    'distance': float(distance_component),
                    'grasp': float(grasp_component),
                    'compactness': float(compactness_component),
                    'lift': float(lift_component),
                    'orientation': 0.0,
                    'binary': float(binary_component),
                    'target_dist': float(target_dist_component),
                }

                reward = (
                    distance_component +
                    grasp_component +
                    compactness_component +
                    binary_component +
                    lift_component +
                    1.0 * target_dist_component
                )

                root_body_id = self._get_agent_root_body_id(idx)
                height = float(self.data.xpos[root_body_id][2])
                term_i = not (
                    np.isfinite(s).all()
                    and (height > min_height)
                    and (height < max_height)
                    and (abs(ang) < np.deg2rad(max_ang))
                )
                trunc_i = not (self.agent_control_nsteps[idx] < max_nsteps)
                if self.task_specs.get('mov_goal', False) and box_goal_dist_bef is not None and box_goal_dist_bef < 1.0:
                    trunc_i = True

                ob = self._get_obs(idx)
                info_i = {
                    'use_transform_action': False,
                    'stage': 'execution',
                    'reward_components': reward_components,
                }

                obs[idx] = ob
                agent_rewards[idx] = float(reward)
                agent_infos[idx] = info_i
                processed_agent_ids.append(idx)
                rewards += reward
                termination = term_i
                truncation = trunc_i
                info = info_i

                if termination or truncation:
                    break

            merged_info = dict(info) if isinstance(info, dict) else {}
            merged_info['agent_rewards'] = [agent_rewards[i] for i in processed_agent_ids]
            merged_info['agent_infos'] = [agent_infos[i] for i in processed_agent_ids]
            merged_info['processed_agent_ids'] = processed_agent_ids
            return obs, rewards, termination, truncation, merged_info

        rewards = 0.0
        termination = False
        truncation = False
        info = {}
        obs = [None] * self.num_agents
        agent_rewards = [0.0] * self.num_agents
        agent_infos = [None] * self.num_agents
        processed_agent_ids = []
        for idx in range(self.num_agents):
            ob, reward, termination, truncation, info = self.step(
                agent_actions[idx],
                agent_id=idx
            )
            obs[idx] = ob
            agent_rewards[idx] = float(reward)
            agent_infos[idx] = info
            processed_agent_ids.append(idx)
            rewards += reward

            if termination or truncation:
                break
        merged_info = dict(info) if isinstance(info, dict) else {}
        merged_info['agent_rewards'] = [agent_rewards[i] for i in processed_agent_ids]
        merged_info['agent_infos'] = [agent_infos[i] for i in processed_agent_ids]
        merged_info['processed_agent_ids'] = processed_agent_ids
        return obs, rewards, termination, truncation, merged_info
    
    def step(self, a, agent_id=0):
        self._sync_debug_stage_state(agent_id)
        if not self.is_inited:
            return self._get_obs(agent_id), 0, False, False, {'use_transform_action': False, 'stage': 'execution'}
        stage = self.agent_stages[agent_id]
        if stage == 'execution':
            self.agent_control_nsteps[agent_id] += 1
        else:
            self.agent_stage_steps[agent_id] += 1
        self._sync_debug_stage_state(agent_id)
        # skeleton transform stage
        if stage == 'skeleton_transform':
            skel_a = a[:, -1]
            succ = self.apply_skel_action(skel_a, agent_id=agent_id)
            if not succ:
                return self._get_obs(agent_id), 0.0, True, False, {'use_transform_action': True, 'stage': 'skeleton_transform'}

            if self.agent_stage_steps[agent_id] >= self.cfg.skel_transform_nsteps:
                self.transit_attribute_transform(agent_id=agent_id)

            ob = self._get_obs(agent_id)
            reward = 0.0
            termination = truncation = False
            return ob, reward, termination, truncation, {'use_transform_action': True, 'stage': 'skeleton_transform'}
        # attribute transform stage
        elif stage == 'attribute_transform':
            agent_bodies, agent_indices = self._get_agent_bodies(agent_id)
            design_a = self._align_action_rows(
                a[:, self.control_action_dim:-1],
                len(agent_bodies)
            )
            if self.abs_design:
                design_params = design_a * self.cfg.robot_param_scale
            else:
                design_params = self.design_cur_params[agent_id][agent_indices] + design_a * self.cfg.robot_param_scale
            succ = self.set_design_params(design_params, agent_id=agent_id)
            if not succ:
                return self._get_obs(agent_id), 0.0, True, False, {'use_transform_action': True, 'stage': 'attribute_transform'}

            if self.agent_stage_steps[agent_id] >= self.cfg.skel_transform_nsteps + 1:
                succ = self.transit_execution(agent_id)
                if not succ:
                    return self._get_obs(agent_id), 0.0, True, False, {'use_transform_action': True, 'stage': 'attribute_transform'}

            ob = self._get_obs(agent_id)
            reward = 0.0
            termination = truncation = False
            return ob, reward, termination, truncation, {'use_transform_action': True, 'stage': 'attribute_transform'}
        # execution stage
        else:
            self.control_nsteps = self.agent_control_nsteps[agent_id]
            agent_bodies, _ = self._get_agent_bodies(agent_id)
            a = self._align_action_rows(a, len(agent_bodies))
            control_a = a[:, :self.control_action_dim]
            ctrl = self.action_to_control(control_a, agent_id)
            ctrl_cost_coeff = self.cfg.reward_specs.get('ctrl_cost_coeff', 1e-4)

            self._cache_box_addrs()

            agent_body_name = self.agent_names[agent_id]

            rob_pos_bef = self.get_body_com(agent_body_name)[0:3].copy()
            box_pos_bef = self.get_body_com("box_1")[0:3].copy()
            box_state_bef = self.data.qpos[self.box1_qpos_adr : self.box1_qpos_adr + 7].copy()
            rob_box_dist_bef = np.linalg.norm(rob_pos_bef - box_pos_bef)

            limb_geom_ids = self.get_limb_geom_ids(self.model, root_body_name=agent_body_name)
            # box info
            box_id = self.box1_body_id
            box_pos = self.data.xpos[box_id]

            # IMPORTANT: geom size (half-extents)
            box_geom_id = self.obj_geom_id
            box_size = self.model.geom_size[box_geom_id]
            target_dist_bef = np.linalg.norm(
                self.data.xpos[self.box1_body_id] - self.target_pos
            )
           
            

            points = self.gripper_point_cloud(self.model, self.data, limb_geom_ids)


            _, tri = self.compute_gripper_hull(points)

            # compute overlap
            grasp_score_bef = self.gripper_box_overlap(box_pos, box_size, tri)

            compactness_score_bef = self.gripper_compactness_score(
                points,
                tri,
                box_pos,
                box_size
            )
            

            if self.task_specs.get('mov_goal', False):
                box_goal_dist_bef = np.linalg.norm(box_pos_bef - self.goal_pos)

            

            try:
                self.do_simulation(ctrl, self.frame_skip)
            except:
                print(self.cur_xml_str)
                return self._get_obs(agent_id), 0, True, False, {'use_transform_action': False, 'stage': 'execution'}

            rob_pos_aft = self.get_body_com(agent_body_name)[0:3].copy()
            box_pos_aft = self.get_body_com("box_1")[0:3].copy()
            rob_box_dist_aft = np.linalg.norm(rob_pos_aft - box_pos_aft)
            self.rob_box_dist = rob_pos_aft - box_pos_aft

            box_state_aft = self.data.qpos[self.box1_qpos_adr : self.box1_qpos_adr + 7].copy()
            rob_box_dist_aft = np.linalg.norm(rob_pos_aft - box_state_aft[0:3])
            self.rob_box_dist = rob_pos_aft - box_state_aft[0:3]

            box_pos = self.data.xpos[self.obj_body_id].copy()
            target_dist_aft = np.linalg.norm(
                self.data.xpos[self.box1_body_id] - self.target_pos
            )

            
            

            limb_geom_ids = self.get_limb_geom_ids(self.model, root_body_name=agent_body_name)
            # box info
            box_id = self.box1_body_id
            box_pos = self.data.xpos[box_id]

            # IMPORTANT: geom size (half-extents)
            box_geom_id = self.obj_geom_id
            box_size = self.model.geom_size[box_geom_id]
            gripper_root_body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                agent_body_name
            )

            # Lowest point of box
            box_bottom_z = self.compute_box_lowest_point(
                self.model,
                self.data,
                box_geom_id
            )

            # Distance above floor
            clearance = box_bottom_z - self.box_init_height

            lift_reward_raw = (box_state_aft[2] - self.box_init_height) / self.dt
        
            

            points = self.gripper_point_cloud(self.model, self.data, limb_geom_ids)


            hull, tri = self.compute_gripper_hull(points)

            # compute overlap
            grasp_score_aft = self.gripper_box_overlap(box_pos, box_size, tri)

            compactness_score_aft = self.gripper_compactness_score(
                points,
                tri,
                box_pos,
                box_size
            )


            gripper_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                agent_body_name
            )
            

            gripper_vel = self.data.cvel[gripper_id][:3]
            box_vel = self.data.cvel[box_id][:3]

            relative_vel = box_vel - gripper_vel

            slip_reward = - np.linalg.norm(relative_vel)

            contact_count = self.compute_box_gripper_contacts(
                self.model,
                self.data,
                limb_geom_ids,
                box_geom_id
            )

            force_mag = self.compute_total_contact_force_magnitude(
                self.model,
                self.data,
                limb_geom_ids,
                box_geom_id
            )

            agent_moved, has_box_contact = self._agent_reward_gates(
                rob_pos_bef,
                rob_pos_aft,
                contact_count,
                force_mag,
            )

            fc_score = self.compute_force_closure_reward(
                self.model,
                self.data,
                limb_geom_ids,
                box_geom_id
            )    

            holding_reward = self.compute_holding_reward(
                self.model,
                self.data,
                box_geom_id,
                initial_box_height=0.5,
                root_body_name=agent_body_name
            )
            if contact_count > 0:
                stability_reward = self.compute_stability_reward(
                    self.data,
                    self.box_body_id,
                    gripper_root_body_id
                )
            else:
                stability_reward = 0.0
            

            floor_contact_penalty = 0.0
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if c.geom1 == 0 or c.geom2 == 0:
                    floor_contact_penalty += 1.0

            if force_mag >= 5.0:
                binary_reward = 1.0
            else:
                binary_reward = 0.0

            box1_pos = self.data.xpos[self.box1_body_id]
            box1_height = box1_pos[2]
            target_height = 0.5
            if box1_height >= target_height:
                target_dist = (target_dist_bef - target_dist_aft) / self.dt
            else:
                target_dist = 0.0

            

            distance_component_raw = 0.1 * ((rob_box_dist_bef - rob_box_dist_aft) / self.dt)
            distance_component = distance_component_raw if agent_moved else 0.0
            grasp_component = 1.0 * grasp_score_aft
            compactness_component = 1.0 * compactness_score_aft
            binary_component = 1.0 * binary_reward
            lift_reward = lift_reward_raw if has_box_contact else 0.0
            lift_component = 5.0 * lift_reward
            reward_components = {
                'distance': float(distance_component),
                'grasp': float(grasp_component),
                'compactness': float(compactness_component),
                'lift': float(lift_component),
                'orientation': 0.0,
                'binary': float(binary_component),
                'target_dist': float(target_dist),
            }

            reward = (
                distance_component +
                grasp_component +
                compactness_component +
                binary_component +
                lift_component +
                1.0 * target_dist
            )

            #print("Distance: ", distance_component, " Grasp: ", grasp_component, " Compactness: ", compactness_component, " Force Closure: ", binary_component, " Lift: ", lift_component)
            #print("Total Reward: ", reward, "Box height: ", box_state_aft[2], "Box init height: ", self.box_init_height, "Lift reward: ", lift_reward)
           
            # reward -= ctrl_cost_coeff * np.square(ctrl).mean()
            # reward += self.cfg.reward_specs.get('alive_bonus', 0.0)
            # scale = self.cfg.reward_specs.get('exec_reward_scale', 1.0)
            # reward *= scale
            
            s = self.state_vector()
            # Use the active agent root body height for termination checks.
            # state_vector()[2] can refer to a different generalized coordinate.
            root_body_id = self._get_agent_root_body_id(agent_id)
            height = float(self.data.xpos[root_body_id][2])
            state_vec_height = float(s[2])
            zdir = quaternion_matrix(s[3:7])[:3, 2]
            ang = np.arccos(zdir[2])
            done_condition = self.cfg.done_condition
            min_height = done_condition.get('min_height', 0.0)
            max_height = done_condition.get('max_height', 7.5)
            #max_height = done_condition.get('max_height', 10.0)
            #print("This is the height: ", self.state_vector())
            #height = s[15]
            #max_ang = done_condition.get('max_ang', 3600)
            max_ang = 180
            max_nsteps = done_condition.get('max_nsteps', 1000)
            termination = not (np.isfinite(s).all() and (height > min_height) and (height < max_height) and (abs(ang) < np.deg2rad(max_ang)))
            truncation = not (self.agent_control_nsteps[agent_id] < max_nsteps)
            # if truncation:
            #     print(f'steps: {self.control_nsteps}')
            # if termination:
            #     print(f'termination cause:')
            #     if not (np.isfinite(s).all()):
            #         print('s is not finite: {s}')
            #     elif not (height > min_height):
            #         print(f'height {height} < min_height {min_height}')
            #     elif not (height < max_height):
            #         print(f'height {height} > max_height {max_height}')
            #     elif not (abs(ang) < np.deg2rad(max_ang)):
            #         print(f'ang {abs(ang)} > max_ang {np.deg2rad(max_ang)}')u
            # if self.control_nsteps % 50 == 0:
            #     print(f'box pos: {box_pos_aft} | goal pos: {self.goal_pos} | distance {box_goal_dist_aft}')        
            if self.task_specs.get('mov_goal', False) and box_goal_dist_bef < 1.0:
                truncation = True
            ob = self._get_obs(agent_id)

            #self.print_gripper_body_parts_and_positions()
            #print("Height: ", height, "StateVec Height: ", state_vec_height, "Max Height: ", max_height, "Min height: ", min_height, "Angle: ", ang, "Max Angle: ", max_ang, "Termination: ", termination, "Truncation: ", truncation)
        
            
            return ob, reward, termination, truncation, {
                'use_transform_action': False,
                'stage': 'execution',
                'reward_components': reward_components,
            }
    
    
    def transit_attribute_transform(self, agent_id=0):
        self.agent_stages[agent_id] = 'attribute_transform'
        self._sync_debug_stage_state(agent_id)

    def transit_execution(self, agent_id=0):
        self.agent_stages[agent_id] = 'execution'
        self.agent_control_nsteps[agent_id] = 0
        self._sync_debug_stage_state(agent_id)
        if self._all_agents_in_execution():
            try:
                self.reset_state(True, agent_id=agent_id)
            except:
                print(self.cur_xml_str)
                return False
        self.model.geom_rgba[self.box_id][3] = 1.0
        return True
        

    def if_use_transform_action(self, agent_id=0):
        stage = self.agent_stages[agent_id]
        return ['skeleton_transform', 'attribute_transform', 'execution'].index(stage)
    

    def get_sim_obs(self, agent_id=0):
        rob_obs = []
        env_obs = []
        agent_bodies, _ = self._get_agent_bodies(agent_id)
        if len(agent_bodies) == 0:
            raise RuntimeError(
                f"No robot bodies found for agent_id={agent_id}. "
                "Check agent suffix assignment in xml_robot_multi.Body.reindex/_infer_agent_suffix."
            )

        # Root position (for offsets)
        if 'root_offset' in self.sim_specs:
            root_body_name = self._resolve_model_body_name(agent_bodies[0].name, agent_id=agent_id)
            root_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                root_body_name
            )
            root_pos = self.data.xpos[root_id]

        qvel = self.data.qvel.copy()
        if self.clip_qvel:
            qvel = np.clip(qvel, -10, 10)

        # =========================
        # Robot observations
        # =========================
        for i, body in enumerate(agent_bodies):
            model_body_name = self._resolve_model_body_name(body.name, agent_id=agent_id)

            if i == 0:
                root_qpos, root_qvel = self._get_agent_root_obs_window(agent_id, qpos_len=6, qvel_len=7)
                # root body
                """
                obs_i = [
                    self.data.qpos[2:7],
                    qvel[:6],
                    np.zeros(2)
                ]
                """
                obs_i = [
                    root_qpos,
                    root_qvel,
                ]
            else:
                qs, qe = get_single_body_qposaddr(self.model, model_body_name)

                # 🚨 CRITICAL SAFETY CHECK (prevents segfault)
                if qs < 0 or qe > self.model.nq or qs > qe:
                    print("\n🚨 INVALID QPOS RANGE 🚨")
                    print(f"Body: {body.name}")
                    print(f"qs: {qs}, qe: {qe}, nq: {self.model.nq}")
                    raise RuntimeError("Bad qpos indexing")

                # 🚨 QVEL CHECK
                if (qs - 1) < 0 or (qe - 1) > self.model.nv:
                    print("\n🚨 INVALID QVEL RANGE 🚨")
                    print(f"Body: {body.name}")
                    print(f"qs: {qs}, qe: {qe}, nv: {self.model.nv}")
                    raise RuntimeError("Bad qvel indexing")

                if qe - qs >= 1:
                    # NOTE: your assumption is 1 DOF joints
                    if (qe - qs) != 1:
                        print(f"⚠️ Unexpected joint dim for body {body.name}: {qe - qs}")

                    obs_i = [
                        np.zeros(11),
                        self.data.qpos[qs:qe],
                        qvel[qs - 1:qe - 1]
                    ]
                else:
                    obs_i = [np.zeros(13)]

            # Root offset feature
            if 'root_offset' in self.sim_specs:
                body_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    model_body_name
                )
                offset = self.data.xpos[body_id][[0, 2]] - root_pos[[0, 2]]
                obs_i.append(offset)

            obs_i = np.concatenate(obs_i)
            rob_obs.append(obs_i)

        rob_obs = np.stack(rob_obs)

        # =========================
        # Environment observations
        # =========================
        self._cache_box_addrs()

        box1_pos = self.data.qpos[self.box1_qpos_adr:self.box1_qpos_adr + 3]
        box1_quat = self.data.qpos[self.box1_qpos_adr + 3:self.box1_qpos_adr + 7]
        box1_vel = qvel[self.box1_qvel_adr:self.box1_qvel_adr + 6]

        box2_pos = self.data.qpos[self.box2_qpos_adr:self.box2_qpos_adr + 3]
        box2_quat = self.data.qpos[self.box2_qpos_adr + 3:self.box2_qpos_adr + 7]
        box2_vel = qvel[self.box2_qvel_adr:self.box2_qvel_adr + 6]

        gripper_pos = self.get_body_com(self.agent_names[agent_id])[:3]
        rob_box1_dist = box1_pos - gripper_pos
        rob_box2_dist = box2_pos - gripper_pos
        box12_dist = box2_pos - box1_pos

        for i in range(rob_obs.shape[0]):
            if i == 0:
                obs_i = [
                    rob_box1_dist,
                    rob_box2_dist,
                    box12_dist,
                    box1_pos,
                    box1_quat,
                    box1_vel,
                    box2_pos,
                    box2_quat,
                    box2_vel,
                ]
            else:
                obs_i = [np.zeros(35)]

            obs_i = np.concatenate(obs_i)
            env_obs.append(obs_i)

        env_obs = np.stack(env_obs)

        # Final observation
        obs = np.concatenate([rob_obs, env_obs], axis=-1)
        return obs
    def get_attr_fixed(self, agent_id=0):
        obs = []
        agent_bodies, _ = self._get_agent_bodies(agent_id)
        if len(agent_bodies) == 0:
            raise RuntimeError(
                f"No robot bodies found for agent_id={agent_id}. "
                "Check agent suffix assignment in xml_robot_multi.Body.reindex/_infer_agent_suffix."
            )

        for i, body in enumerate(agent_bodies):
            obs_i = []
            if 'depth' in self.attr_specs:
                obs_depth = np.zeros(self.cfg.max_body_depth)
                obs_depth[body.depth] = 1.0
                obs_i.append(obs_depth)
            if 'jrange' in self.attr_specs:
                obs_jrange = body.get_joint_range()
                obs_i.append(obs_jrange)
            if 'skel' in self.attr_specs:
                obs_add = self.allow_add_body(body)
                obs_rm = self.allow_remove_body(body)
                obs_i.append(np.array([float(obs_add), float(obs_rm)]))
            if len(obs_i) > 0:
                obs_i = np.concatenate(obs_i)
                obs.append(obs_i)
        
        if len(obs) == 0:
            return None
        obs = np.stack(obs)
        return obs

    """
    def get_attr_design(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = body.get_params([], pad_zeros=True, demap_params=True)
            obs.append(obs_i)
        obs = np.stack(obs)
        return obs
    """
    def get_attr_design(self):
        obs = []

        # First pass: collect all observations
        for i, body in enumerate(self.robot.bodies):
            obs_i = body.get_params([], pad_zeros=True, demap_params=True)
            obs.append(obs_i)

        # Find maximum length
        max_len = max(o.shape[0] for o in obs)

        # Second pass: pad to same length
        obs_padded = []
        for i, o in enumerate(obs):
            if o.shape[0] < max_len:
                pad_width = max_len - o.shape[0]
                o = np.concatenate([o, np.zeros(pad_width)])
            obs_padded.append(o)

        # Stack safely
        obs = np.stack(obs_padded)

        return obs

    def get_body_index(self, agent_id=None):
        if agent_id is None:
            bodies = self.robot.bodies
        else:
            bodies, _ = self._get_agent_bodies(agent_id)
        # Use compact local indices per observation to keep embedding inputs small.
        # This avoids large hierarchical base-5 IDs when morphology grows deeper.
        index = np.arange(len(bodies), dtype=np.int64)
        return index

    def get_body_height(self, agent_id=None):
        if agent_id is None:
            bodies = self.robot.bodies
        else:
            bodies, _ = self._get_agent_bodies(agent_id)
        heights = []
        for i, body in enumerate(bodies):
            h = body.height
            heights.append(h)
        heights = np.array(heights)
        return heights
        
    def get_body_depth(self, agent_id=None):
        if agent_id is None:
            bodies = self.robot.bodies
        else:
            bodies, _ = self._get_agent_bodies(agent_id)
        depths = []
        for i, body in enumerate(bodies):
            d = body.depth
            depths.append(d)
        depths = np.array(depths)
        return depths

    def _get_obs(self, agent_id=0):
        agent_body_indices = self._get_agent_body_indices(agent_id)

        attr_fixed_obs = self.get_attr_fixed(agent_id=agent_id)
        sim_obs = self.get_sim_obs(agent_id=agent_id)

        if self.design_cur_params.ndim == 3:
            design_obs = self.design_cur_params[agent_id]
        else:
            design_obs = self.design_cur_params

        if len(agent_body_indices) > 0 and design_obs.shape[0] >= max(agent_body_indices) + 1:
            design_obs = design_obs[agent_body_indices]

        obs = np.concatenate(list(filter(lambda x: x is not None, [attr_fixed_obs, sim_obs, design_obs])), axis=-1)
        if self.cfg.obs_specs.get('fc_graph', False):
            edges = get_graph_fc_edges(sim_obs.shape[0])
        else:
            full_edges = self.robot.get_gnn_edges()
            if len(agent_body_indices) > 0:
                local_index = {old_i: new_i for new_i, old_i in enumerate(agent_body_indices)}
                edge_list = []
                for src, dst in full_edges.T:
                    if src in local_index and dst in local_index:
                        edge_list.append([local_index[src], local_index[dst]])
                if len(edge_list) > 0:
                    edges = np.asarray(edge_list, dtype=np.int64).T
                else:
                    edges = np.zeros((2, 0), dtype=np.int64)
            else:
                edges = full_edges
        use_transform_action = np.array([self.if_use_transform_action(agent_id=agent_id)])
        num_nodes = np.array([sim_obs.shape[0]])
        all_obs = [obs, edges, use_transform_action, num_nodes]
        if self.use_body_ind:
            body_index = self.get_body_index(agent_id=agent_id)
            all_obs.append(body_index)
        if self.use_body_depth_height:
            body_depths = self.get_body_depth(agent_id=agent_id)
            all_obs.append(body_depths)
            body_heights = self.get_body_height(agent_id=agent_id)
            all_obs.append(body_heights)
        if self.use_shortest_distance:
            distances = self.robot.get_shortest_distances()
            if len(agent_body_indices) > 0:
                distances = distances[np.ix_(agent_body_indices, agent_body_indices)]
            all_obs.append(distances)
        if self.use_position_encoding:
            lapPE = self.robot.get_laplacian_position_encoding(robust=True)
            if len(agent_body_indices) > 0:
                lapPE = lapPE[agent_body_indices]
            all_obs.append(lapPE)
        return all_obs

    def _estimate_box_xy_half_extent(self, geom_id):
        geom_type = self.model.geom_type[geom_id]
        geom_size = self.model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return float(max(geom_size[0], geom_size[1]))
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(geom_size[0])
        if geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            return float(geom_size[0])

        return 0.0

    def _get_wall_spawn_bounds(self):
        if self.box_spawn_range is not None:
            bounds = np.array(self.box_spawn_range, dtype=np.float64).flatten()
            if bounds.shape[0] == 4:
                return bounds[0], bounds[1], bounds[2], bounds[3]

        wall_names = ["wall_left", "wall_right", "wall_back", "wall_front"]
        wall_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in wall_names
        }

        if any(gid < 0 for gid in wall_ids.values()):
            return None

        left_inner = self.model.geom_pos[wall_ids["wall_left"]][0] + self.model.geom_size[wall_ids["wall_left"]][0]
        right_inner = self.model.geom_pos[wall_ids["wall_right"]][0] - self.model.geom_size[wall_ids["wall_right"]][0]
        back_inner = self.model.geom_pos[wall_ids["wall_back"]][1] + self.model.geom_size[wall_ids["wall_back"]][1]
        front_inner = self.model.geom_pos[wall_ids["wall_front"]][1] - self.model.geom_size[wall_ids["wall_front"]][1]
        return left_inner, right_inner, back_inner, front_inner

    def _sample_two_box_spawn_positions(self):
        default_box1 = np.array(self.task_specs.get("box_1_pos", [0.0, 0.0, 0.5]), dtype=np.float64)
        default_box2 = np.array(self.task_specs.get("box_2_pos", [1.5, 0.0, 0.5]), dtype=np.float64)

        if not self.random_boxes_pos:
            return default_box1, default_box2

        bounds = self._get_wall_spawn_bounds()
        if bounds is None:
            return default_box1, default_box2

        x_min, x_max, y_min, y_max = bounds

        box1_rad = self._estimate_box_xy_half_extent(self.box1_geom_id)
        box2_rad = self._estimate_box_xy_half_extent(self.box2_geom_id)
        shared_margin = max(0.0, self.box_random_margin)

        margin1 = shared_margin + box1_rad
        margin2 = shared_margin + box2_rad

        low1 = np.array([x_min + margin1, y_min + margin1], dtype=np.float64)
        high1 = np.array([x_max - margin1, y_max - margin1], dtype=np.float64)
        low2 = np.array([x_min + margin2, y_min + margin2], dtype=np.float64)
        high2 = np.array([x_max - margin2, y_max - margin2], dtype=np.float64)

        if np.any(low1 >= high1) or np.any(low2 >= high2):
            return default_box1, default_box2

        min_center_dist = box1_rad + box2_rad + max(0.0, self.box_pair_clearance)
        max_tries = max(1, self.box_spawn_max_tries)

        for _ in range(max_tries):
            p1 = self.np_random.uniform(low=low1, high=high1)
            p2 = self.np_random.uniform(low=low2, high=high2)
            if np.linalg.norm(p1 - p2) >= min_center_dist:
                box1 = default_box1.copy()
                box2 = default_box2.copy()
                box1[:2] = p1
                box2[:2] = p2
                return box1, box2

        return default_box1, default_box2

    
    def reset_state(self, add_noise, agent_id=0):
        # Always start from XML default initial state.
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        if add_noise:
            qpos += self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
            qvel += self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)

        box1_pos, box2_pos = self._sample_two_box_spawn_positions()

        qpos[self.box1_qpos_adr:self.box1_qpos_adr + 3] = box1_pos
        qpos[self.box1_qpos_adr + 3:self.box1_qpos_adr + 7] = np.array([1, 0, 0, 0])

        qpos[self.box2_qpos_adr:self.box2_qpos_adr + 3] = box2_pos
        qpos[self.box2_qpos_adr + 3:self.box2_qpos_adr + 7] = np.array([1, 0, 0, 0])

        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

        # Keep the cached box position aligned with the actual simulator state.
        self.box_pos = self.data.qpos[self.box1_qpos_adr:self.box1_qpos_adr + 3].copy()

        if self.task_specs.get('mov_goal', False):
            if self.task_specs.get('random_goal'):
                self.goal_pos[:2] = rand_coord(self.np_random, np.array(self.task_specs.get('goal_range')))
            self.box_goal_dist = self.box_pos - self.goal_pos

      

        root_body_id = self._get_agent_root_body_id(agent_id)
        rob_pos = self.data.xpos[root_body_id].copy()[0:3]
        self.rob_box_dist = rob_pos - self.box_pos

    def reset_robot(self):
        del self.robot
        self.robot = Robot(self.cfg.robot_cfg, xml=self.init_xml_str, is_xml_str=True)
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        xml_str_fixed = self.cur_xml_str.replace(' center="0 0 0"', '')
        self.reload_sim_model(xml_str_fixed)
        self.cur_xml_str = xml_str_fixed
        self._cache_box_addrs()
        self._configure_gripper_collision_masks()
        self.box_id = self.box1_geom_id
        self.obj_body_id = self.box1_body_id
        self.target_body_id = self.box2_body_id
        self.obj_geom_id = self.box1_geom_id
        self.target_geom_id = self.box2_geom_id
        self.box_body_id = self.box1_body_id
        self.box_geom_id = self.box1_geom_id
        self.box_qpos_adr = self.box1_qpos_adr
        self.box_qvel_adr = self.box1_qvel_adr
        self.box_init_height = self.model.geom_size[self.box1_geom_id][2]
        self.target_pos = self.data.xpos[self.box2_body_id].copy()
        #self.reload_sim_model(self.cur_xml_str)
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self._tile_design_params(self.design_ref_params)

    def reset_model(self):
        
        self.reset_robot()
        
        self.agent_stages = ['skeleton_transform'] * self.num_agents
        self.agent_stage_steps = [0] * self.num_agents
        self.agent_control_nsteps = [0] * self.num_agents
        self.control_nsteps = 0
        self.stage = 'skeleton_transform'
        self.cur_t = 0
        self.reset_state(False)
        return self._get_obs(0)

    def viewer_setup(self):
        """
        # self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = 10
        # self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.lookat[:2] = self.data.qpos[:2] 
        self.viewer.cam.elevation = -10
        self.viewer.cam.azimuth = 110
        """         
        if self.viewer is not None:
            cam = self.viewer.cam
            cam.distance = 10
            cam.lookat[0] = self.data.qpos[0]
            cam.lookat[1] = self.data.qpos[1]
            cam.elevation = -10
            cam.azimuth = 110                        
    
    def _cache_box_addrs(self):
        self.box1_joint_id = self.model.joint("box_1_joint").id
        self.box2_joint_id = self.model.joint("box_2_joint").id

        self.box1_qpos_adr = self.model.jnt_qposadr[self.box1_joint_id]
        self.box1_qvel_adr = self.model.jnt_dofadr[self.box1_joint_id]
        self.box2_qpos_adr = self.model.jnt_qposadr[self.box2_joint_id]
        self.box2_qvel_adr = self.model.jnt_dofadr[self.box2_joint_id]

        self.box1_geom_id = self.model.geom("box_1").id
        self.box2_geom_id = self.model.geom("box_2").id
        self.box1_body_id = self.model.body("box_1").id
        self.box2_body_id = self.model.body("box_2").id

        # Single-box aliases for existing code paths.
        self.box_joint_id = self.box1_joint_id
        self.box_qpos_adr = self.box1_qpos_adr
        self.box_qvel_adr = self.box1_qvel_adr
        self.box_geom_id = self.box1_geom_id
        self.box_body_id = self.box1_body_id

    def _configure_gripper_collision_masks(self):
        # Enable capsule-capsule contacts so the two grippers cannot pass through each other.
        for geom_id in range(self.model.ngeom):
            if self.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                continue
            self.model.geom_contype[geom_id] = 1
            self.model.geom_conaffinity[geom_id] = 1

    def print_gripper_body_parts_and_positions(self):
        """
        Print all body parts and their positions for each gripper (agent).
        """
        print("\n" + "="*80)
        print("GRIPPER BODY PARTS AND POSITIONS")
        print("="*80)
        
        for agent_id in range(self.num_agents):
            print(f"\n--- GRIPPER {agent_id + 1} ---")
            
            # Get all bodies for this agent
            agent_bodies, body_indices = self._get_agent_bodies(agent_id)
            
            print(f"Total bodies: {len(agent_bodies)}\n")
            
            for i, body in enumerate(agent_bodies):
                # Resolve the actual model body name
                try:
                    model_body_name = self._resolve_model_body_name(body.name, agent_id=agent_id)
                except:
                    model_body_name = body.name
                
                # Get body ID in MuJoCo
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, model_body_name)
                
                if body_id == -1:
                    print(f"  [{i}] {body.name} (model: {model_body_name}) - NOT FOUND in MuJoCo")
                    continue
                
                # Get position
                position = self.data.xpos[body_id].copy()
                
                # Get rotation (quaternion)
                xmat = self.data.xmat[body_id].reshape(3, 3)
                
                # Get velocity
                velocity = self.data.cvel[body_id].copy()  # [angular_vel, linear_vel]

                joint_types = [joint.type for joint in getattr(body, "joints", [])]
                geom_types = [geom.type for geom in getattr(body, "geoms", [])]
                body_type_parts = []
                if joint_types:
                    body_type_parts.append(f"joints={', '.join(joint_types)}")
                if geom_types:
                    body_type_parts.append(f"geoms={', '.join(geom_types)}")
                body_type = " | ".join(body_type_parts) if body_type_parts else "unknown"
                
                print(f"  [{i}] Name: {body.name}")
                print(f"      Model name: {model_body_name}")
                print(f"      Type: {body_type}")
                print(f"      Depth: {body.depth}")
                print(f"      Position (x, y, z): [{position[0]:8.4f}, {position[1]:8.4f}, {position[2]:8.4f}]")
                print(f"      Angular velocity: [{velocity[0]:8.4f}, {velocity[1]:8.4f}, {velocity[2]:8.4f}]")
                print(f"      Linear velocity:  [{velocity[3]:8.4f}, {velocity[4]:8.4f}, {velocity[5]:8.4f}]")
                
                # Print geoms in this body
                geom_start = self.model.body_geomadr[body_id]
                geom_num = self.model.body_geomnum[body_id]
                if geom_num > 0:
                    print(f"      Geoms ({geom_num}):")
                    for j in range(geom_num):
                        geom_id = geom_start + j
                        geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                        geom_pos = self.data.geom_xpos[geom_id]
                        print(f"        - {geom_name}: [{geom_pos[0]:8.4f}, {geom_pos[1]:8.4f}, {geom_pos[2]:8.4f}]")
                print()
        
        print("="*80 + "\n")