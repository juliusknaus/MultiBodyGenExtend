import numpy as np
from gym import utils
from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv
from khrylib.robot.xml_robot import Robot
from khrylib.utils import get_single_body_qposaddr, get_graph_fc_edges
from khrylib.utils.transformation import quaternion_matrix
from copy import deepcopy
import mujoco
import time
from gym.spaces import Box
import os
from scipy.spatial import ConvexHull, Delaunay


def rand_coord(np_random, coord_range):
    coord_range = np.array(coord_range, dtype=np.float64)
    if coord_range.shape == (2, 2):
        low = coord_range[:, 0]
        high = coord_range[:, 1]
    elif coord_range.shape == (4,):
        low = np.array([coord_range[0], coord_range[2]], dtype=np.float64)
        high = np.array([coord_range[1], coord_range[3]], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported coord_range shape: {coord_range.shape}")
    return np_random.uniform(low=low, high=high)


class DexGripperHoldingEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, cfg, agent):
        self.cur_t = 0
        self.cfg = cfg
        self.env_specs = cfg.env_specs
        self.task_specs = cfg.task_specs
        self.obj_name = self.task_specs.get("obj_name", "box")
        self.agent = agent
        if self.cfg.xml_name == "default":
            self.model_xml_file = os.path.join(cfg.project_path, "assets", "mujoco_envs", "dex_gripper_holding.xml")
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
        self.random_box_pos = bool(self.task_specs.get('random_box_pos', False))
        self.box_random_margin = float(self.task_specs.get('box_random_margin', 0.02))
        self.box_spawn_range = self.task_specs.get('box_spawn_range', None)
        self.rob_box_dist = np.array([0.0, 0.0, 0.0])
        if self.task_specs.get('mov_goal', False):
            self.goal_pos = np.array(self.task_specs.get('goal_pos'))
            self.box_goal_dist = self.box_pos - self.goal_pos
        MujocoEnv.__init__(self, self.model_xml_file, 4)
        utils.EzPickle.__init__(self)
        self.box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self._cache_root_actuator_ids()
        # One control channel per hinge body; root gets 3 channels only when root motors exist.
        self.control_action_dim = 3 if len(self.root_actuator_ids) == 3 else 1
        self.prev_action_dim = int(self.task_specs.get('prev_action_dim', self.model.nu))
        self.prev_action = np.zeros(self.prev_action_dim, dtype=np.float64)
        self.num_fingertips_obs = int(self.task_specs.get('num_fingertips_obs', 4))
        self.fingertip_body_names = self.task_specs.get('fingertip_bodies', None)
        self._cache_box_addrs()
        self.skel_num_action = 3 if cfg.enable_remove else 2
        self.sim_obs_dim = self.get_sim_obs().shape[-1]
        self.attr_fixed_dim = self.get_attr_fixed().shape[-1]
        self.weights = cfg.task_specs.get('weights', {})
        self.inhand_yaw_accum = 0.0
        
        # Andrychowitz et al. 2019-style orientation task setup.
        self.random_goal_quat = bool(self.task_specs.get('random_goal_orientation', True))
        goal_orient_cfg = self.task_specs.get('goal_orientation', [1.0, 0.0, 0.0, 0.0])
        self.goal_quat = np.array(goal_orient_cfg, dtype=np.float64)
        goal_norm = np.linalg.norm(self.goal_quat)
        if goal_norm < 1e-8:
            self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.goal_quat = self.goal_quat / goal_norm
        

        # Paper-style thresholds and bonuses.
        self.orientation_success_threshold = float(self.task_specs.get('orientation_success_threshold', 0.4))
        self.orientation_success_bonus = float(self.task_specs.get('orientation_success_bonus', 15.0))
        self.drop_penalty = float(self.task_specs.get('drop_penalty', 20.0))
        self.drop_height_margin = float(self.task_specs.get('drop_height_margin', 0.005))
        self.binary_overlap_threshold = float(self.task_specs.get('binary_overlap_threshold', 0.5))
        self.binary_overlap_reward = float(self.task_specs.get('binary_overlap_reward', 1.0))
        self.goal_reached_count = 0
        
        print(self.task_specs)
        print("Weights for reward components: ", self.weights)
        print(
            "Andrychowitz reward:",
            f"random_goal={self.random_goal_quat},",
            f"success_thresh={self.orientation_success_threshold:.3f}rad,",
            f"success_bonus={self.orientation_success_bonus},",
            f"drop_penalty={self.drop_penalty}"
        )


        self.geom_id_to_body_id = {
            gid: self.model.geom_bodyid[gid]
            for gid in range(self.model.ngeom)
        }
        

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "0")
        self.box_qpos_addr = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "box"
        )
        self.box_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "box"
        )
        #self.box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box")

        # geom id (first geom attached to that body)
        geom_start = self.model.body_geomadr[self.box_body_id]
        if self.obj_name == 'sphere' or self.obj_name == 'cylinder' or self.obj_name == 'capsule' or self.obj_name == 'box':
            geom_num   = self.model.body_geomnum[self.box_body_id]

            if geom_num == 0:
                raise RuntimeError("Box body has no geom!")

            self.box_geom_id = geom_start  # first geom
        else: 
            geom_num = self.model.body_geomnum[self.box_body_id]

            self.box_geom_id = list(
                range(geom_start, geom_start + geom_num)
            )



        # size (half extents for box)
        if self.obj_name == 'cup':
            self.box_init_height = self.data.geom_xpos[self.box_geom_id][0][2]
        else:
            self.box_init_height = self.model.geom_size[self.box_geom_id][2]
        self.box_qpos_adr = self.model.jnt_qposadr[self.box_qpos_addr]
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                i
            )

            qpos_adr = self.model.jnt_qposadr[i]
            qvel_adr = self.model.jnt_dofadr[i]
            jtype = self.model.jnt_type[i]

            print(f"Joint {i}: {name}")
            print(f"  qpos index starts at: {qpos_adr}")
            print(f"  qvel index starts at: {qvel_adr}")
            print(f"  joint type: {jtype}")  # 0=free, 1=ball, 2=slide, 3=hinge
        geom_start = self.model.body_geomadr[body_id]
        geom_num = self.model.body_geomnum[body_id]

        if geom_num > 0:
            geom_id = geom_start
            self.torso_radii = self.model.geom_size[geom_id]
        
        self.goal_quaternions = [
            np.array([1.0, 0.0, 0.0, 0.0]),                    # 0°
            np.array([0.9239, 0.0, 0.0, 0.3827]),             # 45°
            np.array([0.7071, 0.0, 0.0, 0.7071]),             # 90°
            np.array([0.3827, 0.0, 0.0, 0.9239]),             # 135°
            np.array([0.0, 0.0, 0.0, 1.0]),                   # 180°
            np.array([-0.3827, 0.0, 0.0, 0.9239]),            # 225°
            np.array([-0.7071, 0.0, 0.0, 0.7071]),            # 270°
            np.array([-0.9239, 0.0, 0.0, 0.3827]),            # 315°
        ]

        self.goal_index = 0

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

    def apply_skel_action(self, skel_action):
        bodies = list(self.robot.bodies)
        for body, a in zip(bodies, skel_action):
            if a == 1 and self.allow_add_body(body):
                self.robot.add_child_to_body(body)
            if a == 2 and self.allow_remove_body(body):
                self.robot.remove_body(body)

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            xml_str_fixed = self.cur_xml_str.replace(' center="0 0 0"', '')
            self.reload_sim_model(xml_str_fixed)
            self.cur_xml_str = xml_str_fixed
            self._cache_root_actuator_ids()
            self._cache_box_addrs()
            #self.reload_sim_model(xml_str.decode('utf-8'))
        except:
            print(self.cur_xml_str)
            return False      
        self.design_cur_params = self.get_attr_design()
        return True

    def set_design_params(self, in_design_params):
        design_params = in_design_params
        for params, body in zip(design_params, self.robot.bodies):
            body.set_params(params, pad_zeros=True, map_params=True)
            body.sync_node()

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            xml_str_fixed = self.cur_xml_str.replace(' center="0 0 0"', '')
            self.reload_sim_model(xml_str_fixed)
            self.cur_xml_str = xml_str_fixed
            self._cache_root_actuator_ids()
            self._cache_box_addrs()
            #self.reload_sim_model(xml_str.decode('utf-8'))
        except:
            print(self.cur_xml_str)
            return False
        if self.use_projected_params:
            self.design_cur_params = self.get_attr_design()
        else:
            self.design_cur_params = in_design_params.copy()
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
        hull = ConvexHull(points)
        tri = Delaunay(points[hull.vertices])
        return hull, tri

    """
    def sample_points_in_box(self, center, size, n_samples=500):
        
        half = size

        samples = np.random.uniform(-1, 1, size=(n_samples, 3)) * half
        return samples + center
    """

    def sample_points_in_object(self, center, geom_id, n_samples=500):
        geom_type = self.model.geom_type[geom_id]
        size = self.model.geom_size[geom_id]
        if self.obj_name == 'sphere':
            r = size[0]
            samples = np.random.normal(size=(n_samples, 3))
            samples /= np.linalg.norm(samples, axis=1, keepdims=True) + 1e-8
            samples *= np.random.uniform(0, r, size=(n_samples, 1))
            return samples + center

        elif self.obj_name == 'box':
            half = size
            return np.random.uniform(-1, 1, (n_samples, 3)) * half + center
        
        elif self.obj_name == 'cylinder':
            radius = size[0]
            half_height = size[1]

            z = np.random.uniform(-half_height, half_height, (n_samples, 1))

            theta = np.random.uniform(0, 2*np.pi, (n_samples, 1))
            r = np.sqrt(np.random.uniform(0, 1, (n_samples, 1))) * radius

            x = r * np.cos(theta)
            y = r * np.sin(theta)

            samples = np.hstack([x, y, z])
            return samples + center

        # -----------------------
        # CAPSULE
        # -----------------------
        elif self.obj_name == 'capsule':
            radius = size[0]
            half_height = size[1]

            samples = []

            for _ in range(n_samples):
                z = np.random.uniform(-half_height, half_height)

                # cylinder region
                theta = np.random.uniform(0, 2*np.pi)
                r = np.sqrt(np.random.uniform(0, 1)) * radius

                x = r * np.cos(theta)
                y = r * np.sin(theta)

                samples.append([x, y, z])

            return np.array(samples) + center
        elif self.obj_name == 'prism':
            # Analytic triangular prism proxy, aligned with local z-axis.
            # size = [hx, hy, hz] interpreted as half extents.
            hx, hy, hz = size

            # Sample uniformly inside a triangle using barycentric folding.
            u = np.random.rand(n_samples, 2)
            swap = (u[:, 0] + u[:, 1]) > 1.0
            u[swap] = 1.0 - u[swap]

            # Triangle vertices in local xy
            v0 = np.array([-hx, -hy])
            v1 = np.array([ hx, -hy])
            v2 = np.array([ 0.0,  hy])

            xy = (
                v0
                + u[:, [0]] * (v1 - v0)
                + u[:, [1]] * (v2 - v0)
            )

            z = np.random.uniform(-hz, hz, size=(n_samples, 1))
            samples = np.hstack([xy, z])
            return samples + center

        elif self.obj_name == 'cup':
            # Composite object: sample from all geoms on the cup body.
            # This covers bottom, walls, and optional capsule handle.
            if not hasattr(self, "obj_geom_ids"):
                geom_start = self.model.body_geomadr[self.box_body_id]
                geom_num = self.model.body_geomnum[self.box_body_id]
                self.obj_geom_ids = [geom_start + i for i in range(geom_num)]

            # Simple mixture over all geoms
            probs = np.ones(len(self.obj_geom_ids)) / len(self.obj_geom_ids)
            counts = np.random.multinomial(n_samples, probs)

            all_samples = []

            for gid, k in zip(self.obj_geom_ids, counts):
                if k == 0:
                    continue

                gtype = self.model.geom_type[gid]
                gsize = self.model.geom_size[gid]
                gcenter = self.data.geom_xpos[gid].copy()
                R = self.data.geom_xmat[gid].reshape(3, 3).copy()

                if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                    r = gsize[0]
                    pts = np.random.normal(size=(k, 3))
                    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
                    pts *= np.random.uniform(0, r, size=(k, 1))

                elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                    half = gsize
                    pts = np.random.uniform(-1, 1, size=(k, 3)) * half

                elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    radius = gsize[0]
                    half_h = gsize[1]
                    z = np.random.uniform(-half_h, half_h, size=(k, 1))
                    theta = np.random.uniform(0, 2 * np.pi, size=(k, 1))
                    rr = np.sqrt(np.random.uniform(0, 1, size=(k, 1))) * radius
                    x = rr * np.cos(theta)
                    y = rr * np.sin(theta)
                    pts = np.hstack([x, y, z])

                elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    radius = gsize[0]
                    half_h = gsize[1]
                    z = np.random.uniform(-half_h, half_h, size=(k, 1))
                    theta = np.random.uniform(0, 2 * np.pi, size=(k, 1))
                    rr = np.sqrt(np.random.uniform(0, 1, size=(k, 1))) * radius
                    x = rr * np.cos(theta)
                    y = rr * np.sin(theta)
                    pts = np.hstack([x, y, z])

                else:
                    continue

                # local geom frame -> world frame
                pts_world = (pts @ R.T) + gcenter
                all_samples.append(pts_world)

            if len(all_samples) == 0:
                return np.zeros((0, 3))

            return np.vstack(all_samples)

        else:
            raise NotImplementedError("Unsupported geom type")
    def gripper_box_overlap(self, box_pos, box_size, tri, n_samples=500):
        """
        Returns fraction of sampled box volume inside gripper convex hull.
        """

        #samples = self.sample_points_in_box(box_pos, box_size, n_samples)
        samples = self.sample_points_in_object(box_pos, self.box_geom_id, n_samples)

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

    def has_box_plane_contact(self, model=None, data=None, plane_geom_id=0):
        """
        Returns True if any box geom is currently in contact with the ground plane.
        """
        model = self.model if model is None else model
        data = self.data if data is None else data

        if isinstance(self.box_geom_id, (list, tuple, np.ndarray)):
            box_geom_ids = set(self.box_geom_id)
        else:
            box_geom_ids = {self.box_geom_id}

        for i in range(data.ncon):
            c = data.contact[i]
            if (
                (c.geom1 in box_geom_ids and c.geom2 == plane_geom_id)
                or
                (c.geom2 in box_geom_ids and c.geom1 == plane_geom_id)
            ):
                return True

        return False

    """
    def compute_box_lowest_point(
        self,
        model,
        data,
        geom_id
    ):

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
    """

    def compute_object_lowest_point(self, geom_id):
        center = self.data.geom_xpos[geom_id]
        geom_type = self.model.geom_type[geom_id]
        size = self.model.geom_size[geom_id]

        if self.obj_name == 'sphere':
            r = size[0]
            return center[2] - r

        elif self.obj_name == 'box':
            R = self.data.geom_xmat[geom_id].reshape(3, 3)
            half = size

            corners = np.array([
                [sx, sy, sz]
                for sx in (-half[0], half[0])
                for sy in (-half[1], half[1])
                for sz in (-half[2], half[2])
            ])

            corners_world = (corners @ R.T) + center
            return np.min(corners_world[:, 2])
        elif self.obj_name == 'cylinder':
            R = self.data.geom_xmat[geom_id].reshape(3, 3)
            radius, half_h = size[0], size[1]

            # extreme points in local frame
            # we take 4 cardinal directions for circle + top/bottom
            circle_dirs = np.array([
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
            ])

            points = []

            for z in (-half_h, half_h):
                for d in circle_dirs:
                    p_local = np.array([radius * d[0], radius * d[1], z])
                    points.append(p_local)

            points = np.array(points)

            # rotate + translate
            points_world = (points @ R.T) + center

            return np.min(points_world[:, 2])

        elif self.obj_name == 'capsule':
            R = self.data.geom_xmat[geom_id].reshape(3, 3)
            radius, half_h = size[0], size[1]

            points = []

            # cylinder side extremes
            circle_dirs = np.array([
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
            ])

            for z in (-half_h, half_h):
                for d in circle_dirs:
                    points.append([radius * d[0], radius * d[1], z])

            # hemispherical caps (bottom/top tips)
            points.append([0, 0, -(half_h + radius)])
            points.append([0, 0, (half_h + radius)])

            points = np.array(points)

            points_world = (points @ R.T) + center

            return np.min(points_world[:, 2])
        elif self.obj_name == 'prism':
            # Triangular prism proxy, aligned with local z-axis.
            # Use extreme points, rotate, then take min z.
            R = self.data.geom_xmat[geom_id].reshape(3, 3)

            hx, hy, hz = size

            # 6 vertices of the triangular prism
            # triangle in xy extruded to +/- hz in z
            verts_local = np.array([
                [-hx, -hy, -hz],
                [ hx, -hy, -hz],
                [ 0.0,  hy, -hz],
                [-hx, -hy,  hz],
                [ hx, -hy,  hz],
                [ 0.0,  hy,  hz],
            ])

            verts_world = (verts_local @ R.T) + center
            return np.min(verts_world[:, 2])

        elif self.obj_name == "cup":
            # -----------------------------
            # Cup parameters (tune these!)
            # -----------------------------
            R_o = self.cfg.task_specs.get("cup_radius", 0.04)      # outer radius
            H   = self.cfg.task_specs.get("cup_height", 0.10)      # total height
            t   = self.cfg.task_specs.get("cup_thickness", 0.003)  # wall thickness
            b   = self.cfg.task_specs.get("cup_bottom", 0.003)     # bottom thickness

            R_i = R_o - t
            H_i = H - b

            # handle params
            use_handle = self.cfg.task_specs.get("cup_handle", True)
            r_h = self.cfg.task_specs.get("handle_radius", 0.008)
            h_h = self.cfg.task_specs.get("handle_half_len", 0.03)

            # mixture weights (by volume)
            V_shell = np.pi * R_o**2 * H - np.pi * R_i**2 * H_i
            V_handle = 0.0
            if use_handle:
                V_handle = np.pi * r_h**2 * (2 * h_h)

            V_total = V_shell + V_handle
            p_shell = V_shell / V_total
            p_handle = V_handle / V_total if use_handle else 0.0

            u = np.random.rand()

            # -----------------------------
            # 1) SAMPLE CUP SHELL
            # -----------------------------
            if u < p_shell:
                # decide inner vs outer wall sampling
                if np.random.rand() < 0.5:
                    # outer wall
                    theta = 2 * np.pi * np.random.rand()
                    z = H * np.random.rand()
                    r = R_o
                else:
                    # inner wall
                    theta = 2 * np.pi * np.random.rand()
                    z = b + H_i * np.random.rand()
                    r = R_i

                x = r * np.cos(theta)
                y = r * np.sin(theta)

                p = np.array([x, y, z])

            # -----------------------------
            # 2) SAMPLE HANDLE (capsule)
            # -----------------------------
            else:
                # sample along capsule axis (assume +Y direction handle)
                s = np.random.uniform(-h_h, h_h)
                theta = 2 * np.pi * np.random.rand()

                # random cross-section disk around centerline
                rr = r_h * np.sqrt(np.random.rand())

                x = rr * np.cos(theta)
                y = s
                z = rr * np.sin(theta)

                # shift handle outward from cup body
                offset = np.array([R_o + r_h, 0.0, H * 0.6])
                p = np.array([x, y, z]) + offset

            # transform to world frame if needed
            pos = self.data.xpos[self.box_body_id]
            R = self.data.xmat[self.box_body_id].reshape(3, 3)

            p_world = pos + (R @ p)
            return p_world
    def _quat_distance(self, quat1, quat2):
        """
        Compute geodesic distance between two quaternions on SO(3).
        Accounts for the fact that q and -q represent the same rotation.
        
        Distance = 2 * arccos(|q1 · q2|)
        
        Args:
            quat1: quaternion [w, x, y, z] (wxyz format)
            quat2: quaternion [w, x, y, z] (wxyz format)
            
        Returns:
            Rotation angle in radians
        """
        # Normalize quaternions
        q1 = quat1 / np.linalg.norm(quat1)
        q2 = quat2 / np.linalg.norm(quat2)
        
        # Compute dot product
        dot_product = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
        # Geodesic distance: 2 * arccos(|dot|)
        return 2.0 * np.arccos(dot_product)
    """
    def _sample_random_unit_quaternion(self):
        # Isotropic random rotation in quaternion form [w, x, y, z].
        q = self.np_random.normal(size=4)
        n = np.linalg.norm(q)
        if n < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return (q / n).astype(np.float64)
   
    """

    def rotate_quaternion(self, quat, degrees, axis='z'):
        """
        Rotates a quaternion by a specified angle around one axis.

        Parameters
        ----------
        quat : array-like, shape (4,)
            Quaternion [w, x, y, z].

        degrees : float
            Rotation angle in degrees.

        axis : str
            'x', 'y', or 'z'.

        Returns
        -------
        np.ndarray
            Rotated quaternion [w, x, y, z].
        """

        angle = np.deg2rad(degrees)
        half = angle / 2

        if axis == 'x':
            q_rot = np.array([
                np.cos(half),
                np.sin(half),
                0,
                0
            ])

        elif axis == 'y':
            q_rot = np.array([
                np.cos(half),
                0,
                np.sin(half),
                0
            ])

        elif axis == 'z':
            q_rot = np.array([
                np.cos(half),
                0,
                0,
                np.sin(half)
            ])

        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        # Apply rotation
        q_new = self._quat_multiply(q_rot, quat)

        # Normalize
        q_new /= np.linalg.norm(q_new)

        return q_new
    #helper for now
    def _sample_random_unit_quaternion(self):
        self.goal_index = int(self.np_random.randint(len(self.goal_quaternions)))
        self.goal_quat = self.goal_quaternions[self.goal_index].copy()
        n = np.linalg.norm(self.goal_quat)
        if n < 1e-8:
            self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.goal_quat = self.goal_quat / n
        return self.goal_quat
    def _sample_new_goal_quaternion(self):
        if self.random_goal_quat:
            self.goal_quat = self._sample_random_unit_quaternion()
        else:
            # Keep configured static goal in evaluation mode.
            q = np.array(self.task_specs.get('goal_orientation', [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
            n = np.linalg.norm(q)
            self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if n < 1e-8 else (q / n)
    
    def _compute_orientation_reward(self, box_quat_bef, box_quat_aft, dropped=False):
        """
        Andrychowitz et al. 2019: Orientation matching reward.
        
        Reward = d(q_before, q_goal) - d(q_after, q_goal)
        
        This is potential-based shaping that encourages reducing orientation error.
        Also provides a bonus when goal is reached.
        
        Args:
            box_quat_bef: box quaternion before action [w, x, y, z]
            box_quat_aft: box quaternion after action [w, x, y, z]
            
        Returns:
            float: reward value
        """
        # Compute orientation errors
        error_before = self._quat_distance(box_quat_bef, self.goal_quat)
        error_after = self._quat_distance(box_quat_aft, self.goal_quat)
        
        # Main reward: negative derivative of orientation error
        # (positive when error decreases)
        orientation_reward = error_before - error_after
        
        reward = orientation_reward

        # Success bonus and immediate goal resampling after achieving the target.
        reached_goal = error_after < self.orientation_success_threshold
        if reached_goal:
            reward += self.orientation_success_bonus * (self.goal_reached_count + 1)
            self.goal_reached_count += 1
            self._sample_new_goal_quaternion()

        # Penalty if object is dropped.
        if dropped:
            reward -= self.drop_penalty
            #reward -= 20
        
        applied_drop = self.drop_penalty if dropped else 0.0
        applied_goal = self.orientation_success_bonus * (self.goal_reached_count + 1) if reached_goal else 0.0
        """
        if dropped or reached_goal:
            print(
                "Error reward:", orientation_reward,
                "Dropped penalty applied:", applied_drop,
                "Goal bonus applied:", applied_goal,
                "Goals reached:", self.goal_reached_count,
                "Total reward:", reward,
            )
        """
        #print("Dropped: ", dropped, "Reached goal: ", reached_goal, "Reached Count: ", self.goal_reached_count, "Orientation reward: ", orientation_reward, "Total reward: ", reward)
        #print("Goal quaternion:", self.goal_quat)
        #print("Box quaternion after action:", box_quat_aft)
        return reward, dropped, reached_goal

    def get_limb_geom_ids(self, model, root_body_name="0", exclude_geom_type="ellipsoid"):
        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)

        limb_geom_ids = []

        for body_id in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            if body_name is None:
                continue

            # skip root, box, walls
            if body_name in {"0", "box", "box_walls"}:
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

    def action_to_control(self, a):
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        """
        print(
            "Action to control - bodies:",
            len(self.robot.bodies),
            "actions:",
            a.shape,
            "nu:",
            self.model.nu
        )
        """
        # -----------------------------
        # 1. ROOT CONTROL (x, y, z) if available
        # -----------------------------
        root_a = a[0]

        if len(self.root_actuator_ids) > 0:
            for i, act_id in enumerate(self.root_actuator_ids):
                if i >= root_a.shape[0]:
                    break
                ctrl[act_id] = root_a[i]

            #print("ROOT:", aname, "act_id:", act_id, "value:", root_a[i])
        
        #print("Control after root assignment:", ctrl)

        # -----------------------------
        # 2. BODY CONTROL
        # -----------------------------
        body_actions = a[1:, 0]

        #print("This is the body_actions shape: ", body_actions.shape)

        assert len(body_actions) == len(self.robot.bodies) - 1, \
            f"Mismatch between body actions and robot bodies: {len(body_actions)} and {len(self.robot.bodies) - 1}"

        for body, body_a in zip(self.robot.bodies[1:], body_actions):

            aname = body.get_actuator_name()
            if aname is None:
                continue

            act_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                aname
            )

            if act_id != -1:
                ctrl[act_id] = body_a
            """
            print(
                "BODY:",
                body.name,
                "actuator:",
                aname,
                "act_id:",
                act_id,
                "value:",
                body_a
            )
            """
        #print("Final control vector:", ctrl)
            
        return ctrl

    def step(self, a):
        if not self.is_inited:
            return self._get_obs(), 0, False, False, {'use_transform_action': False, 'stage': 'execution'}
        self.cur_t += 1
        # skeleton transform stage
        if self.stage == 'skeleton_transform':
            skel_a = a[:, -1]
            succ = self.apply_skel_action(skel_a)
            if not succ:
                return self._get_obs(), 0.0, True, False, {'use_transform_action': True, 'stage': 'skeleton_transform'}

            if self.cur_t == self.cfg.skel_transform_nsteps:
                self.transit_attribute_transform()

            ob = self._get_obs()
            reward = 0.0
            termination = truncation = False
            return ob, reward, termination, truncation, {'use_transform_action': True, 'stage': 'skeleton_transform'}
        # attribute transform stage
        elif self.stage == 'attribute_transform':
        
            design_a = a[:, self.control_action_dim:-1] 
            if self.abs_design:
                design_params = design_a * self.cfg.robot_param_scale
            else:
                design_params = self.design_cur_params + design_a * self.cfg.robot_param_scale
            succ = self.set_design_params(design_params)
            if not succ:
                return self._get_obs(), 0.0, True, False, {'use_transform_action': True, 'stage': 'attribute_transform'}

            if self.cur_t == self.cfg.skel_transform_nsteps + 1:
                succ = self.transit_execution()
                if not succ:
                    return self._get_obs(), 0.0, True, False, {'use_transform_action': True, 'stage': 'attribute_transform'}

            ob = self._get_obs()
            reward = 0.0
            termination = truncation = False
            return ob, reward, termination, truncation, {'use_transform_action': True, 'stage': 'attribute_transform'}
        # execution stage
        else:
            self.control_nsteps += 1
            assert np.all(a[:, self.control_action_dim:] == 0)
            control_a = a[:, :self.control_action_dim]
            ctrl = self.action_to_control(control_a)
            ctrl_cost_coeff = self.cfg.reward_specs.get('ctrl_cost_coeff', 1e-4)

            self._cache_box_addrs()

            rob_pos_bef = self.get_body_com("0")[0:3].copy()
            box_pos_bef = self.get_body_com("box")[0:3].copy()
            box_state_bef = self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 7].copy()
            rob_box_dist_bef = np.linalg.norm(rob_pos_bef - box_pos_bef)

            limb_geom_ids = self.get_limb_geom_ids(self.model)
            limb_geom_ids = limb_geom_ids[1:]  # exclude root geom if present
            # box info
            box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
            box_pos = self.data.xpos[box_id]

            # IMPORTANT: geom size (half-extents)
            box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box")
            box_size = self.model.geom_size[box_geom_id]
           
            

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
                return self._get_obs(), 0, True, False, {'use_transform_action': False, 'stage': 'execution'}

            self._set_prev_action(ctrl)

            rob_pos_aft = self.get_body_com("0")[0:3].copy()
            box_pos_aft = self.get_body_com("box")[0:3].copy()
            rob_box_dist_aft = np.linalg.norm(rob_pos_aft - box_pos_aft)
            """
            #This is the just moving reward (check also .yml file)
            self.rob_box_dist = rob_pos_aft - box_pos_aft
            
            if self.task_specs.get('mov_goal', False):
                self.box_goal_dist = box_pos_aft - self.goal_pos
                box_goal_dist_aft = np.linalg.norm(self.box_goal_dist)
                reward = (box_goal_dist_bef - box_goal_dist_aft) /self.dt
            else:
                reward = (box_pos_aft[0] - box_pos_bef[0]) /self.dt

            reward += (rob_box_dist_bef - rob_box_dist_aft) / self.dt
            """
            # distances
            #dist = np.linalg.norm(rob_pos_aft - box_pos_aft)
            # positions
            #This is the just lifting reward (check also .yml file)
            rob_pos_aft = self.get_body_com("0")[0:3].copy()
            box_state_aft = self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 7].copy()
            rob_box_dist_aft = np.linalg.norm(rob_pos_aft - box_state_aft[0:3])

            self.rob_box_dist = rob_pos_aft - box_state_aft[0:3]

            if self.task_specs.get('reward_type', 'default') == 'zheight':
                reward = (box_state_aft[2] - self.box_init_height) / self.dt
                #print("Hello everyone.")
            else:
                reward = (box_state_aft[2] - box_state_bef[2]) / self.dt
                print("Also hello everyone.")
            
            if self.task_specs.get('box_cont_cost', False):
                cont_cost = np.linalg.norm(box_state_aft - box_state_bef)
                reward += cont_cost * self.cfg.reward_specs.get('box_cont_cost_coeff', 1.0)
                print("And a last one.")

            
            reward += (rob_box_dist_bef - rob_box_dist_aft) / self.dt
            
            rob_pos_aft = self.get_body_com("0")[0:3].copy()
            box_pos_aft = self.get_body_com("box")[0:3].copy()

            box_pos = self.data.xpos[self.box_body_id].copy()

            
            

            # lift
            #lift_reward = max(0, box_pos_aft[2] - 0.05) if grasp else 0.0
            height = box_pos_aft[2]
            #lift_reward = max(0, height - 0.05)
        

            limb_geom_ids = self.get_limb_geom_ids(self.model)
            limb_geom_ids = limb_geom_ids[1:]  # exclude root geom if present
            # box info
            box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
            box_pos = self.data.xpos[box_id]

            # IMPORTANT: geom size (half-extents)
            box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box")
            box_size = self.model.geom_size[box_geom_id]
            gripper_root_body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "0"
            )

            # Lowest point of box
            """
            box_bottom_z = self.compute_object_lowest_point(
                self.model,
                self.data,
                box_geom_id
            )
            """
            box_bottom_z = self.compute_object_lowest_point(self.box_geom_id)
            


            points = self.gripper_point_cloud(self.model, self.data, limb_geom_ids)
            _, tri = self.compute_gripper_hull(points)
            grasp_score_aft = self.gripper_box_overlap(box_pos, box_size, tri)
            half_inside_gripper = float(grasp_score_aft >= self.binary_overlap_threshold)
            binary_reward = self.binary_overlap_reward * half_inside_gripper






            

            floor_contact_penalty = 0.0
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if c.geom1 == 0 or c.geom2 == 0:
                    floor_contact_penalty += 1.0

            

            # Andrychowitz et al. 2019: Orientation-based reward
            # Main reward: potential-based shaping on orientation distance
            box_quat_bef = box_state_bef[3:7].copy()
            box_quat_aft = box_state_aft[3:7].copy()

            # Dropped is defined by direct box-ground contact.
            #dropped = self.has_box_plane_contact(
            #    self.model,
            #    self.data,
            #    plane_geom_id=0
            #)
            dropped = False
            if box_state_aft[2] < 0.31:
                dropped = True
            orientation_reward, dropped, reached_goal = self._compute_orientation_reward(
                box_quat_bef,
                box_quat_aft,
                dropped=dropped
            )
            #print("Orientation reward: ", orientation_reward, " Dropped: ", dropped, " Reached goal: ", reached_goal)
            #print("Actions: ", a, " Control: ", ctrl, " Rob pos before: ", rob_pos_bef, " Rob pos after: ", rob_pos_aft)
            #print("Box pos before: ", box_pos_bef, " Box pos after: ", box_pos_aft, " Goal quat: ", self.goal_quat, " Box quat after: ", box_quat_aft, " Box quat before: ", box_quat_bef)
            #if dropped or reached_goal: 
            #    print("Dropped status: ", dropped, " Box bottom z: ", box_bottom_z, "Goal reached: ", reached_goal)
            # Grasping emerges naturally from the need to rotate the object
            # No explicit rewards for grasp quality, contact count, or force closure
            orientation_component = self.weights['orientation'] * orientation_reward
            binary_component = self.weights.get('binary', 1.0) * binary_reward
            reward_components = {
                'distance': 0.0,
                'grasp': 0.0,
                'compactness': 0.0,
                'lift': 0.0,
                'orientation': float(orientation_component),
                'binary': float(binary_component),
            }
            reward = orientation_component + binary_component

            #print("Grasp Score before: ", grasp_score_bef, " Grasp Score after: ", grasp_score_aft, " Grasp Delta: ", grasp_delta, " Floor Contact Penalty: ", floor_contact_penalty)
            #print("Grasp compactness before: ", compactness_score_bef, "Grasp compactness after: ", compactness_score_aft, " Grasp Closure: ", grasp_closure)
            #print("Contact reward: ", contact_count, " Rob-Box Dist Reward: ", (rob_box_dist_bef - rob_box_dist_aft) / self.dt, " Force Closure Reward: ", fc_score, " Force Magnitude Reward: ", force_mag, " Lift Reward: ", lift_reward)
            #print("Total reward: ", reward)
           
           
            # reward -= ctrl_cost_coeff * np.square(ctrl).mean()
            # reward += self.cfg.reward_specs.get('alive_bonus', 0.0)
            # scale = self.cfg.reward_specs.get('exec_reward_scale', 1.0)
            # reward *= scale
            
            s = self.state_vector()
            height = s[2]
            zdir = quaternion_matrix(s[3:7])[:3, 2]
            ang = np.arccos(zdir[2])
            done_condition = self.cfg.done_condition
            min_height = done_condition.get('min_height', -1.0)
            max_height = done_condition.get('max_height', 2.5)
            #max_height = done_condition.get('max_height', 10.0)
            #print("This is the height: ", self.state_vector())
            #height = s[15]
            #max_ang = done_condition.get('max_ang', 3600)
            max_ang = 180
            max_nsteps = done_condition.get('max_nsteps', 1000)
            #max_distance = 20.0
            
            termination = not (np.isfinite(s).all() and (height > min_height) and (height < max_height) and (abs(ang) < np.deg2rad(max_ang)))
            truncation = not (self.control_nsteps < max_nsteps)

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
            ob = self._get_obs()

            if termination == True: 
                print("Is finite: ", np.isfinite(s).all(), " Height: ", height, " Min Height: ", min_height, " Max Height: ", max_height, " Angle: ", abs(ang), " Max Angle: ", np.deg2rad(max_ang))
            
            return ob, reward, termination, truncation, {
                'use_transform_action': False,
                'stage': 'execution',
                'goal_quat': self.goal_quat.copy(),
                'goals_reached': self.goal_reached_count,
                'reward_components': reward_components,
            }
    
    def transit_attribute_transform(self):
        self.stage = 'attribute_transform'

    def transit_execution(self):
        self.stage = 'execution'
        self.control_nsteps = 0
        self.inhand_yaw_accum = 0.0
        try:
            self.reset_state(True)
        except:
            print(self.cur_xml_str)
            return False
        self.model.geom_rgba[self.box_id][3] = 1.0
        return True
        

    def if_use_transform_action(self):
        return ['skeleton_transform', 'attribute_transform', 'execution'].index(self.stage)

    def _quat_conjugate(self, q):
        q = np.asarray(q, dtype=np.float64)
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

    def _quat_multiply(self, q1, q2):
        q1 = np.asarray(q1, dtype=np.float64)
        q2 = np.asarray(q2, dtype=np.float64)
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        q = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float64)
        qn = np.linalg.norm(q)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if qn < 1e-8 else (q / qn)

    def _set_prev_action(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] >= self.prev_action_dim:
            self.prev_action = action[:self.prev_action_dim].copy()
        else:
            self.prev_action = np.zeros(self.prev_action_dim, dtype=np.float64)
            self.prev_action[:action.shape[0]] = action

    def _get_fingertip_positions(self):
        tip_positions = []
        num_fingertips_obs = int(getattr(self, 'num_fingertips_obs', 4))
        fingertip_body_names = getattr(self, 'fingertip_body_names', None)

        if fingertip_body_names is not None:
            candidate_names = list(fingertip_body_names)
        else:
            root_name = self.robot.bodies[0].name if len(self.robot.bodies) > 0 else None
            candidate_names = [
                body.name for body in self.robot.bodies
                if len(body.child) == 0 and body.name != root_name
            ]

        for name in candidate_names:
            if len(tip_positions) >= num_fingertips_obs:
                break
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                tip_positions.append(self.data.xpos[body_id].copy())

        while len(tip_positions) < num_fingertips_obs:
            tip_positions.append(np.zeros(3, dtype=np.float64))

        return np.concatenate(tip_positions, axis=0)
    

    def get_sim_obs(self):
        rob_obs = []
        env_obs = []

        # Root position (for offsets)
        if 'root_offset' in self.sim_specs:
            root_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.robot.bodies[0].name
            )
            root_pos = self.data.xpos[root_id]

        has_floating_root = (
            self.model.njnt > 0
            and self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        )

        qvel = self.data.qvel.copy()
        if self.clip_qvel:
            qvel = np.clip(qvel, -10, 10)

        # =========================
        # Robot observations
        # =========================
        for i, body in enumerate(self.robot.bodies):

            if i == 0:
                # root body
                if has_floating_root:
                    obs_i = [
                        self.data.qpos[1:7],
                        qvel[:6],
                    ]
                else:
                    obs_i = [np.zeros(13)]
            else:
                qs, qe = get_single_body_qposaddr(self.model, body.name)

                # 🚨 CRITICAL SAFETY CHECK (prevents segfault)
                if qs < 0 or qe > self.model.nq or qs > qe:
                    print("\n🚨 INVALID QPOS RANGE 🚨")
                    print(f"Body: {body.name}")
                    print(f"qs: {qs}, qe: {qe}, nq: {self.model.nq}")
                    raise RuntimeError("Bad qpos indexing")

                # 🚨 QVEL CHECK
                if has_floating_root:
                    qvel_qs = qs - 1
                    qvel_qe = qe - 1
                else:
                    qvel_qs = qs
                    qvel_qe = qe

                if qvel_qs < 0 or qvel_qe > self.model.nv:
                    print("\n🚨 INVALID QVEL RANGE 🚨")
                    print(f"Body: {body.name}")
                    print(f"qs: {qs}, qe: {qe}, qvel_qs: {qvel_qs}, qvel_qe: {qvel_qe}, nv: {self.model.nv}")
                    raise RuntimeError("Bad qvel indexing")

                if qe - qs >= 1:
                    # NOTE: your assumption is 1 DOF joints
                    if (qe - qs) != 1:
                        print(f"⚠️ Unexpected joint dim for body {body.name}: {qe - qs}")

                    qvel_slice = qvel[qvel_qs:qvel_qe]

                    obs_i = [
                        np.zeros(11),
                        self.data.qpos[qs:qe],
                        qvel_slice
                    ]
                else:
                    obs_i = [np.zeros(13)]

            # Root offset feature
            if 'root_offset' in self.sim_specs:
                body_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body.name
                )
                offset = self.data.xpos[body_id][[0, 2]] - root_pos[[0, 2]]
                obs_i.append(offset)

            obs_i = np.concatenate(obs_i)
            rob_obs.append(obs_i)

        rob_obs = np.stack(rob_obs)

        self._cache_box_addrs()
        box_qpos = self.data.qpos[self.box_qpos_adr:self.box_qpos_adr + 7].copy()
        object_pos = box_qpos[:3]
        object_quat = box_qpos[3:7]
        goal_quat = getattr(self, 'goal_quat', np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)).copy()
        prev_action = getattr(self, 'prev_action', np.zeros(0, dtype=np.float64))
        if self.task_specs.get('mov_goal', False):
            rel_pos_obj_goal = self.goal_pos - object_pos
        else:
            rel_pos_obj_goal = np.zeros(3, dtype=np.float64)
        rel_quat_obj_goal = self._quat_multiply(goal_quat, self._quat_conjugate(object_quat))
        fingertip_positions = self._get_fingertip_positions()

        extra_root_obs = np.concatenate([
            object_pos,
            object_quat,
            goal_quat,
            rel_pos_obj_goal,
            rel_quat_obj_goal,
            prev_action,
            fingertip_positions,
        ])
        extra_zeros = np.zeros_like(extra_root_obs)

        # =========================
        # Environment observations
        # =========================
        for i in range(rob_obs.shape[0]):
            if i == 0:
                obs_i = [
                    self.rob_box_dist[:2],
                    self.data.qpos[self.box_qpos_adr + 2:self.box_qpos_adr + 7],
                    qvel[self.box_qvel_adr:self.box_qvel_adr + 6],
                    extra_root_obs,
                ]
            elif i == 1 and self.task_specs.get('mov_goal', False):
                obs_i = [
                    self.box_goal_dist,
                    np.zeros(10),
                    extra_zeros,
                ]
            else:
                obs_i = [np.zeros(13), extra_zeros]

            obs_i = np.concatenate(obs_i)
            env_obs.append(obs_i)

        env_obs = np.stack(env_obs)

        # Final observation
        obs = np.concatenate([rob_obs, env_obs], axis=-1)
        return obs
    def get_attr_fixed(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
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

    def get_body_index(self):
        index = []
        for i, body in enumerate(self.robot.bodies):
            ind = int(body.name, base=self.index_base)
            index.append(ind)
        index = np.array(index)
        return index

    def get_body_height(self):
        heights = []
        for i, body in enumerate(self.robot.bodies):
            h = body.height
            heights.append(h)
        heights = np.array(heights)
        return heights
        
    def get_body_depth(self):
        depths = []
        for i, body in enumerate(self.robot.bodies):
            d = body.depth
            depths.append(d)
        depths = np.array(depths)
        return depths

    def _get_obs(self):
        attr_fixed_obs = self.get_attr_fixed()
        sim_obs = self.get_sim_obs()
        design_obs = self.design_cur_params
        obs = np.concatenate(list(filter(lambda x: x is not None, [attr_fixed_obs, sim_obs, design_obs])), axis=-1)
        if self.cfg.obs_specs.get('fc_graph', False):
            edges = get_graph_fc_edges(len(self.robot.bodies))
        else:
            edges = self.robot.get_gnn_edges()
        use_transform_action = np.array([self.if_use_transform_action()])
        num_nodes = np.array([sim_obs.shape[0]])
        all_obs = [obs, edges, use_transform_action, num_nodes]
        if self.use_body_ind:
            body_index = self.get_body_index()
            all_obs.append(body_index)
        if self.use_body_depth_height:
            body_depths = self.get_body_depth()
            all_obs.append(body_depths)
            body_heights = self.get_body_height()
            all_obs.append(body_heights)
        if self.use_shortest_distance:
            distances = self.robot.get_shortest_distances()
            all_obs.append(distances)
        if self.use_position_encoding:
            lapPE = self.robot.get_laplacian_position_encoding()
            all_obs.append(lapPE)
        return all_obs

    def _estimate_box_xy_half_extent(self):
        geom_type = self.model.geom_type[self.box_geom_id]
        geom_size = self.model.geom_size[self.box_geom_id]

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

    def _sample_box_spawn_pos(self):
        default_box_pos = np.array(self.task_specs.get('box_pos'), dtype=np.float64)
        if not self.random_box_pos:
            return default_box_pos

        bounds = self._get_wall_spawn_bounds()
        if bounds is None:
            return default_box_pos

        x_min, x_max, y_min, y_max = bounds
        margin = max(0.0, self.box_random_margin) + self._estimate_box_xy_half_extent()

        low = np.array([x_min + margin, y_min + margin], dtype=np.float64)
        high = np.array([x_max - margin, y_max - margin], dtype=np.float64)

        if np.any(low >= high):
            return default_box_pos

        sampled_xy = self.np_random.uniform(low=low, high=high)
        sampled_box_pos = default_box_pos.copy()
        sampled_box_pos[0] = sampled_xy[0]
        sampled_box_pos[1] = sampled_xy[1]
        return sampled_box_pos

    def reset_state(self, add_noise):
        self.inhand_yaw_accum = 0.0
        self.goal_reached_count = 0
        self.prev_action = np.zeros(self.prev_action_dim, dtype=np.float64)
        self._sample_new_goal_quaternion()
        has_floating_root = (
            self.model.njnt > 0
            and self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        )
        if add_noise:
            qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        else:
            qpos = self.init_qpos.copy()
            qvel = self.init_qvel.copy()

        self.box_pos = self._sample_box_spawn_pos()

        if self.env_specs.get('init_height', True) and has_floating_root:
            root_qpos_adr = self.model.jnt_qposadr[0]
            qpos[root_qpos_adr:root_qpos_adr + 3] = np.array([0.0, 0.0, 0.3], dtype=np.float64)

        #qpos[-7:-4] = self.box_pos
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "box"
        )
        adr = self.model.jnt_qposadr[joint_id]

        qpos[adr:adr+3] = self.box_pos
        qpos[adr+3:adr+7] = np.array([1,0,0,0])
        #self.rob_box_dist = self.box_pos
        
        
        if self.task_specs.get('mov_goal', False):
            if self.task_specs.get('random_goal'):
                self.goal_pos[:2] = rand_coord(self.np_random, np.array(self.task_specs.get('goal_range')))
            
            self.box_goal_dist = self.box_pos - self.goal_pos

        
        
        self.set_state(qpos, qvel)

        box_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "box"
            )

        mujoco.mj_forward(self.model, self.data)
        rob_pos = self.get_body_com("0")[0:3]
        self.rob_box_dist = rob_pos - self.box_pos

    def reset_robot(self):
        del self.robot
        self.robot = Robot(self.cfg.robot_cfg, xml=self.init_xml_str, is_xml_str=True)
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        xml_str_fixed = self.cur_xml_str.replace(' center="0 0 0"', '')
        self.reload_sim_model(xml_str_fixed)
        self.cur_xml_str = xml_str_fixed
        self.box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box")
        self.box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self._cache_root_actuator_ids()
        self._cache_box_addrs()
        #self.reload_sim_model(self.cur_xml_str)
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()

    def reset_model(self):
        
        self.reset_robot()
        
        self.control_nsteps = 0
        self.stage = 'skeleton_transform'
        self.cur_t = 0
        self.reset_state(False)
        return self._get_obs()

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
        self.box_joint_id = self.model.joint("box_joint").id
        self.box_qpos_adr = self.model.jnt_qposadr[self.box_joint_id]
        self.box_qvel_adr = self.model.jnt_dofadr[self.box_joint_id]

    def _cache_root_actuator_ids(self):
        self.root_actuator_names = ["root_x_motor", "root_y_motor", "root_z_motor"]
        self.root_actuator_ids = []
        for name in self.root_actuator_names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id != -1:
                self.root_actuator_ids.append(act_id)