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
        self.rob_box_dist = np.array([0.0, 0.0, 0.0])
        if self.task_specs.get('mov_goal', False):
            self.goal_pos = np.array(self.task_specs.get('goal_pos'))
            self.box_goal_dist = self.box_pos - self.goal_pos
        MujocoEnv.__init__(self, self.model_xml_file, 4)
        utils.EzPickle.__init__(self)
        self.box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        #self.control_action_dim = 1
        self.control_action_dim = 3
        self._cache_box_addrs()
        self.skel_num_action = 3 if cfg.enable_remove else 2
        self.sim_obs_dim = self.get_sim_obs().shape[-1]
        self.attr_fixed_dim = self.get_attr_fixed().shape[-1]
        self.weights = cfg.task_specs.get('weights', {})
        self.inhand_yaw_accum = 0.0
        print(self.task_specs)
        print("Weights for reward components: ", self.weights)


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
    def staged_rewards(self):
        """
        Staged reward for grasping one box and stacking it on top of another.

        Returns:
            r_reach_grasp: reward for reaching and grasping
            r_lift_align: reward for lifting and aligning
            r_stack: reward for successful stacking
        """

        # -------------------------
        # Object positions
        # -------------------------
        boxA_pos = self.data.geom_xpos[self.box_geom_id].copy()
        boxB_pos = self.data.geom_xpos[self.target_box_geom_id].copy()

        # Gripper/root position (use whatever is your gripper reference)
        gripper_pos = self.get_body_com("0")[0:3].copy()

        # -------------------------
        # Reach reward
        # -------------------------
        dist = np.linalg.norm(gripper_pos - boxA_pos)
        r_reach = (1.0 - np.tanh(10.0 * dist)) * 0.25

        # -------------------------
        # Grasp reward
        # -------------------------
        limb_geom_ids = self.get_limb_geom_ids(self.model)
        grasping_boxA = self.compute_box_gripper_contacts(
            self.model,
            self.data,
            limb_geom_ids,
            self.box_geom_id
        ) > 0

        if grasping_boxA:
            r_reach += 0.25

        # -------------------------
        # Lift reward
        # -------------------------
        boxA_bottom_z = self.compute_box_lowest_point(
            self.model,
            self.data,
            self.box_geom_id
        )

        cubeA_lifted = boxA_bottom_z > self.box_init_height + 0.04
        r_lift = 1.0 if cubeA_lifted else 0.0

        # -------------------------
        # Align reward
        # -------------------------
        if cubeA_lifted:
            horiz_dist = np.linalg.norm(boxA_pos[:2] - boxB_pos[:2])
            r_lift += 0.5 * (1.0 - np.tanh(horiz_dist))

        # -------------------------
        # Stack reward
        # -------------------------
        r_stack = 0.0
        boxA_touching_boxB = self.compute_box_gripper_contacts(
            self.model,
            self.data,
            [self.target_box_geom_id],   # if you have a contact helper for box-box, use that instead
            self.box_geom_id
        ) > 0

        if (not grasping_boxA) and r_lift > 0 and boxA_touching_boxB:
            r_stack = 2.0

        return r_reach, r_lift, r_stack
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
        hull = ConvexHull(points)
        tri = Delaunay(points[hull.vertices])
        return hull, tri

    def point_in_gripper(self, point, tri):
        return tri.find_simplex(point) >= 0
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
    def occupancy_in_gripper(self, box_center, box_size, tri, n_samples=500):
        #samples = self.sample_points_in_box(box_center, box_size, n_samples)
        samples = self.sample_points_in_object(box_center, self.box_geom_id, n_samples)

        inside = tri.find_simplex(samples) >= 0

        return np.mean(inside)  # fraction

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

        limb_geom_ids = self.get_limb_geom_ids(model)

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

    def _wrap_to_pi(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _quat_wxyz_to_yaw(self, quat_wxyz):
        w, x, y, z = quat_wxyz
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _compute_inhand_rotation_reward(self, box_state_bef, box_state_aft, in_hand):
        if not in_hand:
            return 0.0

        yaw_target_deg = float(self.task_specs.get('inhand_rot_target_deg', 30.0))
        yaw_target = np.deg2rad(max(1e-6, yaw_target_deg))

        yaw_bef = self._quat_wxyz_to_yaw(box_state_bef[3:7])
        yaw_aft = self._quat_wxyz_to_yaw(box_state_aft[3:7])

        # Integrate absolute yaw motion while the object is in hand.
        yaw_step = abs(self._wrap_to_pi(yaw_aft - yaw_bef))
        prev_accum = self.inhand_yaw_accum
        self.inhand_yaw_accum += yaw_step

        # Dense progress up to target plus small completion bonus.
        clipped_prev = min(prev_accum, yaw_target)
        clipped_now = min(self.inhand_yaw_accum, yaw_target)
        progress = (clipped_now - clipped_prev) / yaw_target
        reached_bonus = 0.25 if self.inhand_yaw_accum >= yaw_target else 0.0
        return progress + reached_bonus

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
        # 1. ROOT CONTROL (x, y, z)
        # -----------------------------
        root_actuators = ["root_x_motor", "root_y_motor", "root_z_motor"]
        root_a = a[0]
        #print("This is the root_a shape: ", root_a.shape)

        for i, aname in enumerate(root_actuators):
            if i >= len(a):
                break

            act_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                aname
            )

            if act_id != -1:
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
            # Distance above floor
            clearance = box_bottom_z - self.box_init_height

            clearance = np.min(clearance)

            # Only reward true lift
            if clearance > 0.0:
                lift_reward = clearance / self.dt
            else:
                lift_reward = 0.0
            
            lift_reward = (box_state_aft[2] - self.box_init_height) / self.dt
        
            

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
                "0"
            )



            grasp_delta = (grasp_score_aft - grasp_score_bef) / self.dt

            grasp_closure = (compactness_score_aft - compactness_score_bef) / self.dt

            gripper_volume = hull.volume

            action_mag = np.sum(np.square(self.data.ctrl))

            #action_penalty = - np.sum(np.square(self.data.ctrl - self.prev_action)) 

            vel_mag = np.sum(np.square(self.data.qvel))

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
                initial_box_height=0.5
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

            in_hand = (contact_count > 0) and (lift_reward > 0.0)
            inhand_rotation_reward = self._compute_inhand_rotation_reward(
                box_state_bef,
                box_state_aft,
                in_hand
            )
                

            reward = (
                self.weights['distance'] * ((rob_box_dist_bef - rob_box_dist_aft) / self.dt) +
                #2.0 * multi_contact_reward +
                #1.0 * grasp_shape_reward +
                self.weights['grasp'] * grasp_score_aft +              
                self.weights['compactness'] * compactness_score_aft +
                #1.0 * fc_score +
                self.weights['binary'] * binary_reward +
                self.weights['lift'] * lift_reward 
                #self.weights.get('inhand_rotation', 0.0) * inhand_rotation_reward
                #1.0 * holding_reward
                #-0.1 * floor_contact_penalty 
                #-0.1 * action_mag 
                #-0.1 * vel_mag
                #-0.1 * slip_reward
                #1.0 * stability_reward
                #1.0 * max(0.0, grasp_delta) +
                #1.0 * max(0.0, grasp_closure) 
                #1.0 * gripper_volume                #penalty for the volume of the hull
            )

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
            min_height = done_condition.get('min_height', 0.0)
            max_height = done_condition.get('max_height', 7.5)
            #max_height = done_condition.get('max_height', 10.0)
            #print("This is the height: ", self.state_vector())
            #height = s[15]
            #max_ang = done_condition.get('max_ang', 3600)
            max_ang = 180
            max_nsteps = done_condition.get('max_nsteps', 1000)
            #max_distance = 20.0
            
            termination = not (np.isfinite(s).all() and (height > min_height) and (height < max_height) and (abs(ang) < np.deg2rad(max_ang)) )
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

        
            
            return ob, reward, termination, truncation, {'use_transform_action': False, 'stage': 'execution'}
    
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

        qvel = self.data.qvel.copy()
        if self.clip_qvel:
            qvel = np.clip(qvel, -10, 10)

        # =========================
        # Robot observations
        # =========================
        for i, body in enumerate(self.robot.bodies):

            if i == 0:
                # root body
                """
                obs_i = [
                    self.data.qpos[2:7],
                    qvel[:6],
                    np.zeros(2)
                ]
                """
                obs_i = [
                    self.data.qpos[1:7],   # root + joints
                    self.data.qvel[0:7],   # velocities
                ]
            else:
                qs, qe = get_single_body_qposaddr(self.model, body.name)

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
                    body.name
                )
                offset = self.data.xpos[body_id][[0, 2]] - root_pos[[0, 2]]
                obs_i.append(offset)

            obs_i = np.concatenate(obs_i)
            rob_obs.append(obs_i)

        rob_obs = np.stack(rob_obs)

        # =========================
        # Environment observations
        # =========================
        for i in range(rob_obs.shape[0]):
            if i == 0:
                self._cache_box_addrs()
                obs_i = [
                    self.rob_box_dist[:2],
                    self.data.qpos[self.box_qpos_adr + 2:self.box_qpos_adr + 7],
                    qvel[self.box_qvel_adr:self.box_qvel_adr + 6]
                ]
            elif i == 1 and self.task_specs.get('mov_goal', False):
                obs_i = [
                    self.box_goal_dist,
                    np.zeros(10)
                ]
            else:
                obs_i = [np.zeros(13)]

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

    def reset_state(self, add_noise):
        self.inhand_yaw_accum = 0.0
        if add_noise:
            qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        else:
            qpos = self.init_qpos
            qvel = self.init_qvel
        if self.env_specs.get('init_height', True):
            qpos[0] = 0.0   # x
            qpos[1] = 0.0  # y
            qpos[2] = 0.0  #before it was 0.4 in total and the other two werent there
        
        self.box_pos = np.array(self.task_specs.get('box_pos'))
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
            self.box_pos = np.array(self.task_specs.get('box_pos'))
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