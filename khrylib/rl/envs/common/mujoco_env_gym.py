from collections import OrderedDict
import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
from pathlib import Path
import gym

DEFAULT_SIZE = 500

try:
    import mujoco
    from mujoco import MjModel, MjData
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))
    
ASSET_ROOT = "/home/juliusk/BodyGen/BodyGenExtend/BodyGenExtend/assets/mujoco_envs"
    
def fix_mesh_paths(xml_str):
    xml_str = xml_str.replace(
        'meshes/',
        os.path.join(ASSET_ROOT, 'meshes') + '/'
    )
    return xml_str

def convert_observation_to_space(observation):
    if isinstance(observation, list):
        return None

    if isinstance(observation, dict):   
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, fullpath, frame_skip, mujoco_xml=None):
        if mujoco_xml is not None:
            self.model = mujoco.MjModel.from_xml_string(mujoco_xml)
        else:
            if not path.exists(fullpath):
                # try the default assets path
                fullpath = path.join(Path(__file__).parent.parent.parent.parent, 'assets/mujoco_models', path.basename(fullpath))
                if not path.exists(fullpath):
                    raise IOError("File %s does not exist" % fullpath)
            self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.frame_skip = frame_skip
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.is_inited = False
        
        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, termination, truncation, _info = self.step(action)
        done = (termination or truncation)
        assert not done

        self._set_observation_space(observation)

        self.seed()
        self.is_inited = True

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reload_sim_model(self, xml_str):
        # Clean up old objects
        del self.data
        del self.model

        try:
            xml_str_fixed = fix_mesh_paths(xml_str)
            self.model = mujoco.MjModel.from_xml_string(xml_str_fixed)
            self.data = mujoco.MjData(self.model)

        except Exception as e:
            print("\n🚨 XML PARSING FAILED 🚨")
            print(e)
            print("\n--- Problematic XML ---\n")
            print(xml_str)
            raise

        # Reinitialize state
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Reset viewer (this is fine, keep it)
        self.viewer = None
        self._viewers = {}

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)

    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):  
        if mode == "human":
            # Launch passive viewer once
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            return

        elif mode in ["rgb_array", "depth_array"]:
            # Create renderer on demand
            if not hasattr(self, "_renderer"):
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)

            if camera_name is not None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name
                )

            # Ensure valid default
            if camera_id is None:
                camera_id = -1
            
            # Update scene with determined camera
            self._renderer.update_scene(self.data, camera=camera_id)

            if mode == "rgb_array":
                return self._renderer.render()

            elif mode == "depth_array":
                return self._renderer.render(depth=True)

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            self._viewers[mode] = self.viewer
        self.viewer_setup()

    def set_custom_key_callback(self, key_func):
        self._get_viewer('human').custom_key_callback = key_func

    def get_body_com(self, body_name):
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name
        )
        return self.data.xpos[body_id].copy()

    def state_vector(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat
        ])

    def vec_body2world(self, body_name, vec):
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name
        )
        body_xmat = self.data.xmat[body_id].reshape(3, 3)
        return (body_xmat @ vec.reshape(3, 1)).ravel()

    def pos_body2world(self, body_name, pos):
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name
        )
        body_xpos = self.data.xpos[body_id]
        body_xmat = self.data.xmat[body_id].reshape(3, 3)
        return (body_xmat @ pos.reshape(3, 1)).ravel() + body_xpos

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            self._viewers[mode] = self.viewer
        self.viewer_setup()


from khrylib.rl.envs.common.mjviewer import MjViewer

ASSET_ROOT = "/home/juliusk/BodyGen/BodyGenExtend/BodyGenExtend/assets/mujoco_envs"

def fix_mesh_paths(xml_str):
    xml_str = xml_str.replace(
        'meshes/',
        os.path.join(ASSET_ROOT, 'meshes') + '/'
    )
    return xml_str


def convert_observation_to_space(observation):
    if isinstance(observation, list):
        return None

    if isinstance(observation, dict):   
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, fullpath, frame_skip, mujoco_xml=None):
        if mujoco_xml is not None:
            self.model = mujoco.MjModel.from_xml_string(mujoco_xml)
        else:
            if not path.exists(fullpath):
                # try the default assets path
                fullpath = path.join(Path(__file__).parent.parent.parent.parent, 'assets/mujoco_models', path.basename(fullpath))
                if not path.exists(fullpath):
                    raise IOError("File %s does not exist" % fullpath)
            self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.frame_skip = frame_skip
        self.data = mujoco.MjData(self.model)
        #self.sim = mujoco_py.MjSim(self.model)
        #self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        #self.init_qpos = self.sim.data.qpos.ravel().copy()
        #self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.is_inited = False
        

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, termination, truncation, _info = self.step(action)
        done = (termination or truncation)
        assert not done

        self._set_observation_space(observation)

        self.seed()
        self.is_inited = True

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    """
    def reload_sim_model(self, xml_str):
        #del self.sim
        del self.model
        del self.data
        del self.viewer
        del self._viewers
        #self.model = mujoco_py.load_model_from_xml(xml_str)
        #self.sim = mujoco_py.MjSim(self.model)
        #self.data = self.sim.data
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self.viewer = None
        self._viewers = {}
    """
    def reload_sim_model(self, xml_str):

        # 🔥 NEW: clean renderer FIRST
        if hasattr(self, "_renderer"):
            try:
                self._renderer.close()
            except:
                pass
            del self._renderer

        # Clean up old objects
        del self.data
        del self.model

        try:
            #self.model = mujoco.MjModel.from_xml_string(xml_str)
            xml_str_fixed = fix_mesh_paths(xml_str)
            self.model = mujoco.MjModel.from_xml_string(xml_str_fixed)
            self.data = mujoco.MjData(self.model)

        except Exception as e:
            print("\n🚨 XML PARSING FAILED 🚨")
            print(e)
            print("\n--- Problematic XML ---\n")
            print(xml_str)
            raise

        # Reinitialize state
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Reset viewer (this is fine, keep it)
        self.viewer = None
        self._viewers = {}

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        """
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()
        """
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)
    """
    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model.camera_name2id:
                camera_id = self.model.camera_name2id[camera_name]

            # Create renderer on demand
            if not hasattr(self, "_renderer"):
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)

            self._renderer.update_scene(self.data, camera=camera_id)

            if mode == 'rgb_array':
                return self._renderer.render()
            elif mode == 'depth_array':
                return self._renderer.render_depth()
        elif mode == 'human':
            self._get_viewer(mode).render()
    """
    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):  
        import mujoco
        
        if mode == "human":
            # Launch passive viewer once
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            return

        elif mode in ["rgb_array", "depth_array"]:
            
            # Create renderer on demand
            if not hasattr(self, "_renderer"):
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)

            if camera_name is not None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name
                )

            # Ensure valid default
            if camera_id is None:
                camera_id = -1
            
            # Update scene with determined camera
            self._renderer.update_scene(self.data, camera=camera_id)

            if mode == "rgb_array":
                return self._renderer.render()

            elif mode == "depth_array":
                return self._renderer.render(depth=True)
    
    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            elif mode == 'rgb_array' or mode == 'depth_array':
                #self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            self._viewers[mode] = self.viewer
        self.viewer_setup()
        
    """
    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            elif mode == 'rgb_array' or mode == 'depth_array':
                #self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            self._viewers[mode] = self.viewer
        self.viewer_setup()
        return self.viewer
    """

    def set_custom_key_callback(self, key_func):
        self._get_viewer('human').custom_key_callback = key_func
    """
    def get_body_com(self, body_name):
        #return self.data.get_body_xpos(body_name)
    """

    def get_body_com(self, body_name):
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name
        )
        return self.data.xpos[body_id].copy()

    def state_vector(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat
        ])
    """
    def vec_body2world(self, body_name, vec):
        body_xmat = self.data.get_body_xmat(body_name)
        vec_world = (body_xmat @ vec[:, None]).ravel()
        return vec_world
    """
    def vec_body2world(self, body_name, vec):
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name
        )
        body_xmat = self.data.xmat[body_id].reshape(3, 3)
        return (body_xmat @ vec.reshape(3, 1)).ravel()
    """
    def pos_body2world(self, body_name, pos):
        body_xpos = self.data.get_body_xpos(body_name)
        body_xmat = self.data.get_body_xmat(body_name)
        pos_world = (body_xmat @ pos[:, None]).ravel() + body_xpos
        return pos_world
    """
    def pos_body2world(self, body_name, pos):
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name
        )
        body_xpos = self.data.xpos[body_id]
        body_xmat = self.data.xmat[body_id].reshape(3, 3)
        return (body_xmat @ pos.reshape(3, 1)).ravel() + body_xpos
