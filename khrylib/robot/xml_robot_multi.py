import numpy as np
import math
import re
from copy import deepcopy
from collections import defaultdict
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO
from scipy.sparse.linalg import eigsh
from scipy.linalg import fractional_matrix_power


def parse_vec(string):
    return np.fromstring(string, sep=' ')

def parse_fromto(string):
    fromto = np.fromstring(string, sep=' ')
    return fromto[:3], fromto[3:]

def normalize_range(value, lb, ub):
    '''
    linear normalize value to [-1, 1]
    '''
    return (value - lb) / (ub - lb) * 2 - 1

def denormalize_range(value, lb, ub):
    return (value + 1) * 0.5 * (ub - lb) + lb

def vec_to_polar(v):
    phi = math.atan2(v[1], v[0])
    theta = math.acos(v[2])
    return np.array([theta, phi])

def polar_to_vec(p):
    v = np.zeros(3)
    v[0] = math.sin(p[0]) * math.cos(p[1])
    v[1] = math.sin(p[0]) * math.sin(p[1])
    v[2] = math.cos(p[0])
    return v


class Joint:

    def __init__(self, node, body):
        self.node = node
        self.body = body
        self.cfg = body.cfg
        self.local_coord = body.local_coord
        self.name = node.attrib['name']
        self.type = node.attrib['type']
        if self.type == 'hinge':
            self.range = np.deg2rad(parse_vec(node.attrib.get('range', "-360 360")))
        actu_node = body.tree.getroot().find("actuator").find(f'motor[@joint="{self.name}"]')
        if actu_node is not None:
            self.actuator = Actuator(actu_node, self)
        else:
            self.actuator = None
        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.pos = parse_vec(node.attrib['pos'])
        if self.type == 'hinge':
            self.axis = vec_to_polar(parse_vec(node.attrib['axis']))
        if self.local_coord:
            self.pos += body.pos
        #You have deleted that assertion, not the original code !!!!
        #assert(np.all(self.pos == body.pos))
    
    def __repr__(self):
        return 'joint_' + self.name

    def parse_param_specs(self):
        self.param_specs =  deepcopy(self.cfg['joint_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])

    def sync_node(self):
        pos = self.pos - self.body.pos if self.local_coord else self.pos
        self.name = self.body.name + '_joint'
        self.node.attrib['name'] = self.name
        self.node.attrib['pos'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in pos])
        if self.type == 'hinge':
            axis_vec = polar_to_vec(self.axis)
            self.node.attrib['axis'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in axis_vec])
        if self.actuator is not None:
            self.actuator.sync_node()

    def get_params(self, param_list, get_name=False, pad_zeros=False):
        if 'axis' in self.param_specs:
            if self.type == 'hinge':
                if get_name:
                    param_list += ['axis_theta', 'axis_phi']
                else:
                    axis = normalize_range(self.axis, np.array([0, -2 * np.pi]), np.array([np.pi, 2 * np.pi]))
                    param_list.append(axis)
            elif pad_zeros:
                param_list.append(np.zeros(2))

        if self.actuator is not None:
            self.actuator.get_params(param_list, get_name)
        elif pad_zeros:
            param_list.append(np.zeros(1))


        if not get_name:
            self.param_inited = True

    def set_params(self, params, pad_zeros=False):
        if 'axis' in self.param_specs:
            if self.type == 'hinge':
                self.axis = denormalize_range(params[:2], np.array([0, -2 * np.pi]), np.array([np.pi, 2 * np.pi]))
                params = params[2:]
            elif pad_zeros:
                params = params[2:]

        if self.actuator is not None:
            params = self.actuator.set_params(params)
        elif pad_zeros:
            params = params[1:]
        return params


class Geom:

    def __init__(self, node, body):
        self.node = node
        self.body = body
        self.cfg = body.cfg
        self.local_coord = body.local_coord
        self.name = node.attrib.get('name', '')
        self.type = node.attrib['type']
        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.size = parse_vec(node.attrib['size'])
        if self.type == 'capsule':
            self.start, self.end = parse_fromto(node.attrib['fromto'])
            if self.local_coord:
                self.start += body.pos
                self.end += body.pos
            if body.bone_start is None:
                self.bone_start = self.start.copy()
                body.bone_start = self.bone_start.copy()
            else:
                self.bone_start = body.bone_start.copy()
            self.ext_start = np.linalg.norm(self.bone_start - self.start)
            
        elif self.type == 'ellipsoid':
            self.center = self.body.pos.copy()

            # NEW
            self.ellipsoid_radius = self.size.copy()
    
    def __repr__(self):
        return 'geom_' + self.name
    """
    def parse_param_specs(self):
        self.param_specs = deepcopy(self.cfg['geom_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])
    """
    def parse_param_specs(self):

        geom_cfg = deepcopy(self.cfg['geom_params'])

        # Ellipsoids use separate hyperparameter
        if self.type == 'ellipsoid':

            self.param_specs = {}

            if 'ellipsoid_radius' in geom_cfg:
                self.param_specs['ellipsoid_radius'] = deepcopy(
                    geom_cfg['ellipsoid_radius']
                )

        else:
            # capsules / spheres keep original behavior
            self.param_specs = geom_cfg

        for name, specs in self.param_specs.items():

            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])

            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])

            if 'min' in specs and isinstance(specs['min'], list):
                specs['min'] = np.array(specs['min'])

            if 'max' in specs and isinstance(specs['max'], list):
                specs['max'] = np.array(specs['max'])


    def update_start(self):
        if self.type == 'capsule':
            vec = self.bone_start - self.end
            self.start = self.bone_start + vec * (self.ext_start / np.linalg.norm(vec))

    def sync_node(self):
        # self.node.attrib['name'] = self.name
        self.node.attrib.pop('name', None)
        self.node.attrib['size'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in self.size])
        if self.type == 'capsule':
            start = self.start - self.body.pos if self.local_coord else self.start
            end = self.end - self.body.pos if self.local_coord else self.end
            self.node.attrib['fromto'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in np.concatenate([start, end])])
        elif self.type == 'ellipsoid':
            pos = self.center - self.body.pos if self.local_coord else self.center
            self.node.attrib.pop('center', None)
            self.node.attrib['pos'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in pos])
    """
    def get_params(self, param_list, get_name=False, pad_zeros=False):
        if 'size' in self.param_specs:
            if get_name:
                param_list.append('size')
            else:
                if self.type == 'capsule':
                    if not self.param_inited and self.param_specs['size'].get('rel', False):
                        self.param_specs['size']['lb'] += self.size
                        self.param_specs['size']['ub'] += self.size
                        self.param_specs['size']['lb'] = max(self.param_specs['size']['lb'], self.param_specs['size'].get('min', -np.inf))
                        self.param_specs['size']['ub'] = min(self.param_specs['size']['ub'], self.param_specs['size'].get('max', np.inf))
                    size = normalize_range(self.size, self.param_specs['size']['lb'], self.param_specs['size']['ub'])
                    param_list.append(size.flatten())

                elif self.type == 'ellipsoid':
                    if not self.param_inited and self.param_specs['size'].get('rel', False):
                        self.param_specs['size']['lb'] += self.size
                        self.param_specs['size']['ub'] += self.size
                        self.param_specs['size']['lb'] = np.maximum(self.param_specs['size']['lb'], self.param_specs['size'].get('min', -np.inf))
                        self.param_specs['size']['ub'] = np.minimum(self.param_specs['size']['ub'], self.param_specs['size'].get('max', np.inf))
                    size = normalize_range(self.size, self.param_specs['size']['lb'], self.param_specs['size']['ub'])
                    param_list.append(size.flatten())
                elif pad_zeros:
                    param_list.append(np.zeros(1))
        if 'ext_start' in self.param_specs:
            if get_name:
                param_list.append('ext_start')
            else:
                if self.type == 'capsule':
                    if not self.param_inited and self.param_specs['ext_start'].get('rel', False):
                        self.param_specs['ext_start']['lb'] += self.ext_start
                        self.param_specs['ext_start']['ub'] += self.ext_start
                        self.param_specs['ext_start']['lb'] = max(self.param_specs['ext_start']['lb'], self.param_specs['ext_start'].get('min', -np.inf))
                        self.param_specs['ext_start']['ub'] = min(self.param_specs['ext_start']['ub'], self.param_specs['ext_start'].get('max', np.inf))
                    ext_start = normalize_range(self.ext_start, self.param_specs['ext_start']['lb'], self.param_specs['ext_start']['ub'])
                    param_list.append(ext_start.flatten())
                elif pad_zeros:
                    param_list.append(np.zeros(1))

        if not get_name:
            self.param_inited = True
    """
    def get_params(self, param_list, get_name=False, pad_zeros=False):

        # -------------------------------------------------
        # CAPSULE SIZE
        # -------------------------------------------------
        if 'size' in self.param_specs:
            if get_name:
                if self.type == 'capsule':
                    param_list.append('size')
            else:
                if self.type == 'capsule':

                    if not self.param_inited and self.param_specs['size'].get('rel', False):
                        self.param_specs['size']['lb'] += self.size
                        self.param_specs['size']['ub'] += self.size

                        self.param_specs['size']['lb'] = max(
                            self.param_specs['size']['lb'],
                            self.param_specs['size'].get('min', -np.inf)
                        )

                        self.param_specs['size']['ub'] = min(
                            self.param_specs['size']['ub'],
                            self.param_specs['size'].get('max', np.inf)
                        )

                    size = normalize_range(
                        self.size,
                        self.param_specs['size']['lb'],
                        self.param_specs['size']['ub']
                    )

                    param_list.append(size.flatten())

                elif pad_zeros:
                    param_list.append(np.zeros(1))

        # -------------------------------------------------
        # ELLIPSOID RADIUS
        # -------------------------------------------------
        if 'ellipsoid_radius' in self.param_specs:

            if get_name:

                if self.type == 'ellipsoid':
                    param_list += [
                        'ellipsoid_radius_x',
                        'ellipsoid_radius_y',
                        'ellipsoid_radius_z'
                    ]

            else:

                if self.type == 'ellipsoid':

                    if not self.param_inited and self.param_specs['ellipsoid_radius'].get('rel', False):

                        self.param_specs['ellipsoid_radius']['lb'] += self.ellipsoid_radius
                        self.param_specs['ellipsoid_radius']['ub'] += self.ellipsoid_radius

                        self.param_specs['ellipsoid_radius']['lb'] = np.maximum(
                            self.param_specs['ellipsoid_radius']['lb'],
                            self.param_specs['ellipsoid_radius'].get('min', -np.inf)
                        )

                        self.param_specs['ellipsoid_radius']['ub'] = np.minimum(
                            self.param_specs['ellipsoid_radius']['ub'],
                            self.param_specs['ellipsoid_radius'].get('max', np.inf)
                        )

                    radius = normalize_range(
                        self.ellipsoid_radius,
                        self.param_specs['ellipsoid_radius']['lb'],
                        self.param_specs['ellipsoid_radius']['ub']
                    )

                    param_list.append(radius.flatten())

                elif pad_zeros:
                    param_list.append(np.zeros(3))

        # -------------------------------------------------
        # CAPSULE EXT_START
        # -------------------------------------------------
        if 'ext_start' in self.param_specs:

            if get_name:
                param_list.append('ext_start')

            else:

                if self.type == 'capsule':

                    if not self.param_inited and self.param_specs['ext_start'].get('rel', False):

                        self.param_specs['ext_start']['lb'] += self.ext_start
                        self.param_specs['ext_start']['ub'] += self.ext_start

                        self.param_specs['ext_start']['lb'] = max(
                            self.param_specs['ext_start']['lb'],
                            self.param_specs['ext_start'].get('min', -np.inf)
                        )

                        self.param_specs['ext_start']['ub'] = min(
                            self.param_specs['ext_start']['ub'],
                            self.param_specs['ext_start'].get('max', np.inf)
                        )

                    ext_start = normalize_range(
                        self.ext_start,
                        self.param_specs['ext_start']['lb'],
                        self.param_specs['ext_start']['ub']
                    )

                    param_list.append(ext_start.flatten())

                elif pad_zeros:
                    param_list.append(np.zeros(1))

        if not get_name:
            self.param_inited = True
    """
    def set_params(self, params, pad_zeros=False):
        if 'size' in self.param_specs:
            if self.type == 'capsule':
                self.size = denormalize_range(params[[0]], self.param_specs['size']['lb'], self.param_specs['size']['ub'])
                params = params[1:]
            elif self.type == 'ellipsoid':
                self.size = denormalize_range(params[:3], self.param_specs['size']['lb'], self.param_specs['size']['ub'])
                params = params[3:]
            elif pad_zeros:
                params = params[1:]
        if 'ext_start' in self.param_specs:
            if self.type == 'capsule':
                self.ext_start = denormalize_range(params[[0]], self.param_specs['ext_start']['lb'], self.param_specs['ext_start']['ub'])
                params = params[1:]
            elif pad_zeros:
                params = params[1:]
        return params
    """

    def set_params(self, params, pad_zeros=False):

        # -------------------------------------------------
        # CAPSULE SIZE
        # -------------------------------------------------
        if 'size' in self.param_specs:

            if self.type == 'capsule':

                self.size = denormalize_range(
                    params[[0]],
                    self.param_specs['size']['lb'],
                    self.param_specs['size']['ub']
                )

                params = params[1:]

            elif pad_zeros:
                params = params[1:]

        # -------------------------------------------------
        # ELLIPSOID RADIUS
        # -------------------------------------------------
        if 'ellipsoid_radius' in self.param_specs:

            if self.type == 'ellipsoid':

                self.ellipsoid_radius = denormalize_range(
                    params[:3],
                    self.param_specs['ellipsoid_radius']['lb'],
                    self.param_specs['ellipsoid_radius']['ub']
                )

                # IMPORTANT:
                # MuJoCo ellipsoid size IS the radius vector
                self.size = self.ellipsoid_radius.copy()

                params = params[3:]

            elif pad_zeros:
                params = params[3:]

        # -------------------------------------------------
        # CAPSULE EXT_START
        # -------------------------------------------------
        if 'ext_start' in self.param_specs:

            if self.type == 'capsule':

                self.ext_start = denormalize_range(
                    params[[0]],
                    self.param_specs['ext_start']['lb'],
                    self.param_specs['ext_start']['ub']
                )

                params = params[1:]

            elif pad_zeros:
                params = params[1:]

        return params

class Actuator:

    def __init__(self, node, joint):
        self.node = node
        self.joint = joint
        self.cfg = joint.cfg
        self.joint_name = node.attrib['joint']
        self.name = self.joint_name
        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.gear = float(node.attrib['gear'])

    def parse_param_specs(self):
        self.param_specs =  deepcopy(self.cfg['actuator_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])

    def sync_node(self):
        self.node.attrib['gear'] = f'{self.gear:.6f}'.rstrip('0').rstrip('.')
        self.name = self.joint.name
        self.node.attrib['name'] = self.name
        self.node.attrib['joint'] = self.joint.name

    def get_params(self, param_list, get_name=False):
        if 'gear' in self.param_specs:
            if get_name:
                param_list.append('gear')
            else:
                if not self.param_inited and self.param_specs['gear'].get('rel', False):
                    self.param_specs['gear']['lb'] += self.gear
                    self.param_specs['gear']['ub'] += self.gear
                    self.param_specs['gear']['lb'] = max(self.param_specs['gear']['lb'], self.param_specs['gear'].get('min', -np.inf))
                    self.param_specs['gear']['ub'] = min(self.param_specs['gear']['ub'], self.param_specs['gear'].get('max', np.inf))
                gear = normalize_range(self.gear, self.param_specs['gear']['lb'], self.param_specs['gear']['ub'])
                param_list.append(np.array([gear]))

        if not get_name:
            self.param_inited = True

    def set_params(self, params):
        if 'gear' in self.param_specs:
            self.gear = denormalize_range(params[0].item(), self.param_specs['gear']['lb'], self.param_specs['gear']['ub'])
            params = params[1:]
        return params


class Body:

    def __init__(self, node, parent_body, robot, cfg):
        self.node = node
        self.parent = parent_body
        ## update depth
        if parent_body is not None:
            parent_body.child.append(self)
            parent_body.cind += 1
            self.depth = parent_body.depth + 1
        else:
            self.depth = 0
        self.robot = robot
        self.cfg = cfg
        self.tree = robot.tree
        self.local_coord = robot.local_coord
        self.name = node.attrib['name'] if 'name' in node.attrib else self.parent.name + f'_child{len(self.parent.child)}'
        self.child = []
        self.cind = 0
        self.pos = parse_vec(node.attrib['pos'])
        if self.local_coord and parent_body is not None:
            self.pos += parent_body.pos
        if cfg.get('init_root_from_geom', False):
            self.bone_start = None if parent_body is None else self.pos.copy()
        else:
            self.bone_start = self.pos.copy()
        self.joints = [Joint(x, self) for x in node.findall('joint[@type="hinge"]')] + [Joint(x, self) for x in node.findall('joint[@type="free"]')]
        self.geoms = [Geom(x, self) for x in node.findall('geom[@type="capsule"]')] + [Geom(x, self) for x in node.findall('geom[@type="sphere"]')] + [Geom(x, self) for x in node.findall('geom[@type="ellipsoid"]')]
        self.agent_suffix = self._infer_agent_suffix()
        self.parse_param_specs()
        self.param_inited = False
        # parameters
        self.bone_end = None
        self.bone_offset = None
        ## update height
        self.update_height()

    def __repr__(self):
        return 'body_' + self.name

    def update_height(self):
        '''
        update for the height of current body and its father/grandfather
        '''
        child_height = [child.height for child in self.child]
        if child_height:  # Check if children exist
            self.height = max(child_height) + 1
        else:
            self.height = 0
        if self.parent is not None:
            self.parent.update_height()
    
    def parse_param_specs(self):
        self.param_specs = deepcopy(self.cfg['body_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])
            if name == 'bone_ang':
                specs['lb'] = np.deg2rad(specs['lb'])
                specs['ub'] = np.deg2rad(specs['ub'])

    @staticmethod
    def _extract_agent_suffix(text):
        if text is None:
            return None
        match = re.search(r'_(\d+)$', str(text))
        if match is None:
            return None
        return f"_{match.group(1)}"

    def _infer_agent_suffix(self):
        if self.parent is not None and getattr(self.parent, 'agent_suffix', None) is not None:
            return self.parent.agent_suffix

        suffix = self._extract_agent_suffix(self.name)
        if suffix is not None:
            return suffix

        for joint in self.joints:
            suffix = self._extract_agent_suffix(joint.name)
            if suffix is not None:
                return suffix

        return None

    def reindex(self):
        if self.parent is None:
            base_name = '0'
        else:
            ind = self.parent.child.index(self) + 1
            pname = '' if self.parent.name == '0' else self.parent.name
            base_name = str(ind) + pname

        if self.agent_suffix is not None and not base_name.endswith(self.agent_suffix):
            self.name = f"{base_name}{self.agent_suffix}"
        else:
            self.name = base_name

    def init(self):
        if len(self.child) > 0:
            bone_ends = [x.bone_start for x in self.child]
        else:
            bone_ends = [x.end for x in self.geoms]
        if len(bone_ends) > 0:
            self.bone_end = np.mean(np.stack(bone_ends), axis=0)
            self.bone_offset = self.bone_end - self.bone_start

    def get_actuator_name(self):
        for joint in self.joints:
            if joint.actuator is not None:
                return joint.actuator.name
        return None

    def get_joint_range(self):
        assert len(self.joints) == 1
        return self.joints[0].range

    def sync_node(self):
        pos = self.pos - self.parent.pos if self.local_coord and self.parent is not None else self.pos
        self.node.attrib['name'] = self.name
        self.node.attrib['pos'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in pos])
        for joint in self.joints:
            joint.sync_node()
        for geom in self.geoms:
            geom.sync_node()

    def sync_geom(self):
        for geom in self.geoms:
            geom.bone_start = self.bone_start.copy()
            geom.end = self.bone_end.copy()
            geom.update_start()

    def sync_joint(self):
        if self.parent is not None:
            for joint in self.joints:
                joint.pos = self.pos.copy()

    def rebuild(self):
        if self.parent is not None:
            self.bone_start = self.parent.bone_end.copy()
            self.pos = self.bone_start.copy()
        if self.bone_offset is not None:
            self.bone_end = self.bone_start + self.bone_offset
        if self.parent is None and self.cfg.get('no_root_offset', False):
            self.bone_end = self.bone_start
        self.sync_geom()
        self.sync_joint()

    def get_params(self, param_list, get_name=False, pad_zeros=False, demap_params=False):
        '''
        offset: 2
        joint: 1 (电机扭矩)
        geom: 2 (size, ext_start)
        '''
        if self.bone_offset is not None and 'offset' in self.param_specs:
            if get_name:
                if self.param_specs['offset']['type'] == 'xz':
                    param_list += ['offset_x', 'offset_z']
                elif self.param_specs['offset']['type'] == 'xy':
                    param_list += ['offset_x', 'offset_y']
                else:
                    param_list += ['offset_x', 'offset_y', 'offset_z']
            else:
                if self.param_specs['offset']['type'] == 'xz':
                    offset = self.bone_offset[[0, 2]]
                elif self.param_specs['offset']['type'] == 'xy':
                    offset = self.bone_offset[[0, 1]]
                else:
                    offset = self.bone_offset
                if not self.param_inited and self.param_specs['offset'].get('rel', False):
                    self.param_specs['offset']['lb'] += offset
                    self.param_specs['offset']['ub'] += offset
                    self.param_specs['offset']['lb'] = np.maximum(self.param_specs['offset']['lb'], self.param_specs['offset'].get('min', np.full_like(offset, -np.inf)))
                    self.param_specs['offset']['ub'] = np.minimum(self.param_specs['offset']['ub'], self.param_specs['offset'].get('max', np.full_like(offset, np.inf)))
                offset = normalize_range(offset, self.param_specs['offset']['lb'], self.param_specs['offset']['ub'])
                param_list.append(offset.flatten())

        if self.bone_offset is not None and 'bone_len' in self.param_specs:
            if get_name:
                param_list += ['bone_len']
            else:
                bone_len = np.linalg.norm(self.bone_offset)
                if not self.param_inited and self.param_specs['bone_len'].get('rel', False):
                    self.param_specs['bone_len']['lb'] += bone_len
                    self.param_specs['bone_len']['ub'] += bone_len
                    self.param_specs['bone_len']['lb'] = max(self.param_specs['bone_len']['lb'], self.param_specs['bone_len'].get('min',-np.inf))
                    self.param_specs['bone_len']['ub'] = min(self.param_specs['bone_len']['ub'], self.param_specs['bone_len'].get('max', np.inf))
                bone_len = normalize_range(bone_len, self.param_specs['bone_len']['lb'], self.param_specs['bone_len']['ub'])
                param_list.append(np.array([bone_len]))

        if self.bone_offset is not None and 'bone_ang' in self.param_specs:
            if get_name:
                param_list += ['bone_ang']
            else:
                bone_ang = math.atan2(self.bone_offset[2], self.bone_offset[0])
                if not self.param_inited and self.param_specs['bone_ang'].get('rel', False):
                    self.param_specs['bone_ang']['lb'] += bone_ang
                    self.param_specs['bone_ang']['ub'] += bone_ang
                    self.param_specs['bone_ang']['lb'] = max(self.param_specs['bone_ang']['lb'], self.param_specs['bone_ang'].get('min',-np.inf))
                    self.param_specs['bone_ang']['ub'] = min(self.param_specs['bone_ang']['ub'], self.param_specs['bone_ang'].get('max', np.inf))
                bone_ang = normalize_range(bone_ang, self.param_specs['bone_ang']['lb'], self.param_specs['bone_ang']['ub'])
                param_list.append(np.array([bone_ang]))

        for joint in self.joints:
            joint.get_params(param_list, get_name, pad_zeros)
        if pad_zeros and len(self.joints) == 0:
            param_list.append(np.zeros(1))

        for geom in self.geoms:
            geom.get_params(param_list, get_name, pad_zeros)
        if not get_name:
            self.param_inited = True

        if demap_params and not get_name:
            params = self.robot.demap_params(np.concatenate(param_list))
            return params

    def set_params(self, params, pad_zeros=False, map_params=False):
        if map_params:
            params = self.robot.map_params(params)

        if self.bone_offset is not None and 'offset' in self.param_specs:
            if self.param_specs['offset']['type'] in {'xz', 'xy'}:
                offset = denormalize_range(params[:2], self.param_specs['offset']['lb'], self.param_specs['offset']['ub'])
                if np.all(offset == 0.0):
                    offset[0] += 1e-8
                if self.param_specs['offset']['type'] == 'xz':
                    self.bone_offset[[0, 2]] = offset
                elif self.param_specs['offset']['type'] == 'xy':
                    self.bone_offset[[0, 1]] = offset
                params = params[2:]
            else:
                offset = denormalize_range(params[:3], self.param_specs['offset']['lb'], self.param_specs['offset']['ub'])
                if np.all(offset == 0.0):
                    offset[0] += 1e-8
                self.bone_offset[:] = offset
                params = params[3:]

        if self.bone_offset is not None and 'bone_len' in self.param_specs:
            bone_len = denormalize_range(params[0].item(), self.param_specs['bone_len']['lb'], self.param_specs['bone_len']['ub'])
            bone_len = max(bone_len, 1e-4)
            params = params[1:]
        else:
            bone_len = np.linalg.norm(self.bone_offset)

        if self.bone_offset is not None and 'bone_ang' in self.param_specs:
            bone_ang = denormalize_range(params[0].item(), self.param_specs['bone_ang']['lb'], self.param_specs['bone_ang']['ub'])
            params = params[1:]
        else:
            bone_ang = math.atan2(self.bone_offset[2], self.bone_offset[0])

        if 'bone_len' in self.param_specs or 'bone_ang' in self.param_specs:
            self.bone_offset = np.array([bone_len * math.cos(bone_ang), 0, bone_len * math.sin(bone_ang)])

        for joint in self.joints:
            params = joint.set_params(params, pad_zeros)
        for geom in self.geoms:
            params = geom.set_params(params, pad_zeros)
        # rebuild bone, geom, joint
        self.rebuild()
        return params


class Robot:

    def __init__(self, cfg, xml, is_xml_str=False):
        self.bodies = []
        self.cfg = cfg
        self.param_mapping = cfg.get('param_mapping', 'clip')
        self.tree = None    # xml tree
        self.load_from_xml(xml, is_xml_str)
        self.init_bodies()
        self.param_names = self.get_params(get_name=True)
        self.init_params = self.get_params()

    def load_from_xml(self, xml, is_xml_str=False):
        parser = XMLParser(remove_blank_text=True)
        self.tree = parse(BytesIO(xml) if is_xml_str else xml, parser=parser)
        self.local_coord = self.tree.getroot().find('.//compiler').attrib['coordinate'] == 'local'
        worldbody = self.tree.getroot().find('worldbody')
        root_nodes = list(worldbody.findall('body'))

        # Multi-agent XMLs may have multiple robot roots (e.g., 0_1, 0_2)
        # plus freejoint objects (e.g., box). Keep only articulated robot roots.
        robot_roots = []
        for node in root_nodes:
            joints = node.findall('joint')
            has_robot_joint = any(
                joint.attrib.get('type') in {'hinge', 'slide', 'free'}
                for joint in joints
            )
            if has_robot_joint:
                robot_roots.append(node)

        if not robot_roots and len(root_nodes) > 0:
            # Backward compatibility with single-root XML layouts.
            robot_roots = [root_nodes[0]]

        for root in robot_roots:
            self.add_body(root, None)

    def add_body(self, body_node, parent_body):
        body = Body(body_node, parent_body, self, self.cfg)
        self.bodies.append(body)

        for body_node_c in body_node.findall('body'):
            self.add_body(body_node_c, body)

    def init_bodies(self):
        for body in self.bodies:
            body.init()
        self.sync_node()

    def sync_node(self):
        for body in self.bodies:
            body.reindex()
            body.sync_node()

    @staticmethod
    def _extract_agent_suffix(text):
        if text is None:
            return None
        match = re.search(r'_(\d+)$', str(text))
        if match is None:
            return None
        return f"_{match.group(1)}"

    def _normalize_agent_suffix(self, agent_id=None, body=None):
        if agent_id is not None:
            agent_str = str(agent_id)
            if not agent_str.startswith('_'):
                agent_str = f"_{agent_str}"
            return agent_str

        if body is not None:
            suffix = getattr(body, 'agent_suffix', None)
            if suffix is not None:
                return suffix

            suffix = self._extract_agent_suffix(getattr(body, 'name', None))
            if suffix is not None:
                return suffix

            for joint in getattr(body, 'joints', []):
                suffix = self._extract_agent_suffix(getattr(joint, 'name', None))
                if suffix is not None:
                    return suffix

        return None

    def _find_root_child_for_agent(self, agent_suffix):
        if len(self.bodies) == 0:
            return None

        root = self.bodies[0]
        for child in root.child:
            if getattr(child, 'agent_suffix', None) == agent_suffix:
                return child

        for child in root.child:
            if child.name.endswith(agent_suffix):
                return child

        return None

    def add_child_to_body(self, body, agent_id=None):
        agent_suffix = self._normalize_agent_suffix(agent_id=agent_id, body=body)

        if body == self.bodies[0]:
            if agent_suffix is not None:
                parent_body = self._find_root_child_for_agent(agent_suffix)
                if parent_body is None:
                    raise ValueError(f"Could not resolve root child for agent suffix {agent_suffix}")
            else:
                if len(body.child) == 0:
                    raise ValueError("Root body has no children to clone from")
                parent_body = body.child[0]
            body2clone = parent_body
        else:
            parent_body = body
            body2clone = body

        if agent_suffix is not None and getattr(parent_body, 'agent_suffix', None) not in (None, agent_suffix):
            raise ValueError(
                f"Body {parent_body.name} belongs to {parent_body.agent_suffix}, expected {agent_suffix}"
            )

        child_node = deepcopy(body2clone.node)
        for bnode in child_node.findall('body'):
            child_node.remove(bnode)
        child_body = Body(child_node, parent_body, self, self.cfg)
        if agent_suffix is not None and child_body.agent_suffix is None:
            child_body.agent_suffix = agent_suffix

        actu_node = body.tree.getroot().find("actuator")
        for joint in child_body.joints:
            template_node = actu_node.find(f'motor[@joint="{joint.name}"]')

            if template_node is None:
                template_node = next(
                    (j.actuator.node for j in body2clone.joints if j.actuator is not None),
                    None
                )

            if template_node is None:
                continue

            new_actu_node = deepcopy(template_node)
            actu_node.append(new_actu_node)
            joint.actuator = Actuator(new_actu_node, joint)

        child_body.bone_offset = parent_body.bone_offset.copy()
        child_body.param_specs = deepcopy(parent_body.param_specs)
        child_body.param_inited = True
        child_body.rebuild()
        child_body.sync_node()
        parent_body.node.append(child_node)
        self.bodies.append(child_body)
        self.sync_node()
        
    def remove_body(self, body, agent_id=None):
        if body.parent is None:
            raise ValueError("Cannot remove root body")

        agent_suffix = self._normalize_agent_suffix(agent_id=agent_id, body=body)
        if agent_suffix is not None and getattr(body, 'agent_suffix', None) not in (None, agent_suffix):
            raise ValueError(
                f"Body {body.name} belongs to {body.agent_suffix}, expected {agent_suffix}"
            )

        body.node.getparent().remove(body.node)
        body.parent.child.remove(body)
        body.parent.update_height()
        self.bodies.remove(body)
        actu_node = body.tree.getroot().find("actuator")
        for joint in body.joints:
            if joint.actuator is not None and joint.actuator.node.getparent() is not None:
                actu_node.remove(joint.actuator.node)
        del body
        self.sync_node()

    def write_xml(self, fname):
        self.tree.write(fname, pretty_print=True)

    def export_xml_string(self):
        return etree.tostring(self.tree, pretty_print=True)

    def demap_params(self, params):
        #if not np.all((params <= 1.0) & (params >= -1.0)):
        #    print(f'param out of bounds: {params}')
        params = np.clip(params, -1.0, 1.0)
        if self.param_mapping == 'sin':
            params = np.arcsin(params) / (0.5 * np.pi)
        return params

    def get_params(self, get_name=False):
        param_list = []
        for body in self.bodies:
            body.get_params(param_list, get_name)

        if not get_name:
            params = np.concatenate(param_list)
            params = self.demap_params(params)
        else:
            params = np.array(param_list)
        return params

    def map_params(self, params):
        if self.param_mapping == 'clip':
            params = np.clip(params, -1.0, 1.0)
        elif self.param_mapping == 'sin':
            params = np.sin(params * (0.5 * np.pi))
        return params

    def set_params(self, params):
        # clip params to range
        params = self.map_params(params)
        for body in self.bodies:
            params = body.set_params(params)
        assert(len(params) == 0)    # all parameters need to be consumed!
        self.sync_node()

    def rebuild(self):
        for body in self.bodies:
            body.rebuild()
            body.sync_node()

    def get_gnn_edges(self):
        edges = []
        for i, body in enumerate(self.bodies):
            if body.parent is not None:
                j = self.bodies.index(body.parent)
                edges.append([i, j])
                edges.append([j, i])
        edges = np.stack(edges, axis=1)
        return edges
    
    def build_adjacency_list(self):
        edges = self.get_gnn_edges().T
        adjacency_list = {}
        for edge in edges:
            if edge[0] in adjacency_list:
                adjacency_list[edge[0]].append(edge[1])
            else:
                adjacency_list[edge[0]] = [edge[1]]
        return adjacency_list
    
    def build_adjacency_matrix(self, node_indices=None):
        edges = self.get_gnn_edges().T

        if node_indices is None:
            num_nodes = len(self.bodies)
            adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
            for edge in edges:
                adjacency_matrix[edge[0]][edge[1]] = 1
            return adjacency_matrix

        node_indices = np.asarray(node_indices, dtype=np.int64).reshape(-1)
        num_nodes = int(node_indices.shape[0])
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        local_index = {int(global_idx): local_idx for local_idx, global_idx in enumerate(node_indices)}

        for src, dst in edges:
            src = int(src)
            dst = int(dst)
            if src in local_index and dst in local_index:
                adjacency_matrix[local_index[src]][local_index[dst]] = 1

        return adjacency_matrix
    
    def get_shortest_distances(self):
        num_nodes = len(self.bodies)
        adjacency_list = self.build_adjacency_list()
        distances = np.zeros((num_nodes, num_nodes), dtype=int)
        
        def dfs(i, current_idx, last_idx, dis):
            distances[i][current_idx] = dis
            for next_idx in adjacency_list[current_idx]:
                if next_idx != last_idx: 
                    dfs(i, next_idx, current_idx, dis + 1)
        
        for i in range(num_nodes):
            dfs(i, i, -1, 0)
            
        return distances
    
    def get_laplacian_position_encoding(self, pos_enc_dim=4, robust=False, node_indices=None):
        adjacency_matrix = self.build_adjacency_matrix(node_indices=node_indices)
        num_nodes = adjacency_matrix.shape[0]

        if robust:
            adjacency_matrix = adjacency_matrix.astype(np.float64)

            if num_nodes == 0:
                return np.zeros((0, pos_enc_dim), dtype=np.float64)

            # Use an undirected graph for Laplacian PE and avoid self loops.
            adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
            np.fill_diagonal(adjacency_matrix, 0.0)

            degree = adjacency_matrix.sum(axis=1)
            inv_sqrt_degree = np.zeros_like(degree)
            nz = degree > 0
            inv_sqrt_degree[nz] = 1.0 / np.sqrt(degree[nz])

            # Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
            L_sym = np.eye(num_nodes) - (
                inv_sqrt_degree[:, None] * adjacency_matrix * inv_sqrt_degree[None, :]
            )

            # Dense eigendecomposition is robust for these small morphology graphs.
            vals, vecs = np.linalg.eigh(L_sym)
            order = np.argsort(vals)
            vecs = vecs[:, order]

            # Skip the first (smallest-eigenvalue) eigenvector.
            k = min(pos_enc_dim, max(0, num_nodes - 1))
            position_encoding = vecs[:, 1:1 + k] if k > 0 else np.zeros((num_nodes, 0), dtype=np.float64)

            # Random sign flip keeps the representation sign-invariant.
            if position_encoding.shape[1] > 0:
                flip_signs = np.random.choice([-1, 1], size=(1, position_encoding.shape[1]))
                position_encoding = position_encoding * flip_signs

            if k < pos_enc_dim:
                padding = np.zeros((num_nodes, pos_enc_dim - k), dtype=position_encoding.dtype)
                position_encoding = np.hstack([position_encoding, padding])

            return position_encoding

        degree_matrix = np.diag(adjacency_matrix.sum(axis=1))

        # 计算归一化的拉普拉斯矩阵 L_sym = I - D^-1/2 * A * D^-1/2
        D_inv_sqrt = fractional_matrix_power(degree_matrix, -0.5)
        I = np.eye(num_nodes)
        L_sym = I - np.dot(np.dot(D_inv_sqrt, adjacency_matrix), D_inv_sqrt)

        # 使用稀疏矩阵求解器计算归一化拉普拉斯矩阵的特征值和特征向量
        k = min(pos_enc_dim, num_nodes - 1)
        vals, vecs = eigsh(L_sym, k=k+1, which='SM')

        # 忽略第一个特征向量（对应于最小的特征值）
        position_encoding = vecs[:, 1:k+1]

        flip_signs = np.random.choice([-1, 1], size=(position_encoding.shape[0], 1))
        position_encoding *= flip_signs

        if k < pos_enc_dim:
            padding = np.zeros((num_nodes, pos_enc_dim - k))
            position_encoding = np.hstack([position_encoding, padding])

        return position_encoding

if __name__ == "__main__":
    import os
    import sys
    import time
    import yaml
    sys.path.append(os.getcwd())
    from mujoco_py import load_model_from_path, MjSim, MjViewer

    model_name = 'ant'
    cfg_path = f'khrylib/assets/ant.yml'
    # model = load_model_from_path(f'assets/mujoco_envs/{model_name}.xml')
    yml = yaml.safe_load(open(cfg_path, 'r'))
    cfg = yml['robot']
    xml_robot = Robot(cfg, xml=f'assets/mujoco_envs/{model_name}.xml')
    params_names = xml_robot.get_params(get_name=True)

    params = xml_robot.get_params()
    print(params_names, params)
    new_params = params # + 0.1
    xml_robot.set_params(new_params)
    params_new = xml_robot.get_params()

    os.makedirs('out', exist_ok=True)
    xml_robot.write_xml(f'out/{model_name}_test.xml')

    model = load_model_from_path(f'out/{model_name}_test.xml')
    sim = MjSim(model)
    viewer = MjViewer(sim)
    viewer.cam.distance = 10

    while True:
        sim.data.qpos[2] = 1.0
        sim.data.qpos[7:] = np.pi / 6
        sim.forward()
        viewer.render()