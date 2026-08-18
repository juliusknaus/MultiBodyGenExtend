import math
import pickle
import time
import imageio
#from mujoco_py import GlfwContext
from khrylib.utils import *
from khrylib.utils.torch import *
from khrylib.rl.agents import AgentPPO
from torch.utils.tensorboard import SummaryWriter
from design_opt.tasks import task_dict
from design_opt.envs import env_dict
from design_opt.models.bodygen_policy import BodyGenPolicy
from design_opt.models.bodygen_critic import BodyGenValue
from design_opt.utils.logger import LoggerRLV1
from design_opt.utils.tools import TrajBatchDisc
import multiprocessing
from khrylib.rl.core.running_norm import RunningNorm
from torch.optim.lr_scheduler import LambdaLR
import csv
import json
import ast
import re
import glob


import wandb

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i <= 1 or i == 4 or i >= 7 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]


class BodyGenAgent(AgentPPO):

    def __init__(self, cfg, dtype, device, seed, num_threads, training=True, checkpoint=0):
        self.cfg = cfg
        self.training = training
        self.device = device
        self.loss_iter = 0
        self.setup_env()
        # Prefer cfg.num_agents if present, otherwise read from env, default to 1.
        self.num_agents = max(1, int(getattr(self.cfg, 'num_agents', getattr(self.env, 'num_agents', 1))))
        self.env.seed(seed)


        #for plotting saving purposes: 
        self.info = None
        # self.setup_task()
        self.setup_logger()
        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()
        if self.cfg.norm_return:
            self.design_ret_norm = RunningNorm(1, demean=self.cfg.planner_demean, clip=False)
            self.control_ret_norm = RunningNorm(1, demean=False, clip=False)
            self.ret_norm = RunningNorm(1, demean=False, clip=False)
        else:
            self.design_ret_norm = self.control_ret_norm = self.ret_norm = None
        if cfg.uni_obs_norm:
            self.obs_norm = RunningNorm(self.state_dim).to(self.device)
        else:
            self.obs_norm = None
        self.transfer_init_applied = False
        if checkpoint != 0:
            self.load_checkpoint(checkpoint)
        else:
            self._maybe_transfer_init_from_single_checkpoint()
        super().__init__(env=self.env, dtype=dtype, device=device, running_state=self.running_state,
                         custom_reward=None, logger_cls=LoggerRLV1, traj_cls=TrajBatchDisc, num_threads=num_threads,
                         policy_net=self.policy_net, value_net=self.value_net,
                         optimizer_policy=self.optimizer_policy, optimizer_value=self.optimizer_value, opt_num_epochs=cfg.num_optim_epoch,
                         gamma=cfg.gamma, tau=cfg.tau, clip_epsilon=cfg.clip_epsilon,
                         policy_grad_clip=[(self.policy_net.parameters(), 40)],
                         use_mini_batch=cfg.mini_batch_size < cfg.min_batch_size, mini_batch_size=cfg.mini_batch_size)
        # For IPPO-style multi-agent cases: sample_modules / update_modules must cover all per-agent networks.
        if self._is_multi_agent_case():
            self.sample_modules = list(self.policy_nets)
            self.update_modules = list(self.policy_nets) + list(self.value_nets)

    ## Setting Ups        
    def setup_env(self):
        env_class = env_dict[self.cfg.env_name]
        self.env = env = env_class(self.cfg, self)
        self.attr_fixed_dim = env.attr_fixed_dim
        self.attr_design_dim = env.attr_design_dim
        self.sim_obs_dim = env.sim_obs_dim
        self.state_dim = self.attr_fixed_dim + self.sim_obs_dim + self.attr_design_dim
        self.control_action_dim = env.control_action_dim
        self.skel_num_action = env.skel_num_action
        self.action_dim = self.control_action_dim + self.attr_design_dim
        self.running_state = None

    def _env_name(self):
        return str(getattr(self.cfg, 'env_name', '')).lower()

    def _is_multi_case(self):
        return 'multi' in self._env_name()

    def _is_adversarial_case(self):
        return 'adversarial' in self._env_name()

    def _is_multi_agent_case(self):
        return (self._is_multi_case() or self._is_adversarial_case()) and self.num_agents > 1

    def _is_benchmark_mode(self):
        return bool(getattr(self.cfg, 'benchmark', False))

    def _state_stage_id(self, state):
        stage_raw = state[2]
        stage_arr = np.asarray(stage_raw).reshape(-1)
        if stage_arr.size == 0:
            return 2
        return int(stage_arr[0])

    def _state_num_nodes(self, state):
        num_nodes_raw = state[3]
        num_nodes_arr = np.asarray(num_nodes_raw).reshape(-1)
        if num_nodes_arr.size == 0:
            return int(state[0].shape[0])
        return int(num_nodes_arr[0])

    def _sample_benchmark_transform_action(self, state):
        stage_id = self._state_stage_id(state)
        num_nodes = self._state_num_nodes(state)
        action = np.zeros((num_nodes, self.action_dim + 1), dtype=np.float64)
        return action

    def setup_task(self):
        cfg = self.cfg


    def setup_logger(self):
        cfg = self.cfg
        self.tb_logger = SummaryWriter(cfg.tb_dir) if self.training else None
        self.logger = create_logger(os.path.join(cfg.log_dir, f'log_{"train" if self.training else "eval"}.txt'), file_handle=True)
        self.best_rewards = -1000.0
        self.save_best_flag = False
        self.best_agent_rewards = [-float('inf')] * self.num_agents
        self.save_best_agent_flags = [False] * self.num_agents
        
    def setup_policy(self):
        cfg = self.cfg
        if self._is_multi_agent_case():
            # IPPO: one independent policy network per agent
            self.policy_nets = [BodyGenPolicy(cfg.policy_specs, self) for _ in range(self.num_agents)]
            for net in self.policy_nets:
                to_device(self.device, net)
            self.policy_net = self.policy_nets[0]  # kept for base-class compatibility
        else:
            self.policy_net = BodyGenPolicy(cfg.policy_specs, self)
            to_device(self.device, self.policy_net)
            self.policy_nets = [self.policy_net]

    def setup_value(self):
        cfg = self.cfg
        if self._is_multi_agent_case():
            # IPPO: one independent value network per agent
            self.value_nets = [BodyGenValue(cfg.value_specs, self) for _ in range(self.num_agents)]
            for net in self.value_nets:
                to_device(self.device, net)
            self.value_net = self.value_nets[0]  # kept for base-class compatibility
        else:
            self.value_net = BodyGenValue(cfg.value_specs, self)
            to_device(self.device, self.value_net)
            self.value_nets = [self.value_net]

    def setup_optimizer(self):
        cfg = self.cfg
        if self._is_multi_agent_case():
            # IPPO: one optimizer per agent
            def _make_policy_opt(net):
                if cfg.policy_optimizer == 'Adam':
                    return torch.optim.Adam(net.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
                return torch.optim.SGD(net.parameters(), lr=cfg.policy_lr, momentum=cfg.policy_momentum, weight_decay=cfg.policy_weightdecay)
            def _make_value_opt(net):
                if cfg.value_optimizer == 'Adam':
                    return torch.optim.Adam(net.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
                return torch.optim.SGD(net.parameters(), lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)
            self.optimizer_policies = [_make_policy_opt(net) for net in self.policy_nets]
            self.optimizer_values  = [_make_value_opt(net)  for net in self.value_nets]
            self.optimizer_policy  = self.optimizer_policies[0]  # base-class compat
            self.optimizer_value   = self.optimizer_values[0]
            if cfg.lr_decay:
                self.scheduler_policies = [LambdaLR(opt, lr_lambda=lambda epoch: 1 - epoch / cfg.max_epoch_num) for opt in self.optimizer_policies]
                self.scheduler_values   = [LambdaLR(opt, lr_lambda=lambda epoch: 1 - epoch / cfg.max_epoch_num) for opt in self.optimizer_values]
            else:
                self.scheduler_policies = None
                self.scheduler_values   = None
            self.scheduler_policy = self.scheduler_policies[0] if self.scheduler_policies else None
            self.scheduler_value  = self.scheduler_values[0]  if self.scheduler_values  else None
        else:
            # single-case: unchanged
            if cfg.policy_optimizer == 'Adam':
                self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
            else:
                self.optimizer_policy = torch.optim.SGD(self.policy_net.parameters(), lr=cfg.policy_lr, momentum=cfg.policy_momentum, weight_decay=cfg.policy_weightdecay)
            if cfg.value_optimizer == 'Adam':
                self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
            else:
                self.optimizer_value = torch.optim.SGD(self.value_net.parameters(), lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)
            if self.cfg.lr_decay:
                self.scheduler_policy = LambdaLR(self.optimizer_policy, lr_lambda=lambda epoch: 1 - epoch / self.cfg.max_epoch_num)
                self.scheduler_value  = LambdaLR(self.optimizer_value,  lr_lambda=lambda epoch: 1 - epoch / self.cfg.max_epoch_num)
            else:
                self.scheduler_policy = None
                self.scheduler_value  = None
            self.optimizer_policies = [self.optimizer_policy]
            self.optimizer_values   = [self.optimizer_value]

    ## Sampling
    def sample(self, min_batch_size, mean_action=False, render=False, nthreads=None):
        if nthreads is None:
            nthreads = self.num_threads
        t_start = time.time()

        to_test(*self.sample_modules)
        if self.cfg.uni_obs_norm:
            self.obs_norm.eval()
            self.obs_norm.to('cpu')
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / nthreads)) ## 共同采样 min_batch_size
                queue = multiprocessing.Queue()
                memories = [None] * nthreads
                loggers = [None] * nthreads
                
                for i in range(nthreads-1):
                    
                    worker_args = (i+1, queue, thread_batch_size, mean_action, render)
                    
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                    
                memories[0], loggers[0] = self.sample_worker(0, None, thread_batch_size, mean_action, render)
                for i in range(nthreads - 1):
                    pid, worker_memory, worker_logger = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                traj_batch = self.traj_cls(memories)
                logger = self.logger_cls.merge(loggers, **self.logger_kwargs)
        

        logger.sample_time = time.time() - t_start
        return traj_batch, logger
    
    ## Per worker sampling
    def sample_worker(self, pid, queue, min_batch_size, mean_action, render):
        env_name = str(getattr(self.cfg, 'env_name', '')).lower()
        is_multi_case = 'multi' in env_name
        is_adversarial_case = 'adversarial' in env_name
        use_multi_agent_case = (is_multi_case or is_adversarial_case) and self.num_agents > 1
        benchmark_mode = self._is_benchmark_mode()
        if not use_multi_agent_case:
        ## make seed for the worker
            if pid > 0:
                torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
                if hasattr(self.env, 'np_random'):
                    self.env.np_random.seed(self.env.np_random.randint(5000) * pid)
            
            memory = Memory()
            logger = self.logger_cls(**self.logger_kwargs)

            while logger.num_steps < min_batch_size:
                
                state = self.env.reset()
                logger.start_episode(self.env)
                i = 0
                while True:
                    stage_id = self._state_stage_id(state)
                    use_random_transform = benchmark_mode and stage_id in (0, 1)

                    if use_random_transform:
                        action = self._sample_benchmark_transform_action(state)
                        use_mean_action = False
                    else:
                        state_var = tensorfy([state])
                        
                        ## do obs norm (none-updated)
                        if self.cfg.uni_obs_norm:
                            state_var = self.normalize_observation(state_var)

                        use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                        action = self.policy_net.select_action(state_var, use_mean_action).numpy().astype(np.float64)

                    next_state, env_reward, termination, truncation, info = self.env.step(action)
                    reward_components = info.get('reward_components')
                    if reward_components is None:
                        reward_components = {
                            name: 0.0 for name in self.logger_cls.REWARD_COMPONENT_NAMES
                        }
                    else:
                        reward_components = {
                            name: float(reward_components.get(name, 0.0))
                            for name in self.logger_cls.REWARD_COMPONENT_NAMES
                        }
                    info = dict(info)
                    info['reward_components'] = reward_components
                    # use custom or env reward
                    if self.custom_reward is not None:
                        c_reward, c_info = self.custom_reward(self.env, state, action, env_reward, info)
                        reward = c_reward
                    else:
                        c_reward, c_info = 0.0, np.array([0.0])
                        reward = env_reward
                    # add end reward
                    if self.end_reward and info.get('end', False):
                        reward += self.env.end_reward
                        
                    if info['stage'] == 'execution':
                        reward += self.cfg.reward_shift 
                    
                    # logging
                    logger.step(self.env, env_reward, c_reward, c_info, info)

                    done = (termination or truncation)
                    exp = 1 - use_mean_action
                    memory.push(state, action, termination, done, next_state, reward, exp, 0)
                    i = i + 1
                    if done:
                        break
                    state = next_state 
                logger.end_episode(self.env)        
            logger.end_sampling()
            
            if queue is not None:
                queue.put([pid, memory, logger])
            else:
                return memory, logger

        else:

            ## make seed for the worker
            if pid > 0:
                torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
                if hasattr(self.env, 'np_random'):
                    self.env.np_random.seed(self.env.np_random.randint(5000) * pid)

            logger = self.logger_cls(**self.logger_kwargs)

            # IMPORTANT: store all agent memories separately
            agent_memories = [[] for _ in range(self.num_agents)]
            agent_logs = []

            #print("number of agents: ", self.num_agents)

            while logger.num_steps < min_batch_size:
                self.env.reset()
                states = [self.env.get_agent_obs(agent_id) for agent_id in range(self.num_agents)]
                logger.start_episode(self.env)
                episode_memories = [Memory() for _ in range(self.num_agents)]

                while True:
                    agent_actions = [None] * self.num_agents
                    agent_exps = [0.0] * self.num_agents
                    next_states = list(states)
                    current_states = [None] * self.num_agents
                    agent_rewards = [0.0] * self.num_agents
                    processed_agent_ids = []
                    env_reward = 0.0
                    termination = False
                    truncation = False
                    info = {}
                    reward_component_totals = {
                        name: 0.0 for name in self.logger_cls.REWARD_COMPONENT_NAMES
                    }

                    # Collect all actions from the same environment state first
                    # so multi-agent rollout is parallel in decision time.
                    for agent_id in range(self.num_agents):
                        current_state = states[agent_id]
                        current_states[agent_id] = current_state
                        stage_id = self._state_stage_id(current_state)
                        use_random_transform = benchmark_mode and stage_id in (0, 1)

                        if use_random_transform:
                            action = self._sample_benchmark_transform_action(current_state)
                            use_mean_action = False
                        else:
                            state_var = tensorfy([current_state])

                            if self.cfg.uni_obs_norm:
                                state_var = self.normalize_observation(state_var)

                            use_mean_action = (
                                mean_action
                                or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                            )

                            action_tensor = self.policy_nets[agent_id].select_action(
                                state_var,
                                use_mean_action
                            )

                            action = action_tensor.detach().cpu().numpy().astype(np.float64)

                        agent_actions[agent_id] = action
                        agent_exps[agent_id] = 1 - use_mean_action

                    try:
                        joint_action = np.stack(agent_actions, axis=0)
                    except ValueError:
                        joint_action = list(agent_actions)

                    next_states_batch, env_reward, termination, truncation, info = self.env.multi_step(
                        joint_action,
                        agent_id=None
                    )

                    if isinstance(next_states_batch, list):
                        for i in range(min(len(next_states_batch), self.num_agents)):
                            if next_states_batch[i] is not None:
                                next_states[i] = next_states_batch[i]

                    processed_agent_ids = info.get('processed_agent_ids', list(range(self.num_agents))) if isinstance(info, dict) else list(range(self.num_agents))
                    if len(processed_agent_ids) == 0:
                        processed_agent_ids = list(range(self.num_agents))

                    info_agent_rewards = info.get('agent_rewards', None) if isinstance(info, dict) else None
                    if isinstance(info_agent_rewards, list) and len(info_agent_rewards) == len(processed_agent_ids):
                        for local_idx, aid in enumerate(processed_agent_ids):
                            agent_rewards[aid] = float(info_agent_rewards[local_idx])
                    elif len(processed_agent_ids) > 0:
                        shared_reward = float(env_reward) / float(len(processed_agent_ids))
                        for aid in processed_agent_ids:
                            agent_rewards[aid] = shared_reward

                    info_agent_infos = info.get('agent_infos', None) if isinstance(info, dict) else None
                    if isinstance(info_agent_infos, list) and len(info_agent_infos) == len(processed_agent_ids):
                        for local_idx, aid in enumerate(processed_agent_ids):
                            agent_info = info_agent_infos[local_idx] if isinstance(info_agent_infos[local_idx], dict) else {}
                            agent_reward_components = agent_info.get('reward_components', {})
                            for name in self.logger_cls.REWARD_COMPONENT_NAMES:
                                reward_component_totals[name] += float(agent_reward_components.get(name, 0.0))
                    else:
                        agent_reward_components = info.get('reward_components', {}) if isinstance(info, dict) else {}
                        for name in self.logger_cls.REWARD_COMPONENT_NAMES:
                            reward_component_totals[name] += float(agent_reward_components.get(name, 0.0))

                    if self.custom_reward is not None:
                        processed_actions = [agent_actions[i] for i in processed_agent_ids]
                        reward_states = [
                            current_states[i] if current_states[i] is not None else states[i]
                            for i in range(self.num_agents)
                        ]
                        try:
                            joint_action = np.stack(processed_actions, axis=0)
                        except ValueError:
                            joint_action = processed_actions
                        reward_signal = env_reward
                        if is_adversarial_case and processed_agent_ids:
                            reward_signal = float(np.mean([agent_rewards[i] for i in processed_agent_ids]))
                        c_reward, c_info = self.custom_reward(
                            self.env, reward_states, joint_action, reward_signal, info
                        )
                        rewards = {i: c_reward for i in processed_agent_ids}
                    else:
                        c_reward, c_info = 0.0, np.array([0.0])
                        rewards = {i: agent_rewards[i] for i in processed_agent_ids}

                    if is_adversarial_case and processed_agent_ids:
                        env_reward = float(np.mean([agent_rewards[i] for i in processed_agent_ids]))
                        reward_component_totals = {
                            name: reward_component_totals[name] / float(len(processed_agent_ids))
                            for name in reward_component_totals
                        }

                    if self.end_reward and info.get('end', False):
                        for i in processed_agent_ids:
                            rewards[i] += self.env.end_reward

                    if info['stage'] == 'execution':
                        for i in processed_agent_ids:
                            rewards[i] += self.cfg.reward_shift

                    if info:
                        info = dict(info)
                        info['reward_components'] = reward_component_totals
                        if is_adversarial_case and processed_agent_ids:
                            # Keep per-agent rewards for adversarial logging/selection.
                            info['agent_rewards'] = [float(agent_rewards[i]) for i in processed_agent_ids]

                    logger.step(self.env, env_reward, c_reward, c_info, info)

                    done = (termination or truncation)

                    for agent_id in processed_agent_ids:
                        episode_memories[agent_id].push(
                            current_states[agent_id],
                            agent_actions[agent_id],
                            termination,
                            done,
                            next_states[agent_id],
                            rewards[agent_id],
                            agent_exps[agent_id],
                            agent_id
                        )

                    if done:
                        break

                    states = next_states

                logger.end_episode(self.env)

                for agent_id in range(self.num_agents):
                    agent_memories[agent_id].append(episode_memories[agent_id])

            logger.end_sampling()

            # ============================================================
            # FLATTEN for PPO compatibility
            # (your PPO pipeline still expects a single batch object)
            # ============================================================
            flat_memory = Memory()
            for mem_list in agent_memories:
                for mem in mem_list:
                    for transition in mem.memory:
                        flat_memory.push(*transition)

            if queue is not None:
                queue.put([pid, flat_memory, logger])
            else:
                return flat_memory, logger
    



    



    def optimize(self, epoch):
        info = self.optimize_policy(epoch)
        if self._is_multi_agent_case() and self.scheduler_policies is not None:
            for s in self.scheduler_policies:
                s.step()
            for s in self.scheduler_values:
                s.step()
        else:
            if self.scheduler_policy is not None:
                self.scheduler_policy.step()
            if self.scheduler_value is not None:
                self.scheduler_value.step()
        self.log_optimize_policy(epoch, info)

    def optimize_policy(self, epoch):
        """generate multiple trajectories that reach the minimum batch_size"""
        t0 = time.time()
        batch, log = self.sample(self.cfg.min_batch_size)

        """update networks"""
        t1 = time.time()
        self.update_params(batch)
        t2 = time.time()

        """evaluate policy"""
        _, log_eval = self.sample(self.cfg.eval_batch_size, mean_action=True)
        t3 = time.time() 

        info = {
            'log': log, 'log_eval': log_eval, 'T_sample': t1 - t0, 'T_update': t2 - t1, 'T_eval': t3 - t2, 'T_total': t3 - t0,  **self.info 
        }
        return info
    
    def estimate_advantages(self, states, next_states, rewards, next_terminations, next_dones, state_types, next_state_types, agent_ids=None, value_net=None):
        """Compute GAE advantages.  Pass value_net explicitly for IPPO per-agent critic."""
        _value_net = value_net if value_net is not None else self.value_net
        design_masks = (state_types!=2).bool().to(self.device)
        control_masks = (state_types==2).bool().to(self.device)
        next_design_masks = (next_state_types!=2).bool().to(self.device)
        next_control_masks = (next_state_types==2).bool().to(self.device)

        self.design_ret_norm.to(self.device)
        self.control_ret_norm.to(self.device)

        with to_test(*self.update_modules):
            with torch.no_grad():
                values = []
                chunk = 10000
                for i in range(0, len(states), chunk):
                    states_i = states[i:min(i + chunk, len(states))]
                    values_i = _value_net(states_i)
                    current_range = np.arange(i, min(i + chunk, len(states)))
                    if self.design_ret_norm is not None:
                        local_design_masks = design_masks[current_range]
                        values_i[local_design_masks] = self.design_ret_norm.unscale(values_i[local_design_masks])
                    if self.control_ret_norm is not None:
                        local_control_masks = control_masks[current_range]
                        values_i[local_control_masks] = self.control_ret_norm.unscale(values_i[local_control_masks])
                    values.append(values_i)
                values = torch.cat(values)

                next_values = torch.zeros_like(values)
                # For single-agent data (IPPO) or single case: sequential bootstrap
                next_values[:-1] = values[1:]

                indices = torch.where(next_dones)[0]
                compute_next_states = [next_states[i] for i in indices]
                if compute_next_states:
                    computed_values = _value_net(compute_next_states)
                    if self.design_ret_norm is not None:
                        local_next_design_masks = next_design_masks[indices]
                        computed_values[local_next_design_masks] = self.design_ret_norm.unscale(computed_values[local_next_design_masks])
                    if self.control_ret_norm is not None:
                        local_next_control_masks = next_control_masks[indices]
                        computed_values[local_next_control_masks] = self.control_ret_norm.unscale(computed_values[local_next_control_masks])
                    next_values[indices] = computed_values
                        
        self.design_ret_norm.to('cpu')
        self.control_ret_norm.to('cpu')
        
        device = rewards.device
        rewards, next_terminations, next_dones, values, next_values = batch_to(torch.device('cpu'), rewards, next_terminations, next_dones, values, next_values)
        design_masks, control_masks = batch_to(torch.device('cpu'), design_masks, control_masks)
        
        tensor_type = type(rewards)
        deltas = tensor_type(rewards.size(0), 1)
        advantages = tensor_type(rewards.size(0), 1)
        design_returns = tensor_type(rewards.size(0), 1)

        next_advantage = 0
        next_design_return = 0
        for i in reversed(range(rewards.size(0))):
            deltas[i] = rewards[i] + self.gamma * next_values[i] * (1 - next_terminations[i]) - values[i]
            advantages[i] = next_advantage = deltas[i] + self.gamma * self.tau * next_advantage * (1 - next_dones[i])
            design_returns[i] = next_design_return = rewards[i] + next_design_return * (1 - next_dones[i])

        design_advantages = design_returns - values
        returns = values + advantages
        
        if self.design_ret_norm is not None:
            returns[design_masks] = self.design_ret_norm(design_returns[design_masks])
        else:
            returns[design_masks] = design_returns[design_masks]
            
        if self.control_ret_norm is not None:
            returns[control_masks] = self.control_ret_norm(returns[control_masks])
        
        if self.cfg.norm_advantage:
            advantages[design_masks] = (design_advantages[design_masks] - design_advantages[design_masks].mean()) / design_advantages[design_masks].std()
            advantages[control_masks] = (advantages[control_masks] - advantages[control_masks].mean()) / advantages[control_masks].std()

        advantages, returns = batch_to(device, advantages, returns)
        return advantages, returns

    def update_params(self, batch):
        t0 = time.time()

        to_train(*self.update_modules)

        states       = tensorfy(batch.states,       self.device)
        next_states  = tensorfy(batch.next_states,  self.device)
        actions      = tensorfy(batch.actions,      self.device)
        rewards           = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        next_terminations = torch.from_numpy(batch.next_terminations).to(self.dtype).to(self.device)
        next_dones        = torch.from_numpy(batch.next_dones).to(self.dtype).to(self.device)
        exps              = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        agent_ids = torch.from_numpy(batch.agent_ids).long().to(self.device) if hasattr(batch, 'agent_ids') else None

        if self.cfg.uni_obs_norm:
            self.obs_norm.to(self.device)
            self.obs_norm.train()
            states = self.normalize_observation(states)
            self.obs_norm.eval()
            next_states = self.normalize_observation(next_states)

        if self._is_multi_agent_case() and agent_ids is not None:
            # IPPO: update each agent independently with its own networks
            agent_ids_np = agent_ids.cpu().numpy()
            for aid in range(self.num_agents):
                indices = np.where(agent_ids_np == aid)[0]
                if len(indices) == 0:
                    continue
                self._update_policy_impl(
                    index_select_list(states,      indices),
                    index_select_list(next_states, indices),
                    rewards[indices],
                    next_terminations[indices],
                    next_dones[indices],
                    index_select_list(actions, indices),
                    exps[indices],
                    policy_net=self.policy_nets[aid],
                    value_net=self.value_nets[aid],
                    optimizer_policy=self.optimizer_policies[aid],
                    optimizer_value=self.optimizer_values[aid],
                )
        else:
            self._update_policy_impl(states, next_states, rewards, next_terminations,
                                     next_dones, actions, exps)

        return time.time() - t0
    
    def normalize_observation(self, x):
        obs, edges, use_transform_action, num_nodes, body_ind, body_depths, body_heights, distances, lapPE = zip(*x)
        obs_cat = torch.cat(obs)
        obs_norm = self.obs_norm(obs_cat)
        indices = np.cumsum(num_nodes)
        obs_split = [obs_norm[start:end] for start, end in zip([0] + list(indices[:-1]), indices)]
        x = [list(item) for item in zip(obs_split, edges, use_transform_action, num_nodes, body_ind, body_depths, body_heights, distances, lapPE)]
        return x

    def _to_float_or_nan(self, value):
        if value is None:
            return float('nan')
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        return float('nan')

    def _append_step_values(self, series_dict, step_values, step_idx=None):
        if step_idx is None:
            step_idx = len(next(iter(series_dict.values()))) if series_dict else 0

        # If a series key appears late, backfill missing previous steps with NaN.
        step_idx = max(0, int(step_idx))
        for key in list(series_dict.keys()):
            current_len = len(series_dict[key])
            if current_len < step_idx:
                series_dict[key].extend([float('nan')] * (step_idx - current_len))
            series_dict[key].append(self._to_float_or_nan(step_values.get(key)))

        for key, value in step_values.items():
            if key not in series_dict:
                series_dict[key] = [float('nan')] * step_idx
                series_dict[key].append(self._to_float_or_nan(value))

    def _collect_reward_values_from_info(self, info):
        reward_values = {}
        if not isinstance(info, dict):
            return reward_values

        reward_components = info.get('reward_components')
        if isinstance(reward_components, dict):
            for name, value in reward_components.items():
                reward_values[f'reward_components/{name}'] = value

        for key, value in info.items():
            key_l = str(key).lower()
            if 'reward' not in key_l or key == 'reward_components':
                continue

            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    reward_values[f'{key}/{sub_key}'] = sub_value
            elif isinstance(value, (list, tuple, np.ndarray)):
                continue
            else:
                reward_values[str(key)] = value

        return reward_values

    def _plot_time_series(self, x_values, series_dict, out_path, title, xlabel='Step'):
        if not series_dict:
            return False

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.logger.info(f'Could not import matplotlib for plotting {out_path}: {exc}')
            return False

        x = np.asarray(x_values)
        has_valid_series = False
        fig, ax = plt.subplots(figsize=(12, 6))
        for key, values in series_dict.items():
            y = np.asarray(values, dtype=np.float64)
            if y.size == 0 or np.all(np.isnan(y)):
                continue
            n = min(x.size, y.size)
            if n == 0:
                continue
            has_valid_series = True
            ax.plot(x[:n], y[:n], label=key)

        if not has_valid_series:
            plt.close(fig)
            return False

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        if len(series_dict) <= 12:
            ax.legend(loc='best')
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8)

        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return True

    def _plot_eval_rows(self, eval_rows, out_dir):
        if len(eval_rows) == 0:
            return

        os.makedirs(out_dir, exist_ok=True)

        steps = np.asarray([row['step'] for row in eval_rows], dtype=np.int32)
        scalar_series = {
            'reward': np.asarray([row['reward'] for row in eval_rows], dtype=np.float64).tolist(),
            'done': np.asarray([float(row['done']) for row in eval_rows], dtype=np.float64).tolist(),
            'sim_time': np.asarray([row['sim_time'] for row in eval_rows], dtype=np.float64).tolist(),
        }
        self._plot_time_series(
            x_values=steps,
            series_dict=scalar_series,
            out_path=os.path.join(out_dir, 'eval_scalars.png'),
            title='Eval Scalars Over Time'
        )

        vector_fields = ['obs', 'action', 'qpos', 'qvel']
        for field in vector_fields:
            norms = []
            first_values = []
            for row in eval_rows:
                vec = self._flatten_numeric_values(row[field])
                if vec.size == 0:
                    norms.append(float('nan'))
                    first_values.append(float('nan'))
                else:
                    norms.append(float(np.linalg.norm(vec)))
                    first_values.append(float(vec[0]))

            self._plot_time_series(
                x_values=steps,
                series_dict={f'{field}_l2_norm': norms, f'{field}_first_value': first_values},
                out_path=os.path.join(out_dir, f'eval_{field}.png'),
                title=f'Eval {field} Signals Over Time'
            )

    def _flatten_numeric_values(self, value):
        flat_values = []

        def _visit(x):
            if x is None:
                return
            if isinstance(x, (int, float, np.integer, np.floating, bool, np.bool_)):
                flat_values.append(float(x))
                return
            if isinstance(x, np.ndarray):
                for y in x.reshape(-1):
                    _visit(y)
                return
            if isinstance(x, (list, tuple)):
                for y in x:
                    _visit(y)
                return

        _visit(value)
        return np.asarray(flat_values, dtype=np.float64)

    def _plot_eval_csv_from_file(self, csv_path, out_dir):
        if not os.path.isfile(csv_path):
            return

        eval_rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    eval_rows.append({
                        'step': int(row['step']),
                        'reward': float(row['reward']),
                        'done': row['done'].strip().lower() in ('true', '1'),
                        'sim_time': float(row['sim_time']),
                        'obs': ast.literal_eval(row['obs']),
                        'action': ast.literal_eval(row['action']),
                        'qpos': ast.literal_eval(row['qpos']),
                        'qvel': ast.literal_eval(row['qvel']),
                    })
                except Exception:
                    continue

        self._plot_eval_rows(eval_rows, out_dir)

    def get_perm_batch_design(self, states):
        inds = [[], [], []]
        for i, x in enumerate(states):
            use_transform_action = x[2]
            inds[use_transform_action.item()].append(i)
        perm = np.array(inds[0] + inds[1] + inds[2])
        return perm, LongTensor(perm).to(self.device)
    
    """
    def update_policy(self, states, next_states, rewards, next_terminations, next_dones, actions, exps, agent_ids=None):
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = []
                chunk = 10000
                for i in range(0, len(states), chunk):
                    states_i = states[i:min(i + chunk, len(states))]
                    actions_i = actions[i:min(i + chunk, len(states))]
                    fixed_log_probs_i = self.policy_net.get_log_prob(states_i, actions_i)
                    fixed_log_probs.append(fixed_log_probs_i)
                fixed_log_probs = torch.cat(fixed_log_probs)
        num_state = len(states)

        state_types = torch.tensor(np.array([item[2] for item in states]), dtype=int) # [0, 1, 2] for ['skel_trans', 'attr_trans', 'execution']
        next_state_types = torch.tensor(np.array([item[2] for item in next_states]), dtype=int)
        
        advantages, returns = self.estimate_advantages(states, next_states, rewards, next_terminations, next_dones, state_types, next_state_types)

        for _ in range(self.opt_num_epochs):
                        
            if self.use_mini_batch:
                perm_np = np.arange(num_state)
                np.random.shuffle(perm_np)
                perm = LongTensor(perm_np).to(self.device)

                rnd_states, rnd_actions, rnd_returns, rnd_advantages, rnd_fixed_log_probs, rnd_exps = \
                    index_select_list(states, perm_np), index_select_list(actions, perm_np), returns[perm].clone(), advantages[perm].clone(), \
                    fixed_log_probs[perm].clone(), exps[perm].clone()

                if self.cfg.agent_specs.get('batch_design', False):
                    perm_design_np, perm_design = self.get_perm_batch_design(rnd_states)
                    rnd_states, rnd_actions, rnd_returns, rnd_advantages, rnd_fixed_log_probs, rnd_exps = \
                        index_select_list(rnd_states, perm_design_np), index_select_list(rnd_actions, perm_design_np), rnd_returns[perm_design].clone(), rnd_advantages[perm_design].clone(), \
                        rnd_fixed_log_probs[perm_design].clone(), rnd_exps[perm_design].clone()

                optim_iter_num = int(math.floor(num_state / self.mini_batch_size))
                for i in range(optim_iter_num):
                    ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, num_state))
                    states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                        rnd_states[ind], rnd_actions[ind], rnd_advantages[ind], rnd_returns[ind], rnd_fixed_log_probs[ind], rnd_exps[ind]
                    self.update_value(states_b, returns_b)
                    surr_loss = self.ppo_loss(states_b, actions_b, advantages_b, fixed_log_probs_b)
                    self.optimizer_policy.zero_grad()
                    surr_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
                    self.optimizer_policy.step()
            else:
                ind = exps.nonzero(as_tuple=False).squeeze(1)
                self.update_value(states, returns)
                surr_loss = self.ppo_loss(states, actions, advantages, fixed_log_probs)
                self.optimizer_policy.zero_grad()
                surr_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
                self.optimizer_policy.step()
        
    """
    def update_policy(self, states, next_states, rewards, next_terminations, next_dones, actions, exps, agent_ids=None):
        """Kept for base-class compatibility; routes to _update_policy_impl."""
        self._update_policy_impl(states, next_states, rewards, next_terminations,
                                 next_dones, actions, exps)

    def _update_policy_impl(self, states, next_states, rewards, next_terminations, next_dones,
                             actions, exps,
                             policy_net=None, value_net=None,
                             optimizer_policy=None, optimizer_value=None):
        """Core PPO update.  Pass per-agent nets/optimizers for IPPO; defaults to shared nets."""
        _policy_net      = policy_net      if policy_net      is not None else self.policy_net
        _value_net       = value_net       if value_net       is not None else self.value_net
        _optimizer_policy = optimizer_policy if optimizer_policy is not None else self.optimizer_policy
        _optimizer_value  = optimizer_value  if optimizer_value  is not None else self.optimizer_value

        if self._is_benchmark_mode():
            # In benchmark mode, skip topology/attribute optimization and train only control stage.
            control_idx = [i for i, item in enumerate(states) if int(np.asarray(item[2]).reshape(-1)[0]) == 2]
            if len(control_idx) == 0:
                return

            states = index_select_list(states, control_idx)
            next_states = index_select_list(next_states, control_idx)
            actions = index_select_list(actions, control_idx)
            rewards = rewards[control_idx]
            next_terminations = next_terminations[control_idx]
            next_dones = next_dones[control_idx]
            exps = exps[control_idx]

        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = []
                chunk = 10000
                for i in range(0, len(states), chunk):
                    states_i  = states[i:min(i + chunk, len(states))]
                    actions_i = actions[i:min(i + chunk, len(states))]
                    fixed_log_probs_i = _policy_net.get_log_prob(states_i, actions_i)
                    fixed_log_probs.append(fixed_log_probs_i)
                fixed_log_probs = torch.cat(fixed_log_probs)

        num_state = len(states)
        state_types      = torch.tensor(np.array([item[2] for item in states]),      dtype=int)
        next_state_types = torch.tensor(np.array([item[2] for item in next_states]), dtype=int)

        advantages, returns = self.estimate_advantages(
            states, next_states, rewards, next_terminations, next_dones,
            state_types, next_state_types, value_net=_value_net
        )

        for _ in range(self.opt_num_epochs):

            if self.use_mini_batch:
                perm_np = np.arange(num_state)
                np.random.shuffle(perm_np)
                perm = LongTensor(perm_np).to(self.device)

                rnd_states, rnd_actions, rnd_returns, rnd_advantages, rnd_fixed_log_probs, rnd_exps = \
                    index_select_list(states, perm_np), index_select_list(actions, perm_np), \
                    returns[perm].clone(), advantages[perm].clone(), \
                    fixed_log_probs[perm].clone(), exps[perm].clone()

                if self.cfg.agent_specs.get('batch_design', False):
                    perm_design_np, perm_design = self.get_perm_batch_design(rnd_states)
                    rnd_states, rnd_actions, rnd_returns, rnd_advantages, rnd_fixed_log_probs, rnd_exps = \
                        index_select_list(rnd_states, perm_design_np), \
                        index_select_list(rnd_actions, perm_design_np), \
                        rnd_returns[perm_design].clone(), rnd_advantages[perm_design].clone(), \
                        rnd_fixed_log_probs[perm_design].clone(), rnd_exps[perm_design].clone()

                optim_iter_num = int(math.floor(num_state / self.mini_batch_size))
                for i in range(optim_iter_num):
                    ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, num_state))
                    states_b         = rnd_states[ind]
                    actions_b        = rnd_actions[ind]
                    advantages_b     = rnd_advantages[ind]
                    returns_b        = rnd_returns[ind]
                    fixed_log_probs_b = rnd_fixed_log_probs[ind]

                    try:
                        values_fn = _value_net(states_b).detach()
                    except AttributeError:
                        values_fn = None

                    self.update_value(states_b, returns_b,
                                      value_net=_value_net, optimizer_value=_optimizer_value)
                    surr_loss = self.ppo_loss(states_b, actions_b, advantages_b, fixed_log_probs_b,
                                             policy_net=_policy_net)

                    _optimizer_policy.zero_grad()
                    surr_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(_policy_net.parameters(), self.cfg.max_grad_norm)
                    _optimizer_policy.step()

                    self.info = {
                        'values_fn':    values_fn.cpu() if values_fn is not None else None,
                        'grad_norm':    float(grad_norm),
                        'advantage_fn': advantages_b.detach().cpu(),
                        'loss':         float(surr_loss.detach()),
                    }

            else:
                ind = exps.nonzero(as_tuple=False).squeeze(1)  # noqa: F841 (kept for callers)

                try:
                    values_fn = _value_net(states).detach()
                except AttributeError:
                    values_fn = None

                self.update_value(states, returns,
                                  value_net=_value_net, optimizer_value=_optimizer_value)
                surr_loss = self.ppo_loss(states, actions, advantages, fixed_log_probs,
                                         policy_net=_policy_net)

                _optimizer_policy.zero_grad()
                surr_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(_policy_net.parameters(), self.cfg.max_grad_norm)
                _optimizer_policy.step()

                self.info = {
                    'values_fn':    values_fn.cpu() if values_fn is not None else None,
                    'grad_norm':    float(grad_norm),
                    'advantage_fn': advantages.detach().cpu(),
                    'loss':         float(surr_loss.detach()),
                }

    def update_value(self, states, returns, agent_ids=None,
                     value_net=None, optimizer_value=None):
        """update critic.  Pass per-agent value_net/optimizer_value for IPPO."""
        _value_net      = value_net      if value_net      is not None else self.value_net
        _optimizer_value = optimizer_value if optimizer_value is not None else self.optimizer_value
        for _ in range(self.value_opt_niter):
            values_pred = _value_net(states)
            value_loss  = (values_pred - returns).pow(2).mean()
            _optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(_value_net.parameters(), self.cfg.max_grad_norm)
            _optimizer_value.step()

    def ppo_loss(self, states, actions, advantages, fixed_log_probs, policy_net=None):
        """Clipped PPO surrogate loss.  Pass per-agent policy_net for IPPO."""
        _policy_net = policy_net if policy_net is not None else self.policy_net
        log_probs = _policy_net.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        surr_loss = - torch.min(surr1, surr2).mean()
        return surr_loss

    @staticmethod
    def _remap_state_dict_legacy_keys(state_dict):
        out = {}
        for k, v in state_dict.items():
            k = k.replace(".lin_r.", ".lin_root.").replace(".lin_l.", ".lin_rel.")
            out[k] = v
        return out

    @staticmethod
    def _checkpoint_is_multi(cp):
        return 'policy_dicts' in cp and 'value_dicts' in cp

    @staticmethod
    def _checkpoint_is_single(cp):
        return 'policy_dict' in cp and 'value_dict' in cp

    @staticmethod
    def _normalize_checkpoint_tag(checkpoint):
        if isinstance(checkpoint, int):
            return f'epoch_{checkpoint:04d}.p'
        assert isinstance(checkpoint, str)
        return checkpoint if checkpoint.endswith('.p') else f'{checkpoint}.p'

    def _resolve_checkpoint_path_from_model_dir(self, model_dir, checkpoint):
        cp_name = self._normalize_checkpoint_tag(checkpoint)
        cp_path = cp_name if os.path.isabs(cp_name) else os.path.join(model_dir, cp_name)
        if not os.path.isfile(cp_path):
            raise FileNotFoundError(f'Checkpoint not found: {cp_path}')
        return cp_path

    def _resolve_transfer_checkpoint_path(self, source_dir, checkpoint_tag):
        if source_dir is None:
            raise FileNotFoundError('transfer_init_dir is not set')

        source_dir = os.path.expanduser(str(source_dir))
        if not os.path.isabs(source_dir):
            raise FileNotFoundError(
                f"transfer_init_dir must be an absolute path, got: {source_dir}"
            )
        source_dir = os.path.abspath(source_dir)

        # If a checkpoint file path is provided, require it to exist and end with .p.
        # Do not silently reinterpret missing files as directories.
        if source_dir.endswith('.p'):
            if os.path.isfile(source_dir):
                return source_dir
            raise FileNotFoundError(
                f"transfer_init_dir points to checkpoint file but it does not exist: {source_dir}"
            )

        if os.path.isfile(source_dir):
            if source_dir.endswith('.p'):
                return source_dir
            raise FileNotFoundError(f'transfer_init_dir points to a non-checkpoint file: {source_dir}')

        cp_name = self._normalize_checkpoint_tag(checkpoint_tag)
        candidate_model_dirs = []

        if os.path.basename(source_dir) == 'models':
            candidate_model_dirs.append(source_dir)
        candidate_model_dirs.append(os.path.join(source_dir, 'models'))
        candidate_model_dirs.extend(sorted(glob.glob(os.path.join(source_dir, 'results', '*', 'models'))))
        candidate_model_dirs.extend(sorted(glob.glob(os.path.join(source_dir, '*', 'results', '*', 'models'))))

        seen = set()
        for model_dir in candidate_model_dirs:
            model_dir = os.path.abspath(model_dir)
            if model_dir in seen:
                continue
            seen.add(model_dir)

            cp_path = os.path.join(model_dir, cp_name)
            if os.path.isfile(cp_path):
                return cp_path

        raise FileNotFoundError(
            f"Could not resolve transfer checkpoint '{cp_name}' under: {source_dir}"
        )

    def _load_single_checkpoint_into_multi(self, model_cp):
        policy_sd = self._remap_state_dict_legacy_keys(model_cp['policy_dict'])
        value_sd = self._remap_state_dict_legacy_keys(model_cp['value_dict'])
        for net in self.policy_nets:
            self._load_state_dict_matching(net, policy_sd, module_name='policy')
        for net in self.value_nets:
            self._load_state_dict_matching(net, value_sd, module_name='value')

    def _load_state_dict_matching(self, module, source_state_dict, module_name='module'):
        target_state = module.state_dict()
        compatible = {}
        skipped_shape = []

        for key, value in source_state_dict.items():
            if key not in target_state:
                continue
            if target_state[key].shape == value.shape:
                compatible[key] = value
            else:
                skipped_shape.append((key, tuple(value.shape), tuple(target_state[key].shape)))

        target_state.update(compatible)
        module.load_state_dict(target_state)

        if skipped_shape:
            preview = ', '.join([f"{k}:{src}->{dst}" for k, src, dst in skipped_shape[:3]])
            self.logger.warning(
                f"partial {module_name} transfer: loaded {len(compatible)} tensors, "
                f"skipped {len(skipped_shape)} shape-mismatched tensors ({preview})"
            )
        else:
            self.logger.info(f"full {module_name} transfer: loaded {len(compatible)} tensors")

    def _maybe_transfer_init_from_single_checkpoint(self):
        if not self._is_multi_agent_case():
            return

        source_dir = getattr(self.cfg, 'transfer_init_dir', None)
        if source_dir in (None, '', 'null'):
            return

        checkpoint_tag = getattr(self.cfg, 'transfer_init_checkpoint', 'best')
        try:
            cp_path = self._resolve_transfer_checkpoint_path(source_dir, checkpoint_tag)
        except Exception as exc:
            self.logger.warning(f'transfer init skipped: {exc}')
            return

        self.logger.info(f'loading transfer-init checkpoint: {cp_path}')
        model_cp = pickle.load(open(cp_path, 'rb'))

        if self._checkpoint_is_multi(model_cp):
            self.logger.info('transfer init skipped: source checkpoint is multi-agent, expected single-agent')
            return
        if not self._checkpoint_is_single(model_cp):
            self.logger.warning('transfer init skipped: source checkpoint has unknown format')
            return

        self._load_single_checkpoint_into_multi(model_cp)

        if self.obs_norm is not None and model_cp.get('obs_norm') is not None:
            try:
                self.obs_norm.load_state_dict(model_cp['obs_norm'])
            except Exception as exc:
                self.logger.warning(f'could not load obs_norm from transfer checkpoint: {exc}')

        self.loss_iter = model_cp.get('loss_iter', self.loss_iter)
        self.best_rewards = model_cp.get('best_rewards', self.best_rewards)
        self.transfer_init_applied = True
        self.logger.info('applied single-agent checkpoint to all multi-agent policy/value nets')
                            
    def load_checkpoint(self, checkpoint):
        cfg = self.cfg
        cp_path = self._resolve_checkpoint_path_from_model_dir(cfg.model_dir, checkpoint)
        print("The Path: ", cp_path)
        self.logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))

        def _checkpoint_control_action_dim(cp):
            if self._is_multi_agent_case() and 'policy_dicts' in cp and len(cp['policy_dicts']) > 0:
                bias = cp['policy_dicts'][0].get('control_action_mean.bias', None)
                return int(bias.shape[0]) if bias is not None else None
            if 'policy_dict' in cp:
                bias = cp['policy_dict'].get('control_action_mean.bias', None)
                return int(bias.shape[0]) if bias is not None else None
            return None

        checkpoint_control_dim = _checkpoint_control_action_dim(model_cp)
        if checkpoint_control_dim is not None and checkpoint_control_dim != self.control_action_dim:
            self.logger.info(
                f"checkpoint/model control dim mismatch: checkpoint={checkpoint_control_dim}, "
                f"current={self.control_action_dim}. Rebuilding policy/value nets to match checkpoint."
            )
            self.control_action_dim = checkpoint_control_dim
            self.env.control_action_dim = checkpoint_control_dim
            self.action_dim = self.control_action_dim + self.attr_design_dim
            self.setup_policy()
            self.setup_value()
            self.setup_optimizer()

        if self._is_multi_agent_case() and self._checkpoint_is_multi(model_cp):
            # IPPO multi-agent checkpoint
            for aid, net in enumerate(self.policy_nets):
                source_idx = min(aid, len(model_cp['policy_dicts']) - 1)
                net.load_state_dict(model_cp['policy_dicts'][source_idx])
            for aid, net in enumerate(self.value_nets):
                source_idx = min(aid, len(model_cp['value_dicts']) - 1)
                net.load_state_dict(model_cp['value_dicts'][source_idx])
        elif self._is_multi_agent_case() and self._checkpoint_is_single(model_cp):
            # Initialize every multi-agent net from the single-agent source.
            self._load_single_checkpoint_into_multi(model_cp)
        else:
            # single-case or legacy checkpoint — apply key remapping then load
            self.policy_net.load_state_dict(self._remap_state_dict_legacy_keys(model_cp['policy_dict']))
            self.value_net.load_state_dict(self._remap_state_dict_legacy_keys(model_cp['value_dict']))

        self.loss_iter = model_cp['loss_iter']
        self.best_rewards = model_cp.get('best_rewards', self.best_rewards)
        if self._is_multi_agent_case():
            agent_best_rewards = model_cp.get('best_agent_rewards', None)
            if isinstance(agent_best_rewards, list):
                self.best_agent_rewards = [float(v) for v in agent_best_rewards[:self.num_agents]]
                if len(self.best_agent_rewards) < self.num_agents:
                    self.best_agent_rewards.extend([-float('inf')] * (self.num_agents - len(self.best_agent_rewards)))
            else:
                self.best_agent_rewards = [-float('inf')] * self.num_agents
            self.save_best_agent_flags = [False] * self.num_agents
        if self.obs_norm is not None and model_cp.get('obs_norm') is not None:
            self.obs_norm.load_state_dict(model_cp['obs_norm'])

    def save_checkpoint(self, epoch):
        is_multi_case = self._is_multi_agent_case()

        def _resolve_legacy_multirun_model_dir():
            """
            Reconstruct legacy Hydra-style save path with a '/0/' job folder:
            <run_root>/0/results/<cfg_id>/models
            when current cfg.model_dir points to
            <run_root>/results/<cfg_id>/models.
            """
            model_dir = os.path.normpath(self.cfg.model_dir)
            marker = f"{os.sep}results{os.sep}"
            if marker not in model_dir:
                return None

            run_root, results_suffix = model_dir.split(marker, 1)
            if os.path.basename(run_root) == '0':
                return None

            run_root_name = os.path.basename(run_root)
            run_parent_name = os.path.basename(os.path.dirname(run_root))

            is_time_stamp = re.fullmatch(r"\d{2}-\d{2}-\d{2}", run_root_name) is not None
            is_date_stamp = re.fullmatch(r"\d{4}-\d{2}-\d{2}", run_parent_name) is not None
            is_compact_stamp = re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(-\d+)?", run_root_name) is not None

            if not ((is_time_stamp and is_date_stamp) or is_compact_stamp):
                return None

            legacy_dir = os.path.join(run_root, '0', 'results', results_suffix)
            os.makedirs(legacy_dir, exist_ok=True)
            return legacy_dir

        legacy_multirun_model_dir = _resolve_legacy_multirun_model_dir()

        def sync_legacy_multirun_hydra_dir():
            if legacy_multirun_model_dir is None:
                return

            legacy_zero_dir = os.path.dirname(os.path.dirname(legacy_multirun_model_dir))
            run_root = os.path.dirname(legacy_zero_dir)
            hydra_src_dir = os.path.join(run_root, '.hydra')
            hydra_dst_dir = os.path.join(legacy_zero_dir, '.hydra')

            if not os.path.isdir(hydra_src_dir):
                return

            os.makedirs(legacy_zero_dir, exist_ok=True)
            shutil.copytree(hydra_src_dir, hydra_dst_dir, dirs_exist_ok=True)

        def save_checkpoint_to_targets(file_name, custom_save=None):
            # Preserve current behavior, then mirror to legacy multirun /0 path when applicable.
            save_fn = save if custom_save is None else custom_save
            primary_path = os.path.join(cfg.model_dir, file_name)
            save_fn(primary_path)

            if legacy_multirun_model_dir is not None:
                legacy_path = os.path.join(legacy_multirun_model_dir, file_name)
                if os.path.normpath(legacy_path) != os.path.normpath(primary_path):
                    save_fn(legacy_path)
                sync_legacy_multirun_hydra_dir()

        def save(cp_path):
            if is_multi_case:
                with to_cpu(*self.policy_nets, *self.value_nets):
                    model_cp = {
                        'obs_norm':     self.obs_norm.state_dict() if self.obs_norm is not None else None,
                        'policy_dicts': [net.state_dict() for net in self.policy_nets],
                        'value_dicts':  [net.state_dict() for net in self.value_nets],
                        'loss_iter':    self.loss_iter,
                        'best_rewards': self.best_rewards,
                        'best_agent_rewards': self.best_agent_rewards,
                        'epoch':        epoch,
                        'num_agents':   self.num_agents,
                        'env_name':     self.cfg.env_name,
                        'transfer_init_dir': getattr(self.cfg, 'transfer_init_dir', None),
                        'transfer_init_applied': bool(getattr(self, 'transfer_init_applied', False)),
                    }
                    pickle.dump(model_cp, open(cp_path, 'wb'))
            else:
                with to_cpu(self.policy_net, self.value_net):
                    model_cp = {
                        'obs_norm':     self.obs_norm.state_dict() if self.obs_norm is not None else None,
                        'policy_dict':  self.policy_net.state_dict(),
                        'value_dict':   self.value_net.state_dict(),
                        'loss_iter':    self.loss_iter,
                        'best_rewards': self.best_rewards,
                        'epoch':        epoch,
                        'num_agents':   self.num_agents,
                        'env_name':     self.cfg.env_name,
                        'transfer_init_dir': getattr(self.cfg, 'transfer_init_dir', None),
                        'transfer_init_applied': bool(getattr(self, 'transfer_init_applied', False)),
                    }
                    pickle.dump(model_cp, open(cp_path, 'wb'))

        def save_agent(cp_path, agent_id):
            with to_cpu(self.policy_nets[agent_id], self.value_nets[agent_id]):
                model_cp = {
                    'obs_norm': self.obs_norm.state_dict() if self.obs_norm is not None else None,
                    'policy_dict': self.policy_nets[agent_id].state_dict(),
                    'value_dict': self.value_nets[agent_id].state_dict(),
                    'loss_iter': self.loss_iter,
                    'best_reward': self.best_agent_rewards[agent_id],
                    'epoch': epoch,
                    'num_agents': self.num_agents,
                    'agent_id': agent_id,
                }
                pickle.dump(model_cp, open(cp_path, 'wb'))

        cfg = self.cfg
        additional_saves = self.cfg.agent_specs.get('additional_saves', None)
        if (cfg.save_model_interval > 0 and (epoch+1) % cfg.save_model_interval == 0) or \
           (additional_saves is not None and (epoch+1) % additional_saves[0] == 0 and epoch+1 <= additional_saves[1]):
            self.tb_logger.flush()
            save_checkpoint_to_targets('epoch_%04d.p' % (epoch + 1))
        if is_multi_case:
            improved_agent_ids = [aid for aid, improved in enumerate(self.save_best_agent_flags) if improved]
            for aid in improved_agent_ids:
                self.tb_logger.flush()
                self.logger.info(f'save best checkpoint for agent {aid} with rewards {self.best_agent_rewards[aid]:.2f}!')
                save_checkpoint_to_targets(f'best_agent_{aid}.p', lambda cp_path, aid=aid: save_agent(cp_path, aid))
            if improved_agent_ids:
                self.save_best_flag = True
                self.tb_logger.flush()
                self.logger.info(f'save best checkpoint with rewards {self.best_rewards:.2f}!')
                save_checkpoint_to_targets('best.p')
        elif self.save_best_flag:
            self.tb_logger.flush()
            self.logger.info(f'save best checkpoint with rewards {self.best_rewards:.2f}!')
            save_checkpoint_to_targets('best.p')
    """
    def log_optimize_policy(self, epoch, info):
        cfg = self.cfg
        log, log_eval = info['log'], info['log_eval']
        logger, tb_logger = self.logger, self.tb_logger
        log_str = f'{epoch}\tT_sample {info["T_sample"]:.2f}\tT_update {info["T_update"]:.2f}\tT_eval {info["T_eval"]:.2f}\t'\
            f'ETA {get_eta_str(epoch, cfg.max_epoch_num, info["T_total"])}\ttrain_R {log.avg_reward:.2f}\ttrain_R_eps {log.avg_episode_reward:.2f}\t'\
            f'exec_R {log_eval.avg_exec_reward:.2f}\texec_R_eps {log_eval.avg_exec_episode_reward:.2f}\t{cfg.id}'
        logger.info(log_str)

        if log_eval.avg_exec_episode_reward > self.best_rewards:
            self.best_rewards = log_eval.avg_exec_episode_reward
            self.save_best_flag = True
        else:
            self.save_best_flag = False

        tb_logger.add_scalar('train_R_avg ', log.avg_reward, epoch)
        tb_logger.add_scalar('policy_learning_rate', self.optimizer_policy.param_groups[0]["lr"], epoch)
        tb_logger.add_scalar('value_learning_rate', self.optimizer_value.param_groups[0]["lr"], epoch)
        tb_logger.add_scalar('train_R_eps_avg', log.avg_episode_reward, epoch)
        tb_logger.add_scalar('eval_R_eps_avg', log_eval.avg_episode_reward, epoch)
        tb_logger.add_scalar('exec_R_avg', log_eval.avg_exec_reward, epoch)
        tb_logger.add_scalar('exec_R_eps_avg', log_eval.avg_exec_episode_reward, epoch)
        tb_logger.add_scalar('reward_shift', self.cfg.reward_shift, epoch)
        #tb_logger.add_scalar('value_fnc', log.value_fnc, epoch)
        #tb_logger.add_scalar('loss', log.loss, epoch)
        #tb_logger.add_scalar('gradient norms', log.gradient_norms, epoch)
        #tb_logger.add_scalar('advantage_fnc', log.advantage_fnc, epoch)
        
        if self.cfg.enable_wandb:                                               # TODO in case I need different logs
            wandb.log({
                'train_R_avg': log.avg_reward,
                'policy_learning_rate': self.optimizer_policy.param_groups[0]["lr"],
                'value_learning_rate': self.optimizer_value.param_groups[0]["lr"],
                'train_R_eps_avg': log.avg_episode_reward,
                'eval_R_eps_avg': log_eval.avg_episode_reward,
                'exec_R_avg': log_eval.avg_exec_reward,
                'exec_R_eps_avg': log_eval.avg_exec_episode_reward,
                'reward_shift': self.cfg.reward_shift 
            }, step = epoch * self.cfg.min_batch_size)
    """

    def log_optimize_policy(self, epoch, info):
        cfg = self.cfg
        log, log_eval = info['log'], info['log_eval']
        logger, tb_logger = self.logger, self.tb_logger
        is_adversarial_case = self._is_adversarial_case()

        if is_adversarial_case:
            checkpoint_metric = getattr(log_eval, 'avg_adv_min_exec_episode_reward', log_eval.avg_exec_episode_reward)
            checkpoint_metric_name = 'adv_min_exec_R_eps'
            log_str = (
                f'{epoch}\tT_sample {info["T_sample"]:.2f}\tT_update {info["T_update"]:.2f}\t'
                f'T_eval {info["T_eval"]:.2f}\tETA {get_eta_str(epoch, cfg.max_epoch_num, info["T_total"])}\t'
                f'train_R {log.avg_reward:.2f}\ttrain_R_eps {log.avg_episode_reward:.2f}\t'
                f'exec_R {log_eval.avg_exec_reward:.2f}\texec_R_eps {log_eval.avg_exec_episode_reward:.2f}\t'
                f'{checkpoint_metric_name} {checkpoint_metric:.2f}\t{cfg.id}'
            )
        else:
            checkpoint_metric = log_eval.avg_exec_episode_reward
            checkpoint_metric_name = 'exec_R_eps'
            log_str = (
                f'{epoch}\tT_sample {info["T_sample"]:.2f}\tT_update {info["T_update"]:.2f}\t'
                f'T_eval {info["T_eval"]:.2f}\tETA {get_eta_str(epoch, cfg.max_epoch_num, info["T_total"])}\t'
                f'train_R {log.avg_reward:.2f}\ttrain_R_eps {log.avg_episode_reward:.2f}\t'
                f'exec_R {log_eval.avg_exec_reward:.2f}\texec_R_eps {log_eval.avg_exec_episode_reward:.2f}\t{cfg.id}'
            )

        logger.info(log_str)

        if self._is_multi_agent_case():
            prev_best_rewards = self.best_rewards
            per_agent_rewards = getattr(log_eval, 'avg_agent_exec_episode_rewards', None)
            if per_agent_rewards is None or len(per_agent_rewards) < self.num_agents:
                per_agent_rewards = [float(log_eval.avg_exec_episode_reward)] * self.num_agents
            for aid in range(self.num_agents):
                current_reward = float(per_agent_rewards[aid]) if aid < len(per_agent_rewards) else float(log_eval.avg_exec_episode_reward)
                if current_reward > self.best_agent_rewards[aid]:
                    self.best_agent_rewards[aid] = current_reward
                    self.save_best_agent_flags[aid] = True
                else:
                    self.save_best_agent_flags[aid] = False
            if checkpoint_metric > prev_best_rewards:
                self.best_rewards = checkpoint_metric
            self.save_best_flag = any(self.save_best_agent_flags) or checkpoint_metric > prev_best_rewards
        elif checkpoint_metric > self.best_rewards:
            self.best_rewards = checkpoint_metric
            self.save_best_flag = True
        else:
            self.save_best_flag = False

        tb_logger.add_scalar('train_R_avg', log.avg_reward, epoch)
        tb_logger.add_scalar('policy_learning_rate', self.optimizer_policy.param_groups[0]["lr"], epoch)
        tb_logger.add_scalar('value_learning_rate', self.optimizer_value.param_groups[0]["lr"], epoch)
        tb_logger.add_scalar('train_R_eps_avg', log.avg_episode_reward, epoch)
        tb_logger.add_scalar('eval_R_eps_avg', log_eval.avg_episode_reward, epoch)
        tb_logger.add_scalar('exec_R_avg', log_eval.avg_exec_reward, epoch)
        tb_logger.add_scalar('exec_R_eps_avg', log_eval.avg_exec_episode_reward, epoch)
        if is_adversarial_case:
            tb_logger.add_scalar('adv_min_exec_R_eps_avg', checkpoint_metric, epoch)
        tb_logger.add_scalar('reward_shift', self.cfg.reward_shift, epoch)

        # --- new values from self.info ---
        values_fn = info.get('values_fn', None)
        grad_norm = info.get('grad_norm', None)
        advantage_fn = info.get('advantage_fn', None)
        loss = info.get('loss', None)

        if values_fn is not None:
            tb_logger.add_scalar('values_fn_mean', values_fn.mean().item(), epoch)
            tb_logger.add_scalar('values_fn_std', values_fn.std().item(), epoch)
            tb_logger.add_histogram('values_fn_hist', values_fn, epoch)

        if advantage_fn is not None:
            tb_logger.add_scalar('advantage_fn_mean', advantage_fn.mean().item(), epoch)
            tb_logger.add_scalar('advantage_fn_std', advantage_fn.std().item(), epoch)
            tb_logger.add_histogram('advantage_fn_hist', advantage_fn, epoch)

        if grad_norm is not None:
            tb_logger.add_scalar('grad_norm', grad_norm, epoch)

        if loss is not None:
            tb_logger.add_scalar('loss', loss, epoch)

        for name in self.logger_cls.REWARD_COMPONENT_NAMES:
            train_step_value = getattr(log, f'avg_exec_{name}_reward', 0.0)
            train_episode_value = getattr(log, f'avg_exec_episode_{name}_reward', 0.0)
            eval_step_value = getattr(log_eval, f'avg_exec_{name}_reward', 0.0)
            eval_episode_value = getattr(log_eval, f'avg_exec_episode_{name}_reward', 0.0)

            tb_logger.add_scalar(f'train_exec_{name}_reward_avg', train_step_value, epoch)
            tb_logger.add_scalar(f'train_exec_{name}_reward_eps_avg', train_episode_value, epoch)
            tb_logger.add_scalar(f'eval_exec_{name}_reward_avg', eval_step_value, epoch)
            tb_logger.add_scalar(f'eval_exec_{name}_reward_eps_avg', eval_episode_value, epoch)

        if self.cfg.enable_wandb:
            wandb_log = {
                'train_R_avg': log.avg_reward,
                'policy_learning_rate': self.optimizer_policy.param_groups[0]["lr"],
                'value_learning_rate': self.optimizer_value.param_groups[0]["lr"],
                'train_R_eps_avg': log.avg_episode_reward,
                'eval_R_eps_avg': log_eval.avg_episode_reward,
                'exec_R_avg': log_eval.avg_exec_reward,
                'exec_R_eps_avg': log_eval.avg_exec_episode_reward,
                'reward_shift': self.cfg.reward_shift,
            }
            if is_adversarial_case:
                wandb_log['adv_min_exec_R_eps_avg'] = checkpoint_metric

            for name in self.logger_cls.REWARD_COMPONENT_NAMES:
                train_step_value = getattr(log, f'avg_exec_{name}_reward', 0.0)
                train_episode_value = getattr(log, f'avg_exec_episode_{name}_reward', 0.0)
                eval_step_value = getattr(log_eval, f'avg_exec_{name}_reward', 0.0)
                eval_episode_value = getattr(log_eval, f'avg_exec_episode_{name}_reward', 0.0)
                wandb_log[f'train_exec_{name}_reward_avg'] = train_step_value
                wandb_log[f'train_exec_{name}_reward_eps_avg'] = train_episode_value
                wandb_log[f'eval_exec_{name}_reward_avg'] = eval_step_value
                wandb_log[f'eval_exec_{name}_reward_eps_avg'] = eval_episode_value

            if values_fn is not None:
                wandb_log['values_fn_mean'] = values_fn.mean().item()
                wandb_log['values_fn_std'] = values_fn.std().item()
                wandb_log['values_fn_hist'] = wandb.Histogram(values_fn.detach().cpu().numpy())

            if advantage_fn is not None:
                wandb_log['advantage_fn_mean'] = advantage_fn.mean().item()
                wandb_log['advantage_fn_std'] = advantage_fn.std().item()
                wandb_log['advantage_fn_hist'] = wandb.Histogram(advantage_fn.detach().cpu().numpy())

            if grad_norm is not None:
                wandb_log['grad_norm'] = grad_norm

            if loss is not None:
                wandb_log['loss'] = loss

            wandb.log(wandb_log, step=epoch * self.cfg.min_batch_size)



            
    def visualize_agent(self, num_episode=1, mean_action=True, save_video=False, pause_design=False, max_num_frames=1000):
        fr = 0
        env = self.env
        paused = not save_video and pause_design
        
        if self.cfg.uni_obs_norm:
            self.obs_norm.eval()
            self.obs_norm.to('cpu')
        
        for _ in range(num_episode):
            state = env.reset()

            env._get_viewer('human')._paused = paused
            env.render()
            for t in range(1000):
                state_var = tensorfy([state])
                
                ## do obs norm (none-updated)
                if self.cfg.uni_obs_norm:
                    state_var = self.normalize_observation(state_var)
                        
                with torch.no_grad():
                    action = self.policy_net.select_action(state_var, mean_action).numpy().astype(np.float64)
                next_state, env_reward, termination, truncation, info = env.step(action)

                
                done = (termination or truncation)
                
                if t < self.cfg.skel_transform_nsteps + 1:
                    env._get_viewer('human')._paused = paused
                    env._get_viewer('human')._hide_overlay = save_video
                for _ in range(15 if save_video else 1):
                    env.render()
                if save_video:
                    frame_dir = f'out/videos/{self.cfg.id}_frames'
                    os.makedirs(frame_dir, exist_ok=True)
                    save_screen_shots(env.viewer.window, f'{frame_dir}/%04d.png' % fr)
                    fr += 1
                    if fr >= max_num_frames:
                        break

                if done:
                    print(f'truncation: {truncation}, termination: {termination}')
                    break
                state = next_state

            if save_video and fr >= max_num_frames:
                break

        if save_video:
            save_video_ffmpeg(f'{frame_dir}/%04d.png', f'out/videos/{self.cfg.id}.mp4', fps=30)
            shutil.rmtree(frame_dir)

    def eval_data(self, out_csv_path, mean_action=True, duration_sec=30, out_plot_dir=None):

        env = self.env

        warmup_steps = 6
        state = env.reset()

        if self.cfg.uni_obs_norm:
            self.obs_norm.eval()
            self.obs_norm.to('cpu')

        for _ in range(warmup_steps):
            state_var = tensorfy([state])
            if self.cfg.uni_obs_norm:
                state_var = self.normalize_observation(state_var)

            with torch.no_grad():
                action = self.policy_net.select_action(
                    state_var, mean_action=True
                ).numpy().astype(np.float64)

            next_state, _, termination, truncation, _ = env.step(action)
            done = termination or truncation
            state = env.reset() if done else next_state


        model = env.model

        joint_info = []
        for i in range(model.njnt):
            name = model.joint(i).name
            jtype = int(model.jnt_type[i])
            qpos_start = int(model.jnt_qposadr[i])
            qvel_start = int(model.jnt_dofadr[i])

            if i < model.njnt - 1:
                qpos_dim = int(model.jnt_qposadr[i+1]) - qpos_start
            else:
                qpos_dim = int(model.nq) - qpos_start

            if i < model.njnt - 1:
                qvel_dim = int(model.jnt_dofadr[i+1]) - qvel_start
            else:
                qvel_dim = int(model.nv) - qvel_start

            joint_info.append({
                "joint_index": i,
                "joint_name": name,
                "joint_type": jtype,
                "qpos_start": qpos_start,
                "qpos_dim": qpos_dim,
                "qvel_start": qvel_start,
                "qvel_dim": qvel_dim
            })

        joint_info_path = out_csv_path.replace(".csv", "_joint_info.json")
        with open(joint_info_path, "w") as jf:
            json.dump(joint_info, jf, indent=2)

        # timestep und Schrittezahlen jetzt aus *aktuellem* Model
        timestep = model.opt.timestep
        max_steps = int(duration_sec / timestep)

        eval_rows = []

        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step","reward","done","sim_time","obs","action","qpos","qvel"])

            sim_time = 0.0
            for step in range(max_steps):
                state_var = tensorfy([state])
                if self.cfg.uni_obs_norm:
                    state_var = self.normalize_observation(state_var)

                with torch.no_grad():
                    action = self.policy_net.select_action(
                        state_var, mean_action
                    ).numpy().astype(np.float64)

                next_state, reward, termination, truncation, info = env.step(action)
                done = termination or truncation
                sim_time += timestep

                qpos = env.data.qpos.copy()
                qvel = env.data.qvel.copy()

                writer.writerow([
                    step,
                    float(reward),
                    bool(done),
                    sim_time,
                    list(state),
                    action.tolist(),
                    qpos.tolist(),
                    qvel.tolist()
                ])
                f.flush()

                eval_rows.append({
                    'step': int(step),
                    'reward': float(reward),
                    'done': bool(done),
                    'sim_time': float(sim_time),
                    'obs': list(state),
                    'action': action.tolist(),
                    'qpos': qpos.tolist(),
                    'qvel': qvel.tolist(),
                })

                if done:
                    break
                state = next_state

        if out_plot_dir is not None:
            self._plot_eval_rows(eval_rows, out_plot_dir)



    def visualize_agent_video(self, video_dir, num_episode=1, mean_action=True, max_num_frames=500):       
                
        width = 1600
        height = 900
        width = 640
        height = 480
        fr = 0
        env = self.env
        if hasattr(self.env, 'np_random'):
            print(f'Random State: {self.env.np_random}')
        timestep = env.model.opt.timestep
        fps=30
        #skip_frames = int(0.01 / timestep)
        skip_frames = 1

        env_name = str(getattr(self.cfg, 'env_name', '')).lower()
        is_multi_case = self._is_multi_agent_case()
        benchmark_mode = self._is_benchmark_mode()
        print("Benchmark: ", benchmark_mode)
        viz_agent_id = 0
        multi_has_track_camera = True
        multi_track_cam_id = None
        multi_track_cam_base_pos = None
        multi_track_cam_base_quat = None
        
        if self.cfg.uni_obs_norm:
            self.obs_norm.eval()
            self.obs_norm.to('cpu')

        frame_dir = f'{video_dir}/frames'
        os.makedirs(frame_dir, exist_ok=True)
        plot_dir = f'{video_dir}/plots'
        os.makedirs(plot_dir, exist_ok=True)
        reward_series = {}
        episode_reward_series = {}

        if is_multi_case:
            try:
                multi_track_cam_id = env.model.camera('track').id
                multi_has_track_camera = True
                multi_track_cam_base_pos = env.model.cam_pos[multi_track_cam_id].copy()
                multi_track_cam_base_quat = env.model.cam_quat[multi_track_cam_id].copy()
            except KeyError:
                multi_has_track_camera = False

        plot_step_idx = 0
        for ep in range(num_episode):
            episode_reward = 0.0
            if is_multi_case:
                env.reset()
                states = [env.get_agent_obs(agent_id) for agent_id in range(self.num_agents)]
                per_agent_episode_rewards = [0.0] * self.num_agents
            else:
                state = env.reset()

            if is_multi_case:
                if multi_has_track_camera and multi_track_cam_id is not None and multi_track_cam_base_pos is not None:
                    env.model.cam_mode[multi_track_cam_id] = 0  # fixed camera, never follows any gripper
                    # Keep camera pose exactly as authored in XML.
                    env.model.cam_pos[multi_track_cam_id] = multi_track_cam_base_pos
                    if multi_track_cam_base_quat is not None:
                        env.model.cam_quat[multi_track_cam_id] = multi_track_cam_base_quat
            else:
                cam_id = env.model.camera('track').id
                env.model.cam_mode[cam_id] = 0  # Set to fixed mode to allow manual positioning
                env.model.cam_pos[cam_id] = [0, 10, 5]
                env.model.cam_quat[cam_id] = [0.556, 0.831, 0, 0]  # Flipped right side up

            for t in range(max_num_frames*skip_frames):
                if is_multi_case:
                    next_states = list(states)
                    env_reward = 0.0
                    termination = False
                    truncation = False
                    info = {}

                    agent_actions = []
                    for agent_id in range(self.num_agents):
                        current_state = states[agent_id]
                        stage_id = self._state_stage_id(current_state)
                        use_random_transform = benchmark_mode and stage_id in (0, 1)

                        if use_random_transform:
                            action = self._sample_benchmark_transform_action(current_state)
                        else:
                            state_var = tensorfy([current_state])

                            if self.cfg.uni_obs_norm:
                                state_var = self.normalize_observation(state_var)

                            with torch.no_grad():
                                action = self.policy_nets[agent_id].select_action(state_var, mean_action).numpy().astype(np.float64)
                        agent_actions.append(action)

                    try:
                        joint_action = np.stack(agent_actions, axis=0)
                    except ValueError:
                        joint_action = list(agent_actions)
                    next_states_batch, env_reward, termination, truncation, info = env.multi_step(
                        joint_action,
                        agent_id=None
                    )

                    if isinstance(next_states_batch, list):
                        for i in range(min(len(next_states_batch), self.num_agents)):
                            if next_states_batch[i] is not None:
                                next_states[i] = next_states_batch[i]

                    state = next_states[viz_agent_id]
                else:
                    stage_id = self._state_stage_id(state)
                    use_random_transform = benchmark_mode and stage_id in (0, 1)

                    if use_random_transform:
                        action = self._sample_benchmark_transform_action(state)
                    else:
                        state_var = tensorfy([state])

                        if self.cfg.uni_obs_norm:
                            state_var = self.normalize_observation(state_var)

                        with torch.no_grad():
                            action = self.policy_net.select_action(state_var, mean_action).numpy().astype(np.float64)
                    print("Step:", t, "Action:", action.shape)
                    next_state, env_reward, termination, truncation, info = env.step(action)

                done = (termination or truncation)
                self._append_step_values(
                    reward_series,
                    self._collect_reward_values_from_info(info),
                    step_idx=plot_step_idx
                )
                self._append_step_values(episode_reward_series, {
                    'env_reward': env_reward,
                    'episode_reward': episode_reward + env_reward,
                    'done': float(done),
                    'termination': float(termination),
                    'truncation': float(truncation),
                }, step_idx=plot_step_idx)
                plot_step_idx += 1
                #The debugging
                model = env.model
                data = env.data

                # Stacking env
                if mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        "box_1"
                    ) != -1:

                    box1_body_id = model.body("box_1").id
                    box2_body_id = model.body("box_2").id

                    print("\n--- MUJOCO DEBUG ---")
                    print("box1 xpos:", data.xpos[box1_body_id])
                    print("box2 xpos:", data.xpos[box2_body_id])
                    if is_multi_case:
                        print("gripper1 xpos:", data.xpos[model.body("0_1").id])
                        print("gripper2 xpos:", data.xpos[model.body("0_2").id])
                    else: 
                        print("gripper xpos:", data.xpos[model.body("0").id])
                    print("contacts:", data.ncon)

                # Original env
                else:

                    box_body_id = model.body("box").id

                    print("\n--- MUJOCO DEBUG ---")
                    print("box xpos:", data.xpos[box_body_id])
                    if is_multi_case:
                        print("gripper1 xpos:", data.xpos[model.body("0_1").id])
                        print("gripper2 xpos:", data.xpos[model.body("0_2").id])
                    else: 
                        print("gripper xpos:", data.xpos[model.body("0").id])
                        print("gripper orientation: ", data.xquat[model.body("0").id])
                        print("box orientation: ", data.xquat[box_body_id])
                    print("contacts:", data.ncon)
                #box_geom_id = model.geom("box").id
                """
                print("\n--- MUJOCO DEBUG ---")
                #print("box qpos:", data.qpos[env.box_qpos_addr:env.box_qpos_addr+7])

                print("box xpos (BODY):", data.xpos[box_body_id])
                #print("box geom xpos:", data.geom_xpos[box_geom_id])

                print("gripper xpos:", data.xpos[model.body("0").id])

                print("contacts:", data.ncon)
                """
                print("Timestep: ", t)
                episode_reward += env_reward
                if is_multi_case:
                    agent_rewards = info.get('agent_rewards', None) if isinstance(info, dict) else None
                    if isinstance(agent_rewards, (list, tuple, np.ndarray)) and len(agent_rewards) == self.num_agents:
                        for idx, value in enumerate(agent_rewards):
                            per_agent_episode_rewards[idx] += float(value)
                    else:
                        shared_reward = float(env_reward) / float(max(1, self.num_agents))
                        for idx in range(self.num_agents):
                            per_agent_episode_rewards[idx] += shared_reward
                    summed_agent_reward = float(np.sum(per_agent_episode_rewards))
                    print(
                        f'Episode {ep}\tTotal reward {episode_reward:.2f}\t'
                        f'Agent rewards {per_agent_episode_rewards}\t'
                        f'Sum agents {summed_agent_reward:.2f}\t'
                        f'Env total {env_reward:.2f}'
                    )
                else:
                    print(f'Episode {ep}\texec_R_eps {episode_reward:.2f}')
                print(f'truncation: {truncation}, termination: {termination}')
                if t % skip_frames == 0 and t > 0:
                    # print(f'Time: {(t/30):3.2f}s | Reward: {env_reward} | Goal Rotation: {env.box_rot} | Roation Distance: {env.}')
                    #frame = env.render(mode='rgb_array', width=width, height=height)
                    if is_multi_case and not multi_has_track_camera:
                        frame = env.render(mode='rgb_array', width=width, height=height)
                    else:
                        frame = env.render(mode='rgb_array', width=width, height=height, camera_name='track')
                    
                    imageio.imwrite(f'{frame_dir}/{fr:04d}.png', frame)
                    fr += 1
                
                    
                if fr >= max_num_frames:
                    
                    break
                
                if done:
                    break
                

                if is_multi_case:
                    states = next_states
                else:
                    state = next_state

            if fr >= max_num_frames:
                break


        output_file = f'{video_dir}/video.mp4'
        save_video_ffmpeg(f'{frame_dir}/%04d.png', output_file, fps=fps)
        shutil.rmtree(frame_dir)

        if reward_series or episode_reward_series:
            max_len = max(
                max((len(v) for v in reward_series.values()), default=0),
                max((len(v) for v in episode_reward_series.values()), default=0)
            )
            steps = np.arange(max_len)
            self._plot_time_series(
                x_values=steps,
                series_dict=reward_series,
                out_path=os.path.join(plot_dir, 'info_reward_components.png'),
                title='Reward Signals From info'
            )
            self._plot_time_series(
                x_values=steps,
                series_dict=episode_reward_series,
                out_path=os.path.join(plot_dir, 'episode_rewards.png'),
                title='Episode Reward Signals'
            )

        eval_dir = f'{video_dir}/eval_data'
        os.makedirs(eval_dir, exist_ok=True)
        eval_csv_path = os.path.join(eval_dir, 'eval_data.csv')
        self.eval_data(
            out_csv_path=eval_csv_path,
            mean_action=mean_action,
            duration_sec=max(1.0, float(max_num_frames) / float(fps)),
            out_plot_dir=os.path.join(eval_dir, 'plots')
        )
        
    def visualize_agent_frames(self, out_dir, num_episode=1, mean_action=True, max_frames=500):
        width = 1600
        height = 900
        fr = 0
        env = self.env

        if self.cfg.uni_obs_norm:
            self.obs_norm.eval()
            self.obs_norm.to('cpu')

        frame_dir = f'{out_dir}/shooting_frames'
        os.makedirs(frame_dir, exist_ok=True)

        for _ in range(num_episode):
            state = env.reset()

            while fr < max_frames:
                state_var = tensorfy([state])

                if self.cfg.uni_obs_norm:
                    state_var = self.normalize_observation(state_var)

                with torch.no_grad():
                    action = self.policy_net.select_action(state_var, mean_action).numpy().astype(np.float64)

                next_state, env_reward, termination, truncation, info = env.step(action)
                done = termination or truncation

                frame = env.render(mode='rgb_array', width=width, height=height)
                imageio.imwrite(f'{frame_dir}/{fr:04d}.png', frame)
                fr += 1

                if done:
                    break

                state = next_state

            if fr >= max_frames:
                break
        
    def debug(self):
        mean_action = True
        env = self.env
        
        if self.cfg.uni_obs_norm:
            self.obs_norm.eval()
            self.obs_norm.to('cpu')

        state = env.reset()

        for t in range(300):
            state_var = tensorfy([state])

            if self.cfg.uni_obs_norm:
                state_var = self.normalize_observation(state_var)
            with torch.no_grad():
                action = self.policy_net.select_action(state_var, mean_action).numpy().astype(np.float64)

            next_state, env_reward, termination, truncation, info = env.step(action)
            print(f'Reward: {env_reward}')
            done = (termination or truncation)
            if done:
                print(f'truncation: {truncation}, termination: {termination}')
                break

            state = next_state
            # print(f'State: {state}')