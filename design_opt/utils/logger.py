import math
import numpy as np
from khrylib.utils.stats_logger import StatsLogger
from khrylib.rl.core.logger_rl import LoggerRL


class LoggerRLV1(LoggerRL):

    REWARD_COMPONENT_NAMES = [
        'distance',
        'grasp',
        'compactness',
        'lift',
        'orientation',
        'binary',
        'idle',
        'goal',
        'goal_line_distance',
        'opponent_goal_line',
    ]

    def __init__(self, init_stats_logger=True, use_c_reward=False):
        super().__init__(init_stats_logger, use_c_reward)
        self.stats_names += ['exec_reward', 'exec_episode_reward']
        # Adversarial-only aggregate: per-episode minimum over agent returns.
        # This remains zero for non-adversarial environments.
        self.stats_names += ['adv_min_exec_episode_reward', 'agent_exec_episode_reward']
        self.reward_component_stat_names = []
        for name in self.REWARD_COMPONENT_NAMES:
            self.reward_component_stat_names.extend([
                f'exec_{name}_reward',
                f'exec_episode_{name}_reward',
            ])
        self.stats_names += self.reward_component_stat_names
        if init_stats_logger:
            self.stats_loggers['exec_reward'] = StatsLogger()
            self.stats_loggers['exec_episode_reward'] = StatsLogger()
            self.stats_loggers['adv_min_exec_episode_reward'] = StatsLogger()
            self.stats_loggers['agent_exec_episode_reward'] = StatsLogger(is_nparray=True)
            for stat_name in self.reward_component_stat_names:
                self.stats_loggers[stat_name] = StatsLogger()

    def start_episode(self, env):
        super().start_episode(env)
        self.exec_episode_reward = 0
        self.exec_episode_reward_components = {
            name: 0.0 for name in self.REWARD_COMPONENT_NAMES
        }
        self._adv_agent_episode_returns = None

    def step(self, env, reward, c_reward, c_info, info):
        super().step(env, reward, c_reward, c_info, info)
        if info.get('stage') == 'execution':
            self.stats_loggers['exec_reward'].log(reward)
            self.exec_episode_reward += reward
            reward_components = info.get('reward_components', {})
            for name in self.REWARD_COMPONENT_NAMES:
                value = float(reward_components.get(name, 0.0))
                self.stats_loggers[f'exec_{name}_reward'].log(value)
                self.exec_episode_reward_components[name] += value

            # Optional adversarial accounting: aggregate each agent's own
            # reward stream and evaluate the episode by its weakest agent.
            agent_rewards = info.get('agent_rewards', None)
            if agent_rewards is not None:
                agent_rewards = [float(x) for x in agent_rewards]
                if self._adv_agent_episode_returns is None:
                    self._adv_agent_episode_returns = [0.0 for _ in agent_rewards]
                count = min(len(self._adv_agent_episode_returns), len(agent_rewards))
                for i in range(count):
                    self._adv_agent_episode_returns[i] += agent_rewards[i]

    def end_episode(self, env):
        super().end_episode(env)
        self.stats_loggers['exec_episode_reward'].log(self.exec_episode_reward)
        for name in self.REWARD_COMPONENT_NAMES:
            self.stats_loggers[f'exec_episode_{name}_reward'].log(
                self.exec_episode_reward_components[name]
            )
        if self._adv_agent_episode_returns is not None and len(self._adv_agent_episode_returns) > 0:
            agent_returns = np.asarray(self._adv_agent_episode_returns, dtype=np.float64)
            self.stats_loggers['adv_min_exec_episode_reward'].log(float(np.min(agent_returns)))
            self.stats_loggers['agent_exec_episode_reward'].log(agent_returns)

    def end_sampling(self):
        super().end_sampling()

    @classmethod
    def merge(cls, logger_list, **kwargs):
        logger = super().merge(logger_list, **kwargs)
        logger.avg_exec_reward = logger.stats_loggers['exec_reward'].avg()
        logger.avg_exec_episode_reward = logger.stats_loggers['exec_episode_reward'].avg()
        logger.avg_adv_min_exec_episode_reward = logger.stats_loggers['adv_min_exec_episode_reward'].avg()
        logger.avg_agent_exec_episode_rewards = logger.stats_loggers['agent_exec_episode_reward'].avg()
        for name in cls.REWARD_COMPONENT_NAMES:
            setattr(
                logger,
                f'avg_exec_{name}_reward',
                logger.stats_loggers[f'exec_{name}_reward'].avg(),
            )
            setattr(
                logger,
                f'avg_exec_episode_{name}_reward',
                logger.stats_loggers[f'exec_episode_{name}_reward'].avg(),
            )
        return logger
