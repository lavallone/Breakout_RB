import shutil
from pathlib import Path

import gym
import numpy as np
import wandb
import pythomata
from gym.spaces import MultiDiscrete
from gym.wrappers import Monitor
from gym_breakout_pygame.breakout_env import BreakoutConfiguration
from pythomata.dfa import DFA
from temprl.wrapper import TemporalGoalWrapper, TemporalGoal

from src.common import BreakoutWrapper, extract_breakout_fluents
from rl_algorithm.brains import Sarsa, QLearning
from rl_algorithm.callbacks import ModelCheckpoint
from rl_algorithm.core import Agent, TrainEpisodeLogger
from rl_algorithm.policies import EpsGreedyQPolicy, AutomataPolicy


class BreakoutLearnerWrapper(gym.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(BreakoutWrapper(*args, **kwargs))

        self.observation_space = MultiDiscrete((
            self.env.observation_space.spaces["paddle_x"].n,
            self.env.observation_space.spaces["ball_x"].n,
            self.env.observation_space.spaces["ball_y"].n,
            self.env.observation_space.spaces["ball_x_speed"].n,
            self.env.observation_space.spaces["ball_y_speed"].n
        ))

def make_env_from_dfa(config: BreakoutConfiguration, dfa: DFA,
                      goal_reward: float = 1000.0, action_type: str = "fire_ball",
                      reward_shaping: bool = True) -> gym.Env:
    """
    Make the Breakout environment.

    :param config: the Breakout configuration.
    :param dfa: the automaton that constitutes the goal.
    :param goal_reward: the reward associated to the goal.
    :param reward_shaping: apply automata-based reward shaping.
    :return: the Gym environment.
    """
    unwrapped_env = BreakoutLearnerWrapper(config, action_type)

    tg = TemporalGoal(automaton=dfa,
                      reward=goal_reward,
                      reward_shaping=reward_shaping,
                      zero_terminal_state=False,
                      extract_fluents=extract_breakout_fluents)

    env = TemporalGoalWrapper(
        unwrapped_env,
        [tg],
        combine=lambda obs, qs: tuple((*obs, *qs)),
        feature_extractor=(lambda obs, action: (
            obs["paddle_x"],
            obs["ball_x"],
            obs["ball_y"],
            obs["ball_x_speed"],
            obs["ball_y_speed"],
        ))
    )

    return env

def run_learner(arguments, configuration, dfa: pythomata.dfa.DFA):
    wandb.login()
    output_dir = "experiments/Imitation/"+arguments.algorithm+"_"+arguments.action_type+"_"+str(arguments.rows)+"x"+str(arguments.cols)+"_"+str(arguments.train_steps)+"_"+arguments.direction
    agent_dir = Path(output_dir) / "learner"
    run_name = "LEARNER_"+arguments.algorithm+"_"+arguments.action_type+"_"+str(arguments.rows)+"x"+str(arguments.cols)+"_"+str(arguments.train_steps)+"_"+arguments.direction
    shutil.rmtree(agent_dir, ignore_errors=True)
    agent_dir.mkdir(parents=True, exist_ok=False)

    fire_enabled = False
    if arguments.action_type == "fire_ball":
        fire_enabled = True
    config = BreakoutConfiguration(brick_rows=arguments.rows, brick_cols=arguments.cols,
                                   brick_reward=arguments.brick_reward, step_reward=arguments.step_reward,
                                   fire_enabled=fire_enabled, ball_enabled=True)
    env = make_env_from_dfa(config, dfa, arguments.goal_reward, arguments.action_type)

    np.random.seed(arguments.seed)
    env.seed(arguments.seed)

    policy = AutomataPolicy((-1, ), nb_steps=arguments.train_steps/10, value_max=0.8, value_min=configuration.min_eps)

    algorithm = Sarsa if arguments.algorithm == "sarsa" else QLearning
    agent = Agent(algorithm(None,
                            env.action_space,
                            gamma=configuration.gamma,
                            alpha=configuration.alpha,
                            lambda_=configuration.lambda_),
                  policy=policy,
                  test_policy=EpsGreedyQPolicy(eps=0.001))

    with wandb.init(entity="lavallone", project="Restraining Bolts", name=run_name, mode="online"):
        _ = agent.fit(
            env,
            nb_steps=arguments.train_steps,
            visualize=configuration.visualize_training,
            callbacks=[
                ModelCheckpoint(str(agent_dir / "checkpoints" / "agent-{}.pkl")),
                TrainEpisodeLogger()
            ]
        )
    wandb.finish()

    agent.save(agent_dir / "checkpoints" / "agent.pkl")
    agent = Agent.load(agent_dir / "checkpoints" / "agent.pkl")
    agent.test(Monitor(env, agent_dir / "videos"), nb_episodes=5, visualize=True)

    env.close()