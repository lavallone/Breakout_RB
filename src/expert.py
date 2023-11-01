import os
import shutil
from pathlib import Path

import gym
import numpy as np
import wandb
from flloat.parser.ldlf import LDLfParser
from gym.spaces import MultiDiscrete
from gym.wrappers import Monitor
from gym_breakout_pygame.breakout_env import BreakoutConfiguration
from temprl.wrapper import TemporalGoalWrapper, TemporalGoal

from src.common import BreakoutWrapper, extract_breakout_fluents, make_goal
from rl_algorithm.brains import Sarsa, QLearning
from rl_algorithm.callbacks import ModelCheckpoint
from rl_algorithm.core import TrainEpisodeLogger, Agent
from rl_algorithm.policies import EpsGreedyQPolicy, AutomataPolicy
from rl_algorithm.temporal import TemporalGoalWrapperLogTraces


class BreakoutExpertWrapper(gym.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(BreakoutWrapper(*args, **kwargs))

        if self.action_type == "fire":
            self.observation_space = MultiDiscrete((
                self.env.observation_space.spaces["paddle_x"].n,
            ))
        else:
            self.observation_space = MultiDiscrete((
                self.env.observation_space.spaces["paddle_x"].n,
                self.env.observation_space.spaces["ball_x"].n,
                self.env.observation_space.spaces["ball_y"].n,
                self.env.observation_space.spaces["ball_x_speed"].n,
                self.env.observation_space.spaces["ball_y_speed"].n
            ))

def make_env(config: BreakoutConfiguration, output_dir, goal_reward: float = 1000.0,
             action_type: str = "fire_ball", direction: str = "sx2dx",
             reward_shaping: bool = True) -> gym.Env:
    """
    Make the Breakout environment.

    :param config: the Breakout configuration.
    :param output_dir: the path to the output directory.
    :param reward_shaping: apply automata-based reward shaping.
    :return: the Gym environment.
    """
    unwrapped_env = BreakoutExpertWrapper(config, action_type)

    print("Instantiating the DFA...")
    formula_string = make_goal(config.brick_cols, direction) # is a simple string representing LDL_f formula
    formula = LDLfParser()(formula_string) # flloat
    labels = {"c{}".format(i) for i in range(config.brick_cols)}
    # abstract class to represent a temporal goal
    tg = TemporalGoal(formula=formula,
                      reward=goal_reward, # reward when the formula is satisfied
                      labels=labels,
                      reward_shaping=reward_shaping,
                      zero_terminal_state=False,
                      extract_fluents=extract_breakout_fluents)

    print("Formula: {}".format(formula_string))
    print("Done!")
    # we save the corrispondent automaton
    tg._automaton.to_dot(os.path.join(output_dir, "true_automaton"))
    print("Original automaton at {}".format(os.path.join(output_dir, "true_automaton.svg")))

    # gym wrapper to include the temporal goal in the environment
    if action_type == "fire":
        env = TemporalGoalWrapper(
            unwrapped_env,
            [tg],
            combine=lambda obs, qs: tuple((*obs, *qs)),
            feature_extractor=(lambda obs, action: (
                obs["paddle_x"],
            ))
        )
    else:
        env = TemporalGoalWrapper(
            unwrapped_env,
            [tg],
            combine=lambda obs, qs: tuple((*obs, *qs)),
            feature_extractor=(lambda obs, action: (
                obs["paddle_x"], 
                obs["ball_x"], obs["ball_y"], obs["ball_x_speed"], obs["ball_y_speed"],
            ))
        )

    # these two files are filled during the learning process
    positive_traces_path = Path(output_dir, "positive_traces.txt")
    negative_traces_path = Path(output_dir, "negative_traces.txt")
    # class only needed for the 'Imitation purposes'
    env = TemporalGoalWrapperLogTraces(env, extract_breakout_fluents, positive_traces_path, negative_traces_path)

    return env

def run_expert(arguments, configuration):
    wandb.login()
    if arguments.imitation == "imitation":
        output_dir = "experiments/Imitation/"+arguments.algorithm+"_"+arguments.action_type+"_"+str(arguments.rows)+"x"+str(arguments.cols)+"_"+str(arguments.train_steps)+"_"+arguments.direction
        agent_dir = Path(output_dir) / "expert"
        action_type = "fire" # we set the action for the 'expert'
        run_name = "EXPERT_"+arguments.algorithm+"_"+arguments.action_type+"_"+str(arguments.rows)+"x"+str(arguments.cols)+"_"+str(arguments.train_steps)+"_"+arguments.direction
    else:
        output_dir = "experiments/RB/"+arguments.algorithm+"_"+arguments.action_type+"_"+str(arguments.rows)+"x"+str(arguments.cols)+"_"+str(arguments.train_steps)+"_"+arguments.direction
        agent_dir = Path(output_dir) / "agent"
        action_type = arguments.action_type
        run_name = "RB_"+arguments.algorithm+"_"+arguments.action_type+"_"+str(arguments.rows)+"x"+str(arguments.cols)+"_"+str(arguments.train_steps)+"_"+arguments.direction
    shutil.rmtree(output_dir, ignore_errors=True)
    agent_dir.mkdir(parents=True, exist_ok=False)

    ball_enabled, fire_enabled = False, False
    if action_type == "fire": # only when the agent is an 'expert' in an imitation learning scenario
        fire_enabled = True
    elif action_type == "fire_ball":
        fire_enabled = True
        ball_enabled = True
    elif action_type == "ball":
        ball_enabled = True
    config = BreakoutConfiguration(brick_rows=arguments.rows, brick_cols=arguments.cols,
                                   brick_reward=arguments.brick_reward, step_reward=arguments.step_reward,
                                   ball_enabled=ball_enabled, fire_enabled=fire_enabled)
    env = make_env(config, output_dir, arguments.goal_reward, action_type, arguments.direction)

    np.random.seed(arguments.seed)
    env.seed(arguments.seed)

    policy = AutomataPolicy((-2, ), nb_steps=arguments.train_steps/10, value_max=1.0, value_min=configuration.min_eps)
    algorithm = Sarsa if arguments.algorithm == "sarsa" else QLearning
    agent = Agent(algorithm(None,
                        env.action_space,
                        gamma=configuration.gamma,
                        alpha=configuration.alpha,
                        lambda_=configuration.lambda_),
                  policy=policy,
                  test_policy=EpsGreedyQPolicy(eps=0.01))
    # during Agent 'fit' the agent is set to the choosen policy

    with wandb.init(entity="lavallone", project="Restraining Bolts", name=run_name, mode="online"):
        # here it starts the learning
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
    
    agent.save(Path(agent_dir, "checkpoints", "agent.pkl"))
    agent = Agent.load(agent_dir / "checkpoints" / "agent.pkl")
    agent.test(Monitor(env, agent_dir / "videos"), nb_episodes=5, visualize=True)
    
    if arguments.imitation == "no_imitation":
        os.remove(output_dir+"/negative_traces.txt")
        os.remove(output_dir+"/positive_traces.txt")

    env.close()
    return output_dir