#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This is the main entry-point for the experiments with the Breakout environment."""
import logging
import os
from argparse import ArgumentParser

import yaml

from src.learner import run_learner
from src.expert import run_expert
from rl_algorithm.utils import Map, learn_dfa

logging.getLogger("temprl").setLevel(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(level=logging.INFO)
logging.getLogger("rl_algorithm").setLevel(level=logging.INFO)


def parse_args():
    parser = ArgumentParser()
    # experiments parameters
    parser.add_argument("--imitation", type=str, default="no_imitation", help="Imitation Learning or not? ('imitation' or 'no_imitation').", required=True)
    parser.add_argument("--cols", type=int, default=3, help="Number of columns.", required=True)
    parser.add_argument("--rows", type=int, default=3, help="Number of rows.", required=True)
    parser.add_argument("--direction", type=str, default="sx2dx", help="Direction of breaking bricks ('sx2dx' or 'dx2sx').", required=True)
    parser.add_argument("--action_type", type=str, default="fire", help="Allowed actions for the agent. ('fire', 'fire_ball' or 'ball').", required=True)
    parser.add_argument("--algorithm", type=str, default="sarsa", help="RL algorithm ('sarsa' or 'q').", required=True)
    parser.add_argument("--train_steps", type=int, default=500000, help="Number of training steps.", required=True)
    
    parser.add_argument("--brick-reward", type=int, default=5, help="The reward for breaking a brick.")
    parser.add_argument("--step-reward", type=float, default=-0.01, help="The reward (cost) when nothing happens.")
    parser.add_argument("--goal-reward", type=int, default=1000, help="The reward for satisfying the temporal goal.")
    parser.add_argument("--seed", type=int, default=99, help="Random seed.") #42
    parser.add_argument("--agent-config", type=str, default="src/agent_config.yaml", help="RL configuration for the agent.")
    return parser.parse_args()


def main(arguments):

    if arguments.imitation == "imitation":
        print("#######################################################################")
        print("# IMITATION LEARNING over Heterogeneous Agents with Restraining Bolts #")
        print("#######################################################################\n")
        
        print(">>>>>>>>>>>>>>>>> BREAKOUT environment <<<<<<<<<<<<<<<<<\n")
        expert_config = Map(yaml.safe_load(open(arguments.agent_config)))
        expert_config.__setattr__("lambda_", 0.0)
        print(">>>>>>>>>>>>>>>>> Run the expert ...\n")
        output_dir = run_expert(arguments, expert_config)
        
        learner_config = Map(yaml.safe_load(open(arguments.agent_config)))
        print(">>>>>>>>>>>>>>>>> Learn the automaton from traces ...\n")
        dfa = learn_dfa(output_dir)
        dfa_dot_file = os.path.join(output_dir, "learned_automaton")
        dfa.to_dot(dfa_dot_file)
        print("Check the file {}.svg".format(dfa_dot_file))
        
        print(">>>>>>>>>>>>>>>>> Run the learner...")
        run_learner(arguments, learner_config, dfa)
    else:
        print("#####################")
        print("# Restraining Bolts #")
        print("#####################\n")   
        
        print(">>>>>>>>>>>>>>>>> BREAKOUT environment <<<<<<<<<<<<<<<<<\n")
        agent_config = Map(yaml.safe_load(open(arguments.agent_config)))
        print(">>>>>>>>>>>>>>>>> Run the agent...\n")
        run_expert(arguments, agent_config)


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)