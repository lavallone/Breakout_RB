# 🔬🧪 Experiments on Restraining Bolts applied to Breakout game
<p align="center">
  <img src="https://i.imgur.com/q5Z5hKj.png" width="500" height="135">
</p>

This repository has been created in conjunction with the essay ***Learning and Reasoning as yin and yang of future AI*** that I produced for the *Planning and Reasoning* course held by professor Paolo Liberatore, where I adresssed the relative new research trend of helping *Reinforcemenet Learning* (**RL**) algorithms with *Knowledge Representation* (**KR**) and conducted an extensive analysis of pertinent research papers about the topic. To make the study even more complete, I decided to run some *experiments* 🔬🧪 in order to acquire practical, hands-on experience of the elucidated methodologies.

In particular I tested and implemented ***Restraining Bolts*** [1]. I was also interested in reproducing the results of [2], where *restraining bolts* are still involved but this time the logical specifications which aim to restrain agent behavior are learnt thanks to an *imitation learning* process. The designated RL enviroment for launching the experiments is the *Breakout* 🎮 environment. Choosing this type of game was, in my opinion, the most straightforward and smart way to let external readers comprehend the methods' inherent potential. 

> ⚠️ To see the results of all the experiments go to the **README.md** of *experiments* 🔬 folder!

## Setup
To run experiments from this repo you first need to create a Python 3.7 *conda* environment:
```
conda create --name KR_RL python=3.7
```
Then you need to install all the dependencies:
```
pip install -r requirements.txt
```

> 💡 [***temprl***](https://github.com/whitemech/temprl) and [***gym-breakout-pygame***](https://github.com/whitemech/gym-breakout-pygame) repositories are worthy to be mentioned. Both implemented by [WhiteMech](https://whitemech.github.io/) research group from Sapienza University. The first one offers a Reinforcement Learning framework for *temporal goals* while the other one defined a version of the Gym Breakout environment using *Pygame*. 

## Run
These are the *arguments* for running a single experiment:
* ***imitation***: if we want to use *imitation learning* or not. 
* ***direction***: in which direction we want to break bricks (*sx2dx* or *dx2sx*).
* ***algorithm***: *SARSA* or *Q-learning* RL algorithms.
* ***action_type***: which actions can the agent use.
* ***train_steps***: how many training steps we can execute.
* ***rows***: vertical number of bricks.
* ***cols***: horizontal number of bricks.

This is an example run for an RL agent in a **4x4** bricks setting scenario, which uses **SARSA** algorithm and only **ball** as action, with *2.000.000* training steps and through *imitation learning*, tries to learn to break the bricks from left to right (**sx2dx**).
```
python -m src --imitation no_imitation --direction sx2dx --algorithm sarsa --action_type ball --train_steps 2000000 --rows 4 --cols 4
```

## Implementation
I didn't implemented all the code from scratch. I started from the [official repo](https://github.com/whitemech/Imitation-Learning-over-Heterogeneous-Agents-with-Restraining-Bolts) of *Imitation Learning over Heterogeneous Agents with Restraining Bolts* paper [2]. My primary focus wasn't a re-implementation of the method, but the execution of the official code to see the effectiveness of it under different scenarios and circumstances (see the possible *arguments* from previous section). I basically adapted the official code to my needs, which include the execution of all the  experiments in the *Breakout* environment and the possibility to run *restraining bolts* with LTLf/LDLf formulas either already specified, or to be learnt (*imitation learning*). <br> All the RL related code is contained in *rl_algorithm* folder, where things such as RL algorithms, decision policies and environment wrappers are stored. In the *src* folder there's instead the *main* function with *expert.py* and *learner.py* files. The latter is used in the *imitation learning* scenario when an agent has to imitate the expert's behaviour. This is done by computing a DFA to be as similar as possible to the one incorporated in the expert's *restraining bolt* which represents its logical specifications and indeed its behaviour. We practically achieve this by employing [inferrer](https://github.com/whitemech/inferrer), an automata learning library, made by *WhiteMech* group. The DFA is extracted using the $L^*$ method which exploits the set of positive and negative traces produced by the automata of the *expert*. Below there's the code that returns the approximated DFA for the the *learner* agent:

```
def learn_dfa(output_dir) -> pythomata.dfa.DFA:
    positive_traces = os.path.join(output_dir, "positive_traces.txt")
    negative_traces = os.path.join(output_dir, "negative_traces.txt")
    dfa = inferrer.learn.learn(positive_traces, negative_traces, algorithm_id="lstar", separator=";")
    dfa = cast(DFAWrapper, dfa)
    new_dfa = from_inferrer_to_pythomata(dfa)
    new_dfa = new_dfa.complete()
    new_dfa = post_process_dfa(new_dfa)
    return new_dfa
```

Depending on the case, *expert.py* takes on different "roles". In the *imitation learning* scenario represents the *expert* which produces traces for the *learner*. In the normal scenario, instead, it embodies the *Restraining Bolt* agent that needs both to learn the optimal policy and to act so as to conform as much as possible to the LTLF/LDLf specifications. 

## References
<a id="1">[1]</a> 
De Giacomo, G., Iocchi, L., Favorito, M., & Patrizi, F. (2019). Foundations for Restraining Bolts: Reinforcement Learning with LTLf/LDLf Restraining Specifications. Proceedings of the International Conference on Automated Planning and Scheduling, 29(1), 128-136.

<a id="2">[2]</a> 
De Giacomo, G., Favorito, M., Iocchi, L., & Patrizi, F. (2020). Imitation Learning over Heterogeneous Agents with Restraining Bolts. Proceedings of the International Conference on Automated Planning and Scheduling, 30(1), 517-521.
