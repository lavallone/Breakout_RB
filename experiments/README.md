# Breakout 💥 Experiments
In the following there are all the results of the *experiments* made with the purpose of investigating and testinig further the techniques explained in "*Foundations for Restraining Bolt: Reinforcement Learning with LTLf/LDLf Restraining Specifications*" and in "*Imitation Learning over Heterogeneous Agents with Restraining Bolts*" papers. 

> The choosen environment is the **Breakout** game where the aim is to break all the present bricks. Thanks to ***restraining bolts*** (logical specifications which restrains agent behaviour in a custom manner), we can guide the way in which is possible to break the bricks. In these experiments it's shown how we can teach the breaking direction: both *left-to-right* ⏩⏩⏩ and *right-to-left* ⏪⏪⏪. We employed two different RL algorithms: ***Q-learning*** and ***SARSA***. In all the experiments is shown which one is used. 

To highlight strengths and weaknesses of the methods, we differentiated the experiments based on *bricks setting* (**3x3**, **4x4** and **8x3**) and on the *actions* the agent could use it (**only ball**⚾️  and **fire plus ball**🔥⚾️). We first tested the "plain" *restraining bolts* where the LTLf/LDLf specifications are already known in advance, and then, through a mechanism of *Imitation Learning*, *restraining bolts* where its logical specifications are learned by observing an *expert*. The expert agent in our case has a single state represantion and can only fire 🔥 (to make the learning of the optimal policy as simple as possible). The latest experiments show how the *learners* (agents which imitate expert's behaviour) behave conformly to the restraining bolt specifications without knowing them a priori.

## Restraining Bolts

### 3x3 bricks setting <br> <br> 🧱 🧱 🧱 <br> 🧱 🧱 🧱 <br> 🧱 🧱 🧱

#### ⚾️ BALL

<div style="display: flex; justify-content:space-between;">
  <figure style="text-align:center; flex:1;">
    <img src="gifs/3x3_ball/q-ball-3x3-dx2sx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>Q-learning</b> <br> ⏪⏪⏪</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/3x3_ball/sarsa-ball-3x3-dx2sx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏪⏪⏪</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/3x3_ball/q-ball-3x3-sx2dx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>Q-learning</b> <br> ⏩⏩⏩</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/3x3_ball/sarsa-ball-3x3-sx2dx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏩⏩⏩</figcaption>
  </figure>
</div>
__ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ 

#### 🔥⚾️ FIRE + BALL

<div style="display: flex; justify-content:space-between;">
  <figure style="text-align:center; flex:1;">
    <img src="gifs/3x3_fire_ball/q-fire-ball-3x3-dx2sx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>Q-learning</b> <br> ⏪⏪⏪</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/3x3_fire_ball/sarsa-fire-ball-3x3-dx2sx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏪⏪⏪</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/3x3_fire_ball/q-fire-ball-3x3-sx2dx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>Q-learning</b> <br> ⏩⏩⏩</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/3x3_fire_ball/sarsa-fire-ball-3x3-sx2dx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏩⏩⏩</figcaption>
  </figure>
</div>
___________________________________________________________________________________________________________

### 4x4 bricks setting <br> <br> 🧱 🧱 🧱 🧱 <br> 🧱 🧱 🧱 🧱 <br> 🧱 🧱 🧱 🧱 <br> 🧱 🧱 🧱 🧱

#### ⚾️ BALL

<div style="display: flex; justify-content:space-between;">
  <figure style="text-align:center; flex:1;">
    <img src="gifs/4x4_ball/q-ball-4x4-dx2sx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>Q-learning</b> <br> ⏪⏪⏪</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/4x4_ball/sarsa-ball-4x4-dx2sx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏪⏪⏪</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/4x4_ball/q-ball-4x4-sx2dx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>Q-learning</b> <br> ⏩⏩⏩</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/4x4_ball/sarsa-ball-4x4-sx2dx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏩⏩⏩</figcaption>
  </figure>
</div>
__ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ 

#### 🔥⚾️ FIRE + BALL

<div style="display: flex; justify-content:space-between;">
  <figure style="text-align:center; flex:1;">
    <img src="gifs/4x4_fire_ball/q-fire-ball-4x4-dx2sx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>Q-learning</b> <br> ⏪⏪⏪</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/4x4_fire_ball/sarsa-fire-ball-4x4-dx2sx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏪⏪⏪</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/4x4_fire_ball/q-fire-ball-4x4-sx2dx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>Q-learning</b> <br> ⏩⏩⏩</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/4x4_fire_ball/sarsa-fire-ball-4x4-sx2dx.gif" width="200" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏩⏩⏩</figcaption>
  </figure>
</div>
___________________________________________________________________________________________________________

### 8x3 bricks setting <br> <br> 🧱 🧱 🧱 <br> 🧱 🧱 🧱 <br> 🧱 🧱 🧱 <br> 🧱 🧱 🧱 <br> 🧱 🧱 🧱 <br> 🧱 🧱 🧱 <br> 🧱 🧱 🧱 <br> 🧱 🧱 🧱

<div style="display: flex; justify-content:space-between;">
  <figure style="text-align:center; flex:1;">
    <img src="gifs/8x3/ball-8x3.gif" width="200" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏩⏩⏩ <br> ⚾️</figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/8x3/fire-ball-8x3.gif" width="200" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏩⏩⏩ <br> 🔥⚾️</figcaption>
  </figure>
</div>
___________________________________________________________________________________________________________

## Imitation Learning

### Expert 👨🏼‍🏫

<div style="display: flex; justify-content:space-between;">
  <figure style="text-align:center; flex:1;">
    <img src="gifs/Imitation/EXPERT.gif" width="200" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏪⏪⏪ <br> 🔥 </figcaption>
  </figure>
</div>
__ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ 

### Learners 👨🏼‍🎓

<div style="display: flex; justify-content:space-between;">
  <figure style="text-align:center; flex:1;">
    <img src="gifs/Imitation/LEARNER-ball-3x3.gif" height="250" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏪⏪⏪ <br> ⚾️ <br> <b>3x3</b></figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/Imitation/LEARNER-fire-ball-3x3.gif" height="250" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏪⏪⏪ <br> 🔥⚾️ <br> <b>3x3</b></figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/Imitation/LEARNER-ball-4x4.gif" height="250" />
    <figcaption styles="text-align: center;"><b><b>SARSA</b> <br> ⏪⏪⏪ <br> ⚾️ <br> <b>4x4</b></figcaption>
  </figure>
  <figure style="text-align:center; flex:1;">
    <img src="gifs/Imitation/LEARNER-fire-ball-4x4.gif" height="250" />
    <figcaption styles="text-align: center;"><b>SARSA</b> <br> ⏪⏪⏪ <br> 🔥⚾️ <br> <b>4x4</b></figcaption>
  </figure>
</div>
___________________________________________________________________________________________________________