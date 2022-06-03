from envs import ArmEnv2D
from rl_agents import SAC
import numpy as np

# create environemnt
env = ArmEnv2D(sim_time=3, visualize=True, fixed_init=False, fixed_target=True)
# create agent
agent = SAC(env=env, mem_size=10000, batch_size=500, gamma=0.99, alpha=1.0, dir_name='arm\\sac', save_rate=200, print_rate=1, load_model=False)
# train agent
agent.learn(n_epochs=1000)

# test agent
#agent.test(n_attemps=10)
"""
# create environemnt
env = CustomArmEnv(visualize=True)
obs, done = env.reset(random_target=True,obs_as_dict=False), False

counter =0
while True:
    counter+=1
    obs,  reward, done, _ = env.step(np.zeros((6,)))
    if counter%100==0:
        print(f"obs:{obs}")
    #print(f"obs: {len(obs)}, {type(obs)}, r: {type(reward)}, d: {type(done)}")
"""

"""
Updates:
-------
    Env:
    ---
    - gaussian reward function to have positive reinforcement
    - observation vector has wrist's marker position (x,y). Thus, new observation vector is 14
    - punishment for weird motions from -1 to -0.1 to avoid high loss functions and reduce oscillations
    - new terminal condition for weird elbow angular position, this will accelerate the training process

    Agent:
    -----
    - third version of soft actor-critic algorithm: double Q networks instead of Vf network
    - load memory buffer after resuming training
    - save models per epochs


To Do:
-----
    Agent:
    -----
    - plot of muscle activation: biceps should be greather than triceps


    GPU:
    ---
    - best pytorch version for python3
    - gpu characterisctis for our library

    Env:
    ---
    - normalize the observations to reduce oscillation 
    - set initial conditions
    - add electrical stimulation

    Luca:
    ----
    - documentation to create a musculoskeletal model (tutorial)
    - which signal we are applying?
    - what mean activation from 0 to 1?
    - 

"""