from envs import Arm2DEnv
from rl_agents import SAC
import numpy as np

pulse_frequency_steps = 3
integrator_accuracy=1e-5
step_size=1e-2
agent_params_directory ='arm_6\\sac'

load_model = True
train = True

if train:

    # create environemnt
    env = Arm2DEnv(sim_time=4, 
                    visualize=False,
                    show_act_plot=False,
                    show_goal=False, 
                    fixed_init=False, 
                    fixed_target=False,
                    integrator_accuracy=integrator_accuracy,
                    step_size=step_size,
                    with_fes = True)

    # create agent
    agent = SAC(env=env, 
                mem_size=10000, 
                batch_size=500, 
                gamma=0.99, 
                alpha=1.0, 
                dir_name=agent_params_directory, 
                save_rate=500, 
                print_rate=1, 
                load_model=load_model)


    
    # train agent
    agent.learn(n_epochs=20000, verbose=True,pulse_frequency_steps = pulse_frequency_steps)
else:
    # create environemnt
    env = Arm2DEnv(sim_time=4, 
                    visualize=True,
                    show_act_plot=False,
                    show_goal=True, 
                    fixed_init=False, 
                    fixed_target=False,
                    integrator_accuracy=integrator_accuracy,
                    step_size=step_size,
                    with_fes = True)

    # create agent
    agent = SAC(env=env, 
                mem_size=10000, 
                batch_size=500, 
                gamma=0.99, 
                alpha=1.0, 
                dir_name=agent_params_directory, 
                save_rate=500, 
                print_rate=1, 
                load_model=load_model)
    # test agent
    agent.test(n_attemps=50, verbose=True,pulse_frequency_steps = pulse_frequency_steps)






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
    - normalize the observations to reduce oscillation 
    - set initial conditions

    Agent:
    -----
    - third version of soft actor-critic algorithm: double Q networks instead of Vf network
    - load memory buffer after resuming training
    - save models per epochs

    GPU:
    ---
    - best pytorch version for python38
    - gpu characterisctis for our library

To Do:
-----
    Agent:
    -----
    - plot of muscle activation: biceps should be greather than triceps

    Env:
    ---
    - add electrical stimulation

    Luca:
    ----
    - documentation to create a musculoskeletal model (tutorial)
    - which signal we are applying?
    - what mean activation from 0 to 1?
    - 

"""