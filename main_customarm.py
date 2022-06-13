from envs import Arm2DEnv
from rl_agents import SAC


# create environemnt
env = Arm2DEnv(sim_time=4, 
                visualize=False,
                show_act_plot=True,
                show_goal=True, 
                fixed_init=False, 
                fixed_target=False,
                integrator_accuracy=1e-5,
                step_size=1e-2,
                with_fes = True)

# create agent
agent = SAC(env=env, 
            mem_size=10000, 
            batch_size=500, 
            gamma=0.99, 
            alpha=1.0, 
            dir_name='arm_7\\sac', 
            save_rate=500, 
            print_rate=1, 
            load_model=False,
            hidden_layers=(256,128,64,32))            


    
# train agent
agent.learn(n_epochs=20000, verbose=True, pulse_frequency_steps = 3)

#agent.test(n_attemps=10, verbose=True, pulse_frequency_steps = 3)