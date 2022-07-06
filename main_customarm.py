from envs import Arm1DEnv
from rl_agents import SAC


# create environemnt
env = Arm1DEnv(sim_time=12, 
                visualize=True,
                show_act_plot=False,
                fixed_init=True, 
                integrator_accuracy=1e-5,
                step_size=1e-2,
                with_fes = True)

# create agent
agent = SAC(env=env, 
            mem_size=10000, 
            batch_size=500, 
            gamma=0.99, 
            alpha=1.0, 
            dir_name='Arm1DEnv_FES_brach_freq_1_nn_64_32_16\\sac', 
            save_rate=10, 
            print_rate=1, 
            load_model=True,
            hidden_layers=(64,32,16))            



# train agent
#agent.learn(n_epochs=1000000, verbose=True, pulse_frequency_steps = 1, plot_tensorboard =True)

agent.test(n_attemps=30, verbose=True, pulse_frequency_steps = 1)