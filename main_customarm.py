from envs import Arm1DEnv,ExerciseEnv
from rl_agents import SAC

train = False

if train:
# create environemnt
    env = Arm1DEnv(sim_time=3, 
                    visualize=False,
                    show_act_plot=False,
                    fixed_init=False, 
                    fixed_target=False, 
                    integrator_accuracy=1e-5,
                    step_size=1e-2,
                    with_fes=True)

    # create agent
    agent = SAC(env=env, 
                mem_size=10000, 
                batch_size=500, 
                gamma=0.99, 
                alpha=1.0, 
                dir_name='Arm1DEnv_FES_brach_freq_1_nn_64_32_16_v3\\sac', 
                save_rate=50, 
                print_rate=1, 
                start_lr = 5e-4,
                load_model=True,
                hidden_layers=(64,32,16))            



    # train agent
    agent.learn(n_epochs=1000000, verbose=True, pulse_frequency_steps = 1, plot_tensorboard =True)
else:
    env = ExerciseEnv(sim_time=12, 
                    visualize=True,
                    show_act_plot=False,
                    fixed_init=True, 
                    #fixed_target=False, 
                    integrator_accuracy=1e-5,
                    step_size=1e-2,
                    with_fes=True)

    # create agent
    agent = SAC(env=env, 
                mem_size=10000, 
                batch_size=500, 
                gamma=0.99, 
                alpha=1.0, 
                dir_name='Arm1DEnv_FES_brach_freq_1_nn_64_32_16_v3\\sac', 
                save_rate=50, 
                print_rate=1, 
                start_lr = 1e-4,
                load_model=True,
                hidden_layers=(64,32,16))       

    agent.test(n_attemps=100, verbose=True, pulse_frequency_steps = 1)