from rl_agents import SAC
from envs import ContinuousCartPoleEnv

# create environemnt
env = ContinuousCartPoleEnv(render_flag=True, 
                            sim_time=3)
# create agent
agent = SAC(env=env, 
            mem_size=1000, 
            batch_size=100, 
            gamma=0.99, 
            alpha=1.0, 
            dir_name='cart_pole\\sac_3', 
            save_rate=50, 
            print_rate=1, 
            load_model=True,
            hidden_layers=(256,128,64,32))
# train agent
#agent.learn(n_epochs=500, verbose=True)

# test agent
agent.test(n_attemps=10, verbose=True)