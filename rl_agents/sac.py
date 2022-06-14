from rl_utils import GaussianPolicyNetwork
from rl_utils import DoubleQNetwork
from rl_utils import MemoryBuffer, Logger
from rl_utils import print_info
import torch
from torch.utils.tensorboard import SummaryWriter
from copy import copy
import os

class SAC():
    def __init__(self, 
                env, 
                mem_size=20000, 
                batch_size=1000, 
                gamma=0.98, 
                alpha=1.0, 
                dir_name='./task/agent',
                save_rate=100,
                print_rate=10,
                load_model=False,
                hidden_layers=list()):
        
        self.batch_size = batch_size
        self.gamma = gamma # discount factor
        self.alpha = alpha # temperature
        self.env = env # RL environment
        self.obs_dim = self.env.observation_space.shape[0] # number of observations
        self.act_dim = self.env.action_space.shape[0] # number of actions
        self.save_path = os.path.join(os.getcwd(), 'trained_models', dir_name) # directory to save training data
        self.save_rate = save_rate # number of epochs to save agent parameters

        print(f"============================================")
        print(f"\tInitializing agent parameters")
        print(f"============================================")  
        # create directory to save neural nets and training data
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print_info(f"creating training directory: {self.save_path}")    

        # create buffer
        self.mem_buffer = MemoryBuffer(self.obs_dim, self.act_dim, mem_size, batch_size, load_model, self.save_path)
        # create logger
        self.logger = Logger(self.save_path, print_rate=print_rate, save_rate=save_rate, resume_training=load_model)

        # create Q networks: (i) target and (ii) predict
        self.q_target = DoubleQNetwork(self.obs_dim, self.act_dim, hidden_layers)
        self.q_predict = DoubleQNetwork(self.obs_dim, self.act_dim, hidden_layers)

        # create policy
        self.pi_net = GaussianPolicyNetwork(self.obs_dim, self.act_dim, env.action_space, hidden_layers)

        # automatic entropy
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape))
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-4)

        # resume training
        if load_model:
            self.load_agent_parameters(load_path=self.save_path, last_epoch=self.logger.last_epoch)
            self.alpha=self.log_alpha.exp() 

    def save_agent_parameters(self, save_path, epoch):
        new_model_save_path = os.path.join(save_path,'agent_parameters', str(epoch))

        # create directory to save neural nets and training data
        if not os.path.exists(new_model_save_path):
            os.makedirs(new_model_save_path)
            print_info(f"creating directory to save agent parameters")  

        torch.save(self.q_target.state_dict(), os.path.join(new_model_save_path,'q_target_sac'))
        torch.save(self.q_predict.state_dict(), os.path.join(new_model_save_path,'q_predict_sac'))
        torch.save(self.pi_net.state_dict(), os.path.join(new_model_save_path,'pi_net_sac'))
        torch.save(self.log_alpha, os.path.join(new_model_save_path, 'log_alpha_sac'))
        print_info(f"saving agent parameters in {new_model_save_path}")
        

    def load_agent_parameters(self, load_path, last_epoch):
        # compute name (epoch) of the last model
        epoch = self.save_rate*int(last_epoch/self.save_rate)
        last_model_load_path = os.path.join(load_path,'agent_parameters', str(epoch))
        
        # load parameters of last model
        self.q_target.load_state_dict(torch.load(os.path.join(last_model_load_path, 'q_target_sac')))
        self.q_predict.load_state_dict(torch.load(os.path.join(last_model_load_path, 'q_predict_sac')))
        self.pi_net.load_state_dict(torch.load(os.path.join(last_model_load_path, 'pi_net_sac')))
        self.log_alpha =torch.load(os.path.join(last_model_load_path, 'log_alpha_sac'))
        print_info(f"loading agent parameters from {last_model_load_path}")

    def update_target_networks(self, tau):
        for q_target_param, q_predict_param in zip(self.q_target.parameters(), self.q_predict.parameters()):
            q_target_param.data.copy_(q_target_param.data*(1-tau) + q_predict_param*tau)


    def update_agent_parameters(self):
        # sample a transition (obs, act, reward, new_obs, done)
        obs, act, r, next_obs, done = self.mem_buffer.sample_memory(self.batch_size)  
        
        #print(f"obs: {obs.size()}")
        #print(f"act: {act.size()}")
        #print(f"r: {r.size()}")
        #print(f"new_obs: {next_obs.size()}")
        #print(f"done: {done.size()}")
        

        with torch.no_grad():
            # from numpy to torch.tensor
            #act = torch.tensor(act) #.reshape(self.batch_size, 1)
            #obs = torch.tensor(obs)
            #r = torch.tensor(reward)
            #next_obs = torch.tensor(next_obs)
            #mask = torch.logical_not(torch.tensor(done)).reshape((-1,1))
            mask = torch.logical_not(done)

            # compute TD target
            next_act, next_logprob = self.pi_net.predict_action(next_obs)
            next_q1_target, next_q2_target = self.q_target.forward(next_obs, next_act)
            y_t = r + mask*self.gamma*(torch.min(next_q1_target,next_q2_target) - self.alpha*next_logprob)

        # compute critic loss
        q1_predict, q2_predict = self.q_predict.forward(obs,act)
        q_loss = (0.5*(q1_predict-y_t)**2 + 0.5*(q2_predict-y_t)**2 ).mean()
        # update parameters
        self.q_predict.optimizer.zero_grad()
        q_loss.backward()
        self.q_predict.optimizer.step()

        # compute actor loss
        act, logprob = self.pi_net.predict_action(obs) 
        q1_predict, q2_predict = self.q_predict.forward(obs, act)
        pi_loss = (self.alpha*logprob - torch.min(q1_predict, q2_predict)).mean()
        # update parameters
        self.pi_net.optimizer.zero_grad()
        pi_loss.backward()
        self.pi_net.optimizer.step()    

        # compute entropy loss
        alpha_loss = -(self.log_alpha*(logprob+self.target_entropy).detach()).mean()
        # terminar 
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        # update alpha
        self.alpha = self.log_alpha.exp()       

        # just to print
        self.logger.data['pi_loss'].append(pi_loss.detach())
        self.logger.data['q_loss'].append(q_loss.detach())




    def learn(self, n_epochs, verbose=False, pulse_frequency_steps = None,plot_tensorboard = False):       
        print(f"================================")
        print(f"\tStarting Training")
        print(f"================================")
        # same networks
        writer = SummaryWriter()
        self.update_target_networks(tau=1)
        for epoch in range(1,n_epochs+1):
            # reset environment
            obs, reward, done = self.env.reset(verbose=verbose), 0, False
            score = 0
            
            
            while not done:
                # get action
                if pulse_frequency_steps is not None:
                    env_steps = self.env.get_sim_timesteps()
                    if env_steps % pulse_frequency_steps == 0:
                        act, _ = self.pi_net.predict_action(torch.tensor(obs, dtype=torch.float32)) 
                else:
                    act, _ = self.pi_net.predict_action(torch.tensor(obs, dtype=torch.float32)) 
                # interact with the environment
                new_obs, reward, done, info = self.env.step(act.detach().numpy())

                # store transition in memory
                self.mem_buffer.store_transition(obs, act.detach().numpy(), reward, new_obs, done)
                # update observation
                obs = copy(new_obs)
                # just to print
                score +=reward
                
                # update network paramteres: q_predict(critic) and pi_net(actor) 
                if self.mem_buffer.allow_sample:
                    self.update_agent_parameters()
                    self.update_target_networks(tau=0.05) 
                
            # plot epoch avg loss to tensorboard server
            if plot_tensorboard and len(self.logger.data['pi_loss'])>0:
                writer.add_scalar('pi_loss', sum(self.logger.data['pi_loss'])/len(self.logger.data['pi_loss']), epoch+self.logger.last_epoch)
                writer.add_scalar('Q_Loss', sum(self.logger.data['q_loss'])/len(self.logger.data['q_loss']), epoch+self.logger.last_epoch)
                writer.add_scalar('Score', score, epoch+self.logger.last_epoch)

     
            # just to print data
            self.logger.data['score'].append(score)
            self.logger.data['sim_time'].append(info['sim_timesteps'])
            self.logger.print_data_buf(epoch=epoch, verbose=verbose)

            # save neural network parameters
            if (epoch+self.logger.last_epoch)%self.save_rate==0:
                self.save_agent_parameters(save_path=self.save_path, epoch=epoch+self.logger.last_epoch)
                self.logger.reset_data_buffer()
                self.mem_buffer.save_memory_buffer()        
            #self.logger.print_training_data()


    def test(self, n_attemps, verbose=False,pulse_frequency_steps = None):
        print(f"============================")
        print(f"\tstaring test")
        print(f"============================")        
        for attemp in range(n_attemps):
            # reset environment
            obs, reward, done = self.env.reset(verbose=verbose), 0, False
            score = 0                

            while not done:
                # get action
                if pulse_frequency_steps is not None:
                        env_steps = self.env.get_sim_timesteps()
                        if env_steps % pulse_frequency_steps == 0:
                            act, _ = self.pi_net.predict_action(torch.tensor(obs, dtype=torch.float32)) 
                else:
                    act, _ = self.pi_net.predict_action(torch.tensor(obs, dtype=torch.float32)) 
                # interact with the environment
                new_obs, reward, done, info = self.env.step(act.detach().numpy())
                #render
                #self.env.render()
                # update observation
                obs = copy(new_obs)
                # just to print
                score +=reward        
            
            print(f"attemp: {attemp+1}, score: {score:.2f}, sim_time: {info['sim_timesteps']}")
            print(f"")