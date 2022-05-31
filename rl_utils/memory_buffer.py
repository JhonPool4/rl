import pandas as pd
import random
import torch
import numpy as np
import os
from .color_print import print_info

class MemoryBuffer():
    def __init__(self, obs_dim, act_dim, mem_size, batch_size, resume_training, save_path):
        self.mem_size = mem_size # number of trasitions to be stored
        self.mem_index = 0 # number of stored trajectories
        self.batch_size = batch_size # number of samples
        self.obs_dim = obs_dim # number of observations
        self.act_dim = act_dim # number of actions
        self.save_path = save_path # directory to save data
        self.memory_path = os.path.join(self.save_path, 'memory_buffer')
        self.allow_sample = False
        self.full_memory = False

        # create headers for each property
        self.obs_header = self.create_header('obs_', self.obs_dim)
        self.act_header = self.create_header('act_', self.act_dim)        
        self.new_obs_header = self.create_header('new_obs_', self.obs_dim)
        self.reward_header = ['reward']  
        self.done_header = ['done']

        # creating data frame structure
        column_names = []
        column_names += self.obs_header
        column_names += self.act_header
        column_names += self.reward_header
        column_names += self.new_obs_header
        column_names += self.done_header               
        
        
        if not resume_training: # create new file             
            self.df = pd.DataFrame(columns=column_names, dtype=object) 
            self.df.to_csv(self.memory_path, mode='w', index=False, header=True) 
            print_info(f"creating new memory buffer file")     
        else: # load file
            self.df = pd.read_csv(self.memory_path)
            self.full_memory = bool(len(self.df)>=self.mem_size)
            self.allow_sample = bool(len(self.df)>=self.batch_size)
            self.mem_index = len(self.df)%self.mem_size
            print_info(f"loading memory buffer from {self.memory_path}")
            print_info(f"memomry index: {self.mem_index}")
            print_info(f"allow sample: {self.allow_sample}")
            print_info(f"full memory: {self.full_memory}")

    def create_header(self, parameter_name, n_elements):
        return [parameter_name+str(i) for i in range(1, n_elements+1) ]

    def store_transition(self, obs, act, reward, new_obs, done):
        # get index to store or replace data
        if not (self.mem_index < self.mem_size):
            self.mem_index = 0
            self.full_memory = True

        # save data in .csv file
        self.df.loc[self.mem_index, self.obs_header]= obs
        self.df.loc[self.mem_index, self.act_header]= act
        self.df.loc[self.mem_index, self.reward_header]= reward
        self.df.loc[self.mem_index, self.new_obs_header]= new_obs 
        self.df.loc[self.mem_index, self.done_header]= done

        self.df.to_csv(self.memory_path, mode='w', index=False, header=True)
        # increase memory counter
        self.mem_index += 1

        if self.mem_index>=self.batch_size:
            self.allow_sample=True

    def sample_memory(self, batch_size):
        """
        @info get transitions from memory buffer
        @inputs:
            - batch_size: number of transitions
        @outputs:
            - transitions, numpy arrays
        """
        # number of available transitions
        if self.full_memory:
            max_mem = self.mem_size
        else:
            max_mem = min(self.mem_index, self.mem_size)
        # sample memory
        batch_index = random.sample(range(max_mem),batch_size)
        batch_mem = self.df.loc[batch_index]
        #batch_mem = pd.read_csv(self.memory_path,skiprows=skip)    
        
        #print(f"mem_index: {self.mem_index}, max_mem: {max_mem}, batch: {batch_index}")
        #print(self.df.loc[batch_index, self.obs_header].values)    


        return  torch.as_tensor(np.array(batch_mem[self.obs_header], dtype=float), dtype=torch.float), \
                torch.as_tensor(np.array(batch_mem[self.act_header], dtype=float), dtype=torch.float), \
                torch.as_tensor(np.array(batch_mem[self.reward_header], dtype=float), dtype=torch.float), \
                torch.as_tensor(np.array(batch_mem[self.new_obs_header], dtype=float), dtype=torch.float), \
                torch.as_tensor(np.array(batch_mem[self.done_header], dtype=float), dtype=torch.float)

    def save_memory_buffer(self):
        self.df.to_csv(self.memory_path, mode='w', index=False, header=True)
        print_info(f"memory buffer in {self.memory_path}")