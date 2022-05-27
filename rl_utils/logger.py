import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .color_print import print_info

class Logger():
    def __init__(self, save_path, print_rate=100, resume_training=False):
        self.best_score = -1000
        self.print_rate = print_rate
               
        # create data frame structure
        self.data_path = os.path.join(save_path, 'training_data')
        self.column_names=['epoch','score', 'pi_loss', 'q_loss']
        self.df = pd.DataFrame(columns=self.column_names, dtype=object) 
        # create new file
        if not resume_training:
            self.last_epoch = 0
            self.df.to_csv(self.data_path, mode='w' ,index=False)   
            print_info(f"creating new training data file")
        else:
            tmp_df = pd.read_csv(self.data_path)
            self.last_epoch = int(tmp_df['epoch'].iloc[-1])##self.print_rate*len(tmp_df)
            del tmp_df
            print_info(f"loading training data from {self.data_path}")
            print_info(f"last epoch: {self.last_epoch}")


        self.data_buf = dict(zip(self.column_names, [0,[],[],[]]))

    def print_data_buf(self, epoch, verbose=False):

        if epoch%self.print_rate==0 and len(self.data_buf['pi_loss'])>0:
            mean_pi_loss = sum(self.data_buf['pi_loss'])/len(self.data_buf['pi_loss'])
            mean_q_loss = sum(self.data_buf['q_loss'])/len(self.data_buf['q_loss'])    
            mean_score = sum(self.data_buf['score'])/len(self.data_buf['score'])

            self.data_buf['epoch']=self.last_epoch+epoch
            self.data_buf['pi_loss'] = mean_pi_loss.item() # to avoid save in tensor format
            self.data_buf['q_loss'] = mean_q_loss.item() # to avoid save in tensor format
            self.data_buf['score'] = mean_score
                 
            # save training data
            self.df.append(self.data_buf, ignore_index=True).to_csv(self.data_path, mode='a', index=False, header=False)  

            # clean buffer
            self.data_buf['pi_loss'] = []
            self.data_buf['q_loss'] = []
            self.data_buf['score'] = []

            # new best score
            if mean_score>self.best_score:
                self.best_score=mean_score
            # just to print
            if verbose:            
                print(f"epoch: {self.last_epoch+epoch}, best_score: {self.best_score:.3f}, avg_score: {mean_score:.3f}, pi_loss: {mean_pi_loss:.3f}, q_loss: {mean_q_loss:.3f}")   

    def print_training_data(self):
        # read .text file
        new_df = pd.read_csv(self.data_path)

        fig, axs = plt.subplots(3)
        axs[0].set(xlabel='timesteps', ylabel='pi_loss')
        axs[0].plot(new_df['pi_loss'])
        axs[1].set(xlabel='timesteps', ylabel='q_loss')        
        axs[1].plot(new_df['q_loss'])
        axs[2].set(xlabel='timesteps', ylabel='reward')        
        axs[2].plot(new_df['score'])        
        plt.show()

