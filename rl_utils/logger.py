from lib2to3.pgen2.literals import simple_escapes
import os
import pandas as pd
import matplotlib.pyplot as plt
from .color_print import print_info
import csv

class Logger():
    def __init__(self, save_path, print_rate=100, save_rate=100, resume_training=False):
        self.best_score = -1000
        self.print_rate = print_rate
        self.save_rate = save_rate
               
        # create data frame structure
        self.data_path = os.path.join(save_path, 'training_data')
        self.column_names=['epoch','score', 'pi_loss', 'q_loss', 'sim_time']

        # create new file
        if not resume_training:
            self.last_epoch = 0
            # create new file
            with open(self.data_path, 'w',newline='') as f:
                # create the csv writer
                csv_writer = csv.writer(f)
                # write header
                csv_writer.writerow(self.column_names)
            print_info(f"creating new training data file")
        else:
            tmp_df = pd.read_csv(self.data_path)
            self.last_epoch = int(tmp_df['epoch'].iloc[-1])##self.print_rate*len(tmp_df)
            del tmp_df
            print_info(f"loading training data from {self.data_path}")
            print_info(f"last epoch: {self.last_epoch}")

        # data buffer
        self.data = dict(zip(self.column_names, [0,[],[],[],[]]))
        # writer handle
        self.open_csv_handle()

    def open_csv_handle(self):
        # file and csv handle
        self.f = open(self.data_path, 'a', newline='')
        self.csv_writer = csv.writer(self.f)
    
    def close_csv_handle(self):
        # close file
        self.f.close()

    def print_data_buf(self, epoch, verbose=False):

        if len(self.data['pi_loss'])>0:
            # save training data
            self.csv_writer.writerow( [ self.last_epoch+epoch, \
                                        self.data['score'][-1], \
                                        self.data['pi_loss'][-1].item(), \
                                        self.data['q_loss'][-1].item(), \
                                        self.data['sim_time'][-1]])
            # print training data
            # new best score
            if self.data['score'][-1]>self.best_score:
                self.best_score=self.data['score'][-1]
            if epoch%self.print_rate==0 and len(self.data['pi_loss'])>=self.print_rate:
                # compute average values
                mean_pi_loss = sum(self.data['pi_loss'][-self.print_rate:])/self.print_rate
                mean_q_loss = sum(self.data['q_loss'][-self.print_rate:])/self.print_rate    
                mean_score = sum(self.data['score'])/len(self.data['score'])
                    
                
                # just to print
                if verbose:            
                    print(f"epoch: {self.last_epoch+epoch}, t: {self.data['sim_time'][-1]}, s: {self.data['score'][-1]:.1f}, avg_s: {mean_score:.1f}, best_s: {self.best_score:.1f}, pi_l: {mean_pi_loss:.2f}, q_l: {mean_q_loss:.2f}")   

    def reset_data_buffer(self):
        # clean data buffer
        self.data['pi_loss'] = []
        self.data['q_loss'] = []
        self.data['score'] = []   
        self.data['sim_time']=[]         
        # close file
        self.close_csv_handle()
        # open file
        self.open_csv_handle()
        print_info(f"training data in {self.data_path}")



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
