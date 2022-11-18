import os
import pandas as pd
from .color_print import print_info
import csv
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, 
                        save_path='./task/agent', 
                        save_rate=100, 
                        resume_training=False,
                        plot_tensorboard=False):
        self.best_score = -1000
        self.save_rate = save_rate
        self.plot_tensorboard = plot_tensorboard
        # create data frame structure
        self.data_path = os.path.join(save_path, 'training_data')
        self.column_names=['epoch','sim_timesteps','score', 'pi_loss', 'q_loss']

        if self.plot_tensorboard:
            self.tb_writer = SummaryWriter() # create a tensorboard plot        

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
            # load the table to get the last epoch, then delete it
            tmp_df = pd.read_csv(self.data_path)
            self.last_epoch = int(tmp_df['epoch'].iloc[-1])
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

    def end_episode(self, epoch):
        # number of elements
        n_epochs = len(self.data['score'])
        # save data
        if n_epochs>self.save_rate:
            n_timesteps = sum(self.data['sim_timesteps'])         
            # compute average values on one epoch
            mean_score = (sum(self.data['score']) / n_epochs).item()
            mean_timesteps = int(n_timesteps / n_epochs)
            mean_pi_loss = (sum(self.data['pi_loss'])/n_timesteps ).item()
            mean_q_loss = (sum(self.data['q_loss']) /n_timesteps ).item()
            # save training data
            self.csv_writer.writerow( [ self.last_epoch+epoch, \
                                        mean_timesteps, \
                                        mean_score, \
                                        mean_pi_loss, \
                                        mean_q_loss] )

            # plot epoch avg loss to tensorboard server
            if self.plot_tensorboard:
                self.tb_writer.add_scalar(f'Loss/pi_loss', mean_pi_loss, epoch+self.last_epoch)
                self.tb_writer.add_scalar(f'Loss/Q_Loss', mean_q_loss, epoch+self.last_epoch)

                self.tb_writer.add_scalar('Score', mean_score, epoch+self.last_epoch)
                self.tb_writer.add_scalar('Timesteps', mean_timesteps, epoch+self.last_epoch)                                        

            # clean data buffer
            self.reset_data_buffer()
            # print training data
            #if epoch%self.print_rate==0 and verbose:
                #print(f"epoch: {self.last_epoch+epoch}, t: {self.data['sim_timesteps']}, s: {self.data['score']:.1f}, pi_l: {mean_pi_loss:.2f}, q_l: {mean_q_loss:.2f}") 
                #print(f"")  

    def reset_data_buffer(self):
        # clean data buffer
        self.data['sim_timesteps'] = []
        self.data['score'] = []
        self.data['pi_loss'] = []
        self.data['q_loss'] = []         
        # close file
        self.close_csv_handle()
        # open file
        self.open_csv_handle()
        #print_info(f"training data in {self.data_path}")
