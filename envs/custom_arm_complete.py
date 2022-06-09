import os
import numpy as np
from random import uniform
from gym import spaces
from rl_utils.color_print import print_warning, print_info
from envs.custom_osim_model import CustomOsimModel

# how to create a model from scratch
# https://simtk-confluence.stanford.edu:8443/display/OpenSim/Building+a+Dynamic+Walker+in+Matlab

# how to add fatigue to a muscle model
# https://simtk-confluence.stanford.edu:8443/display/OpenSim/Creating+a+Customized+Muscle+Model

# useful commands
#https://simtk-confluence.stanford.edu:8443/display/OpenSim/Common+Scripting+Commands


_JOINT_LIST = ["sc1", "sc2","sc3",\
               "ac1", "ac2","ac3",\
               "gh1", "gh2","gh3",\
               "hu", "ur"]#"rc"] 
_MARKER_LIST =  []#["r_radius_styloid"]#, "r_humerus_epicondyle"]

_AXIS_LIST = ["x", "y", "z"]
_MUSCLE_LIST = ["trap_scap_1", "trap_scap_2", "trap_scap_3", \
                "trap_scap_4", "trap_scap_5", "trap_scap_6", \
                "trap_scap_7", "trap_scap_8", "trap_scap_9", \
                "trap_scap10", "trap_scap11", "trap_clav_1", \
                "trap_clav_2", "lev_scap_1", "lev_scap_2", \
                "pect_min_1", "pect_min_2", "pect_min_3", \
                "pect_min_4", "rhomboid_1", "rhomboid_2", \
                "rhomboid_3", "rhomboid_4", "rhomboid_5", \
                "serr_ant_1", "serr_ant_2", "serr_ant_3", \
                "serr_ant_4", "serr_ant_5", "serr_ant_6", \
                "serr_ant_7", "serr_ant_8", "serr_ant_9", \
                "serr_ant10", "serr_ant11", "serr_ant12", \
                "delt_scap_1", "delt_scap_2", "delt_scap_3", \
                "delt_scap_4", "delt_scap_5", "delt_scap_6", \
                "delt_scap_7", "delt_scap_8", "delt_scap_9", \
                "delt_scap10", "delt_scap11", "delt_clav_1", \
                "delt_clav_2", "delt_clav_3", "delt_clav_4", \
                "coracobr_1", "coracobr_2", "coracobr_3", \
                "infra_1", "infra_2", "infra_3", \
                "infra_4", "infra_5", "infra_6", \
                "ter_min_1", "ter_min_2", "ter_min_3", \
                "ter_maj_1", "ter_maj_2", "ter_maj_3", \
                "ter_maj_4", "supra_1", "supra_2", \
                "supra_3", "supra_4","subscap_1", "subscap_2", "subscap_3", \
                "subscap_4", "subscap_5", "subscap_6", \
                "subscap_7", "subscap_8", "subscap_9", \
                "subscap10", "subscap11" "bic_l", \
                "bic_b_1", "bic_b_2", "tri_long_1", \
                "tri_long_2", "tri_long_3", "tri_long_4", \
                "lat_dorsi_1", "lat_dorsi_2", "lat_dorsi_3", \
                "lat_dorsi_4", "lat_dorsi_5", "lat_dorsi_6", \
                "pect_maj_t_1", "pect_maj_t_2", "pect_maj_t_3", \
                "pect_maj_t_4", "pect_maj_t_5", "pect_maj_t_6", \
                "pect_maj_c_1", "pect_maj_c_2", "tric_med_1", \
                "tric_med_2", "tric_med_3", "tric_med_4", \
                "tric_med_5", "brachialis_1", "brachialis_2", \
                "brachialis_3", "brachialis_4", "brachialis_5", \
                "brachialis_6", "brachialis_7", "brachiorad_1", \
                "brachiorad_2", "brachiorad_3", "pron_teres_1", \
                "pron_teres_2", "supinator_1", "supinator_2", \
                "supinator_3", "supinator_4", "supinator_5", \
                "pron_quad_1", "pron_quad_2", "pron_quad_3", \
                "tric_lat_1", "tric_lat_2", "tric_lat_3", \
                "tric_lat_4", "tric_lat_5", "anconeus_1", \
                "anconeus_2", "anconeus_3", "anconeus_4", \
                "anconeus_5"]

# pos: rad
# vel: rad/s
# x,y: meters
# activation: ?

              
max_activation_muscles = 1
_MAX_LIST = {"pos_des":{'x':0.5, 'y':0.8, 'z':0.5}, \
            "sc1":{'pos':-np.deg2rad(19), 'vel': np.deg2rad(7)}, \
            "sc2":{'pos':np.deg2rad(30), 'vel': np.deg2rad(7)}, \
            "sc3":{'pos':np.deg2rad(82), 'vel': np.deg2rad(7)}, \
            "ac1":{'pos':np.deg2rad(69), 'vel': np.deg2rad(7)}, \
            "ac2":{'pos':np.deg2rad(20), 'vel': np.deg2rad(7)}, \
            "ac3":{'pos':np.deg2rad(18), 'vel': np.deg2rad(7)}, \
            "gh1":{'pos':np.deg2rad(174), 'vel': np.deg2rad(7)}, \
            "gh2":{'pos':np.deg2rad(84), 'vel': np.deg2rad(7)}, \
            "gh3":{'pos':np.deg2rad(178), 'vel': np.deg2rad(7)}, \
            "hu":{'pos':np.deg2rad(140), 'vel': np.deg2rad(7)}, \
            "ur":{'pos':np.deg2rad(160), 'vel': np.deg2rad(7)}, \
            #"rc":{'pos':np.deg2rad(0), 'vel': np.deg2rad(0)}, \
            #"r_radius_styloid":{'x':0.5, 'y':0.8}, \
            #"r_humerus_epicondyle":{'x':0.24, 'y':0.8} , \
            "trap_scap_1": {"act":max_activation_muscles}, "trap_scap_2": {"act":max_activation_muscles}, "trap_scap_3": {"act":max_activation_muscles}, \
            "trap_scap_4": {"act":max_activation_muscles}, "trap_scap_5": {"act":max_activation_muscles}, "trap_scap_6": {"act":max_activation_muscles}, \
            "trap_scap_7": {"act":max_activation_muscles}, "trap_scap_8": {"act":max_activation_muscles}, "trap_scap_9": {"act":max_activation_muscles}, \
            "trap_scap10": {"act":max_activation_muscles}, "trap_scap11": {"act":max_activation_muscles}, "trap_clav_1": {"act":max_activation_muscles}, \
            "trap_clav_2": {"act":max_activation_muscles}, "lev_scap_1": {"act":max_activation_muscles}, "lev_scap_2": {"act":max_activation_muscles}, \
            "pect_min_1": {"act":max_activation_muscles}, "pect_min_2": {"act":max_activation_muscles}, "pect_min_3": {"act":max_activation_muscles}, \
            "pect_min_4": {"act":max_activation_muscles}, "rhomboid_1": {"act":max_activation_muscles}, "rhomboid_2": {"act":max_activation_muscles}, \
            "rhomboid_3": {"act":max_activation_muscles}, "rhomboid_4": {"act":max_activation_muscles}, "rhomboid_5": {"act":max_activation_muscles}, \
            "serr_ant_1": {"act":max_activation_muscles}, "serr_ant_2": {"act":max_activation_muscles}, "serr_ant_3": {"act":max_activation_muscles}, \
            "serr_ant_4": {"act":max_activation_muscles}, "serr_ant_5": {"act":max_activation_muscles}, "serr_ant_6": {"act":max_activation_muscles}, \
            "serr_ant_7": {"act":max_activation_muscles}, "serr_ant_8": {"act":max_activation_muscles}, "serr_ant_9": {"act":max_activation_muscles}, \
            "serr_ant10": {"act":max_activation_muscles}, "serr_ant11": {"act":max_activation_muscles}, "serr_ant12": {"act":max_activation_muscles}, \
            "delt_scap_1": {"act":max_activation_muscles}, "delt_scap_2": {"act":max_activation_muscles}, "delt_scap_3": {"act":max_activation_muscles}, \
            "delt_scap_4": {"act":max_activation_muscles}, "delt_scap_5": {"act":max_activation_muscles}, "delt_scap_6": {"act":max_activation_muscles}, \
            "delt_scap_7": {"act":max_activation_muscles}, "delt_scap_8": {"act":max_activation_muscles}, "delt_scap_9": {"act":max_activation_muscles}, \
            "delt_scap10": {"act":max_activation_muscles}, "delt_scap11": {"act":max_activation_muscles}, "delt_clav_1": {"act":max_activation_muscles}, \
            "delt_clav_2": {"act":max_activation_muscles}, "delt_clav_3": {"act":max_activation_muscles}, "delt_clav_4": {"act":max_activation_muscles}, \
            "coracobr_1": {"act":max_activation_muscles}, "coracobr_2": {"act":max_activation_muscles}, "coracobr_3": {"act":max_activation_muscles}, \
            "infra_1": {"act":max_activation_muscles}, "infra_2": {"act":max_activation_muscles}, "infra_3": {"act":max_activation_muscles}, \
            "infra_4": {"act":max_activation_muscles}, "infra_5": {"act":max_activation_muscles}, "infra_6": {"act":max_activation_muscles}, \
            "ter_min_1": {"act":max_activation_muscles}, "ter_min_2": {"act":max_activation_muscles}, "ter_min_3": {"act":max_activation_muscles}, \
            "ter_maj_1": {"act":max_activation_muscles}, "ter_maj_2": {"act":max_activation_muscles}, "ter_maj_3": {"act":max_activation_muscles}, \
            "ter_maj_4": {"act":max_activation_muscles}, "supra_1": {"act":max_activation_muscles}, "supra_2": {"act":max_activation_muscles}, \
            "supra_3": {"act":max_activation_muscles}, "supra_4": {"act":max_activation_muscles},"subscap_1": {"act":max_activation_muscles}, "subscap_2": {"act":max_activation_muscles}, "subscap_3": {"act":max_activation_muscles}, \
            "subscap_4": {"act":max_activation_muscles}, "subscap_5": {"act":max_activation_muscles}, "subscap_6": {"act":max_activation_muscles}, \
            "subscap_7": {"act":max_activation_muscles}, "subscap_8": {"act":max_activation_muscles}, "subscap_9": {"act":max_activation_muscles}, \
            "subscap10": {"act":max_activation_muscles}, "subscap11": {"act":max_activation_muscles}, "bic_l": {"act":max_activation_muscles}, \
            "bic_b_1": {"act":max_activation_muscles}, "bic_b_2": {"act":max_activation_muscles}, "tri_long_1": {"act":max_activation_muscles}, \
            "tri_long_2": {"act":max_activation_muscles}, "tri_long_3": {"act":max_activation_muscles}, "tri_long_4": {"act":max_activation_muscles}, \
            "lat_dorsi_1": {"act":max_activation_muscles}, "lat_dorsi_2": {"act":max_activation_muscles}, "lat_dorsi_3": {"act":max_activation_muscles}, \
            "lat_dorsi_4": {"act":max_activation_muscles}, "lat_dorsi_5": {"act":max_activation_muscles}, "lat_dorsi_6": {"act":max_activation_muscles}, \
            "pect_maj_t_1": {"act":max_activation_muscles}, "pect_maj_t_2": {"act":max_activation_muscles}, "pect_maj_t_3": {"act":max_activation_muscles}, \
            "pect_maj_t_4": {"act":max_activation_muscles}, "pect_maj_t_5": {"act":max_activation_muscles}, "pect_maj_t_6": {"act":max_activation_muscles}, \
            "pect_maj_c_1": {"act":max_activation_muscles}, "pect_maj_c_2": {"act":max_activation_muscles}, "tric_med_1": {"act":max_activation_muscles}, \
            "tric_med_2": {"act":max_activation_muscles}, "tric_med_3": {"act":max_activation_muscles}, "tric_med_4": {"act":max_activation_muscles}, \
            "tric_med_5": {"act":max_activation_muscles}, "brachialis_1": {"act":max_activation_muscles}, "brachialis_2": {"act":max_activation_muscles}, \
            "brachialis_3": {"act":max_activation_muscles}, "brachialis_4": {"act":max_activation_muscles}, "brachialis_5": {"act":max_activation_muscles}, \
            "brachialis_6": {"act":max_activation_muscles}, "brachialis_7": {"act":max_activation_muscles}, "brachiorad_1": {"act":max_activation_muscles}, \
            "brachiorad_2": {"act":max_activation_muscles}, "brachiorad_3": {"act":max_activation_muscles}, "pron_teres_1": {"act":max_activation_muscles}, \
            "pron_teres_2": {"act":max_activation_muscles}, "supinator_1": {"act":max_activation_muscles}, "supinator_2": {"act":max_activation_muscles}, \
            "supinator_3": {"act":max_activation_muscles}, "supinator_4": {"act":max_activation_muscles}, "supinator_5": {"act":max_activation_muscles}, \
            "pron_quad_1": {"act":max_activation_muscles}, "pron_quad_2": {"act":max_activation_muscles}, "pron_quad_3": {"act":max_activation_muscles}, \
            "tric_lat_1": {"act":max_activation_muscles}, "tric_lat_2": {"act":max_activation_muscles}, "tric_lat_3": {"act":max_activation_muscles}, \
            "tric_lat_4": {"act":max_activation_muscles}, "tric_lat_5": {"act":max_activation_muscles}, "anconeus_1": {"act":max_activation_muscles}, \
            "anconeus_2": {"act":max_activation_muscles}, "anconeus_3": {"act":max_activation_muscles}, "anconeus_4": {"act":max_activation_muscles}, \
            "anconeus_5": {"act":max_activation_muscles}}

_MIN_LIST = {#"r_radius_styloid":{'x':-0.5, 'y':0.27}, \
            #"r_humerus_epicondyle":{'x':-0.24, 'y':0.51} , \
            "pos_des":{'x':-0.5, 'y':0.27, 'z':0}, \
            "sc1":{'pos':-np.deg2rad(45), 'vel': -np.deg2rad(3)}, \
            "sc2":{'pos':np.deg2rad(5), 'vel': -np.deg2rad(3)}, \
            "sc3":{'pos':np.deg2rad(0), 'vel': -np.deg2rad(3)}, \
            "ac1":{'pos':np.deg2rad(33), 'vel': -np.deg2rad(3)}, \
            "ac2":{'pos':-np.deg2rad(12), 'vel': -np.deg2rad(3)}, \
            "ac3":{'pos':-np.deg2rad(17), 'vel': -np.deg2rad(3)}, \
            "gh1":{'pos':np.deg2rad(96), 'vel': -np.deg2rad(3)}, \
            "gh2":{'pos':np.deg2rad(12), 'vel': -np.deg2rad(3)}, \
            "gh3":{'pos':np.deg2rad(150), 'vel': -np.deg2rad(3)}, \
            "hu":{'pos':np.deg2rad(5), 'vel': -np.deg2rad(3)}, \
            "ur":{'pos':np.deg2rad(5), 'vel': -np.deg2rad(3)}, \
            #"rc":{'pos':-np.deg2rad(0), 'vel': -np.deg2rad(0)}, # this joint does nothing, is just an offset.
            "trap_scap_1": {"act":0}, "trap_scap_2": {"act":0}, "trap_scap_3": {"act":0}, \
            "trap_scap_4": {"act":0}, "trap_scap_5": {"act":0}, "trap_scap_6": {"act":0}, \
            "trap_scap_7": {"act":0}, "trap_scap_8": {"act":0}, "trap_scap_9": {"act":0}, \
            "trap_scap10": {"act":0}, "trap_scap11": {"act":0}, "trap_clav_1": {"act":0}, \
            "trap_clav_2": {"act":0}, "lev_scap_1": {"act":0}, "lev_scap_2": {"act":0}, \
            "pect_min_1": {"act":0}, "pect_min_2": {"act":0}, "pect_min_3": {"act":0}, \
            "pect_min_4": {"act":0}, "rhomboid_1": {"act":0}, "rhomboid_2": {"act":0}, \
            "rhomboid_3": {"act":0}, "rhomboid_4": {"act":0}, "rhomboid_5": {"act":0}, \
            "serr_ant_1": {"act":0}, "serr_ant_2": {"act":0}, "serr_ant_3": {"act":0}, \
            "serr_ant_4": {"act":0}, "serr_ant_5": {"act":0}, "serr_ant_6": {"act":0}, \
            "serr_ant_7": {"act":0}, "serr_ant_8": {"act":0}, "serr_ant_9": {"act":0}, \
            "serr_ant10": {"act":0}, "serr_ant11": {"act":0}, "serr_ant12": {"act":0}, \
            "delt_scap_1": {"act":0}, "delt_scap_2": {"act":0}, "delt_scap_3": {"act":0}, \
            "delt_scap_4": {"act":0}, "delt_scap_5": {"act":0}, "delt_scap_6": {"act":0}, \
            "delt_scap_7": {"act":0}, "delt_scap_8": {"act":0}, "delt_scap_9": {"act":0}, \
            "delt_scap10": {"act":0}, "delt_scap11": {"act":0}, "delt_clav_1": {"act":0}, \
            "delt_clav_2": {"act":0}, "delt_clav_3": {"act":0}, "delt_clav_4": {"act":0}, \
            "coracobr_1": {"act":0}, "coracobr_2": {"act":0}, "coracobr_3": {"act":0}, \
            "infra_1": {"act":0}, "infra_2": {"act":0}, "infra_3": {"act":0}, \
            "infra_4": {"act":0}, "infra_5": {"act":0}, "infra_6": {"act":0}, \
            "ter_min_1": {"act":0}, "ter_min_2": {"act":0}, "ter_min_3": {"act":0}, \
            "ter_maj_1": {"act":0}, "ter_maj_2": {"act":0}, "ter_maj_3": {"act":0}, \
            "ter_maj_4": {"act":0}, "supra_1": {"act":0}, "supra_2": {"act":0}, \
            "supra_3": {"act":0}, "supra_4": {"act":0},"subscap_1": {"act":0}, "subscap_2": {"act":0}, "subscap_3": {"act":0}, \
            "subscap_4": {"act":0}, "subscap_5": {"act":0}, "subscap_6": {"act":0}, \
            "subscap_7": {"act":0}, "subscap_8": {"act":0}, "subscap_9": {"act":0}, \
            "subscap10": {"act":0}, "subscap11": {"act":0}, "bic_l": {"act":0}, \
            "bic_b_1": {"act":0}, "bic_b_2": {"act":0}, "tri_long_1": {"act":0}, \
            "tri_long_2": {"act":0}, "tri_long_3": {"act":0}, "tri_long_4": {"act":0}, \
            "lat_dorsi_1": {"act":0}, "lat_dorsi_2": {"act":0}, "lat_dorsi_3": {"act":0}, \
            "lat_dorsi_4": {"act":0}, "lat_dorsi_5": {"act":0}, "lat_dorsi_6": {"act":0}, \
            "pect_maj_t_1": {"act":0}, "pect_maj_t_2": {"act":0}, "pect_maj_t_3": {"act":0}, \
            "pect_maj_t_4": {"act":0}, "pect_maj_t_5": {"act":0}, "pect_maj_t_6": {"act":0}, \
            "pect_maj_c_1": {"act":0}, "pect_maj_c_2": {"act":0}, "tric_med_1": {"act":0}, \
            "tric_med_2": {"act":0}, "tric_med_3": {"act":0}, "tric_med_4": {"act":0}, \
            "tric_med_5": {"act":0}, "brachialis_1": {"act":0}, "brachialis_2": {"act":0}, \
            "brachialis_3": {"act":0}, "brachialis_4": {"act":0}, "brachialis_5": {"act":0}, \
            "brachialis_6": {"act":0}, "brachialis_7": {"act":0}, "brachiorad_1": {"act":0}, \
            "brachiorad_2": {"act":0}, "brachiorad_3": {"act":0}, "pron_teres_1": {"act":0}, \
            "pron_teres_2": {"act":0}, "supinator_1": {"act":0}, "supinator_2": {"act":0}, \
            "supinator_3": {"act":0}, "supinator_4": {"act":0}, "supinator_5": {"act":0}, \
            "pron_quad_1": {"act":0}, "pron_quad_2": {"act":0}, "pron_quad_3": {"act":0}, \
            "tric_lat_1": {"act":0}, "tric_lat_2": {"act":0}, "tric_lat_3": {"act":0}, \
            "tric_lat_4": {"act":0}, "tric_lat_5": {"act":0}, "anconeus_1": {"act":0}, \
            "anconeus_2": {"act":0}, "anconeus_3": {"act":0}, "anconeus_4": {"act":0}, \
            "anconeus_5": {"act":0}}             

#_POS_DES = {'x':0.2, 'y':0.6}
_DES_PARAM = {'theta':np.deg2rad(0), 'radio':0.2370}
_INIT_POS = {'r_shoulder':0, 'r_elbow':np.deg2rad(15)}
_INIT_POS = {"sc1":-np.deg2rad(27), \
            "sc2":np.deg2rad(5), \
            "sc3":np.deg2rad(32), \
            "ac1":np.deg2rad(45), \
            "ac2":np.deg2rad(0.458), \
            "ac3":-np.deg2rad(12), \
            "gh1":np.deg2rad(43), \
            "gh2":-np.deg2rad(0.218), \
            "gh3":-np.deg2rad(34), \
            "hu":np.deg2rad(5), \
            "ur":np.deg2rad(5)}
            #"rc":np.deg2rad(0)} # this joint does nothing, is just an offset.}
_REWARD = {'nan':-5, 'weird_joint_pos':-1}

class ArmEnv3D():

    def __init__(self, sim_time=3,
                 fixed_target=True, 
                 fixed_init=True, 
                 show_goal=False, 
                 visualize=False, 
                 integrator_accuracy = 1e-5, 
                 step_size=1e-2):
        # load arm model
        model_path = os.path.join(os.path.dirname(__file__), '../models/Arm10DOF/completearm.osim')    
        
        

        # create osim model
        self.osim_model = CustomOsimModel(model_path=model_path,\
                                            visualize=visualize,\
                                            integrator_accuracy=integrator_accuracy,\
                                            step_size=step_size, \
                                            add_bullet=show_goal)
        
        # simulation parameters
        self.max_timesteps = sim_time/step_size
        self.sim_timesteps = 0
        self.show_goal = show_goal
        print(f"flag 1") 
        # model configuration
        self.osim_model.update_joint_limits(joint_list=_JOINT_LIST,\
                                                 max_list=_MAX_LIST, min_list=_MIN_LIST)
        print(f"flag 2") 
        # RL environemnt parameters
        self.fixed_target = fixed_target
        self.fixed_init = fixed_init    
        self.pos_des = None # desired wrist position
        self.initial_condition=None # initial joint configuration

        # observation space
        high = [var for lvl1 in _MAX_LIST.values() for var in lvl1.values()] 
        n_obs = len(high)
        high = np.array(high, dtype=np.float32)
        low = [var for lvl1 in _MIN_LIST.values() for var in lvl1.values()] 
        low = np.array(low, dtype=np.float32)  

       

        self.observation_space = spaces.Box(low=low, \
                                            high=high, \
                                            shape=(n_obs,))
        # action space
        self.action_space = spaces.Box(low=np.zeros((self.osim_model._n_muscles,), dtype=np.float32), \
                                        high=np.ones((self.osim_model._n_muscles,), dtype=np.float32), \
                                        shape=(self.osim_model._n_muscles,))

        
        
    def get_observations(self):
        # compute forces
        self.osim_model._model.realizeAcceleration(self.osim_model._state)
        # des x y
        # elbow pos vel
        # shoulder pos vel
        
        # wrist marker x y
        # elbow marker x y

        # observation list
        obs = []

        # desired wrist position
        for pos in self.pos_des.values():
            obs.append(pos)
        
        # joint position
        for joint_name in _JOINT_LIST:
            obs.append(self.osim_model._joints.get(joint_name).getCoordinate().getValue(self.osim_model._state))
            obs.append(self.osim_model._joints.get(joint_name).getCoordinate().getSpeedValue(self.osim_model._state))

        # marker position
        for marker_name in _MARKER_LIST:
            for axis_name in range(len(_AXIS_LIST)):
                obs.append(self.osim_model._markers.get(marker_name).getLocationInGround(self.osim_model._state)[axis_name])
                #print(f"marker {axis_name}: {self.osim_model._markers.get(marker_name).getLocationInGround(self.osim_model._state)[axis_name]}")
        # muscle activation
        for idx, muscle_name in enumerate(_MUSCLE_LIST):             
            obs.append(self.osim_model._muscles.get(muscle_name).getActivation(s=self.osim_model._state))
        return obs

    def normalize_observations(self, obs):
        # desired pos: 0 1
        # joint pos and vel: 2 3  4 5
        # marker pos: 6 7
        # muscle activation        
        return (obs-self.observation_space.low)/(self.observation_space.high-self.observation_space.low)

    def initial_joint_configuration(self):
        if not self.fixed_init:
            #init_pos =  [uniform(0.001*_MIN_LIST['r_shoulder']['pos'], 0.001*_MAX_LIST['r_shoulder']['pos']), \
            #             uniform(_MIN_LIST['r_elbow']['pos']+np.deg2rad(10), _MAX_LIST['r_elbow']['pos']-np.deg2rad(10))]
            
            init_pos = [0 for _ in range(12)]
            return dict(zip(_JOINT_LIST, init_pos))
        else:
            return {joint_name:_INIT_POS[joint_name] for joint_name in _JOINT_LIST}

    def get_goal(self):
        if self.fixed_target:
            theta = _DES_PARAM["theta"]- np.deg2rad(70)
            radio = _DES_PARAM["radio"]
            return  {'x':radio*np.cos(theta), 'y':radio*np.sin(theta) + 0.563}
        else:
            theta = uniform(_MIN_LIST["r_elbow"]["pos"], 0.7*_MAX_LIST["r_elbow"]["pos"]) - np.deg2rad(70)
            radio = _DES_PARAM["radio"]

            return {'x':radio*np.cos(theta), 'y':radio*np.sin(theta) + 0.563}
    
    def reset(self, verbose=False):
        # compute intitial joint configuration
        init_joint_pos = self.initial_joint_configuration()

        # compute wrist position
        self.pos_des = self.get_goal() # x, y

        # reset model variables
        self.osim_model.reset(init_pos=init_joint_pos, \
                                bullet_pos=self.pos_des) 

        # get observations
        obs = self.get_observations()
        if verbose:
            print(f"goal pos: {obs[0]:.3f}, {obs[1]:.3f}")
            print(f"wrist pos: {obs[6]:.3f}, {obs[7]:.3f}")
            print(f"elbow pos: {obs[8]:.3f}, {obs[9]:.3f}")
        # get observations and normalize
        return self.normalize_observations(obs=obs)


    def gaussian_reward(self, metric, max_error):
        mean = 0
        std = max_error/2
        return np.exp(-(metric-mean)**2/(2*std**2))        


    def step(self, action):
        # apply action
        self.osim_model.step(action=action)
        # get environemnt observations
        obs=self.normalize_observations(self.get_observations())

        # compute distance from wrist to target point
        distance = ((obs[6]-obs[0])**2 + (obs[7]-obs[1])**2)**0.5
        # reward system
        reward = self.gaussian_reward(metric=distance, max_error=0.3) # reward to achieve desired position 
        reward -= 0.01*sum(action) # punishment for inefficient motion
       
        # terminal condition: max simulation steps reached

        # check is there are nan values
        if np.isnan(obs).any(): 
            #print_warning(f"terminal state for nan observations")
            obs = np.nan_to_num(obs)
            obs = np.clip(obs, 0, 1)
            return obs, _REWARD['nan'], False, {'sim_time':self.osim_model._sim_timesteps}

        if not self.osim_model._sim_timesteps < self.max_timesteps:
            return obs, reward, False, {'sim_time':self.osim_model._sim_timesteps}

        # terminal condition: out of bounds (joint pos or vel)
        if not np.logical_and(np.all(np.array(obs)<=1), np.all(np.array(obs)>=0)):
            #print_warning(f"terminal state for weird joint position or velocity")
            #idx = 0
            #for obj_name in _MAX_LIST.keys():
            #    print(f"{obj_name}")
            #    for name in _MAX_LIST[obj_name].keys():
            #        print(f"\t{name}: {n_obs[idx]}")
            #        idx +=1
            #print(f"\n")
            return obs, _REWARD['weird_joint_pos'], False, {'sim_time':self.osim_model._sim_timesteps}

        # all fine
        return obs, reward, False, {'sim_time':self.osim_model._sim_timesteps}






        


