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


_JOINT_LIST = ["r_shoulder", "r_elbow"] 
_MARKER_LIST =  ["r_radius_styloid"]#, "r_humerus_epicondyle"]
_AXIS_LIST = ["x", "y"]
_MUSCLE_LIST = ["TRIlong", "TRIlat", "TRImed", \
                "BIClong", "BICshort", "BRA"]

# pos: rad
# vel: rad/s
# x,y: meters
# activation: ?
_MAX_LIST = {"pos_des":{'x':0.5, 'y':0.8}, \
            "r_shoulder":{'pos':np.deg2rad(180), 'vel': np.deg2rad(180)}, \
            "r_elbow":{'pos':np.deg2rad(150), 'vel': np.deg2rad(180)}, \
            "r_radius_styloid":{'x':0.5, 'y':0.8} , \
            #"r_humerus_epicondyle":{'x':0.24, 'y':0.8} , \
            "TRIlong": {"act":1},\
            "TRIlat": {"act":1},\
            "TRImed": {"act":1},\
            "BIClong": {"act":1},\
            "BICshort": {"act":1},\
            "BRA": {"act":1}}

_MIN_LIST = {"pos_des":{'x':-0.5, 'y':0.27}, \
            "r_shoulder":{'pos':np.deg2rad(-90), 'vel': -np.deg2rad(180)}, \
            "r_elbow":{'pos':np.deg2rad(0), 'vel': -np.deg2rad(180)}, \
            "r_radius_styloid":{'x':-0.5, 'y':0.27}, \
            #"r_humerus_epicondyle":{'x':-0.24, 'y':0.51} , \
            "TRIlong": {"act":0},\
            "TRIlat": {"act":0},\
            "TRImed": {"act":0},\
            "BIClong": {"act":0},\
            "BICshort": {"act":0},\
            "BRA": {"act":0}}             

#_POS_DES = {'x':0.2, 'y':0.6}
_DES_PARAM = {'theta':np.deg2rad(0), 'radio':0.2370}
_INIT_POS = {'r_shoulder':0, 'r_elbow':np.deg2rad(15)}
_REWARD = {'nan':-5, 'weird_joint_pos':-1}

class ArmEnv2D():

    def __init__(self, sim_time=3,
                 fixed_target=True, 
                 fixed_init=True, 
                 show_goal=False, 
                 visualize=False, 
                 integrator_accuracy = 1e-5, 
                 step_size=1e-2):
        # load arm model
        model_path = os.path.join(os.path.dirname(__file__), '../models/arm2dof6musc.osim')    
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

        # model configuration
        self.osim_model.update_joint_limits(joint_list=_JOINT_LIST,\
                                                 max_list=_MAX_LIST, min_list=_MIN_LIST)

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
            init_pos =  [uniform(0.001*_MIN_LIST['r_shoulder']['pos'], 0.001*_MAX_LIST['r_shoulder']['pos']), \
                         uniform(_MIN_LIST['r_elbow']['pos']+np.deg2rad(10), _MAX_LIST['r_elbow']['pos']-np.deg2rad(10))]
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
            return obs, _REWARD['nan'], True, {'sim_time':self.osim_model._sim_timesteps}

        if not self.osim_model._sim_timesteps < self.max_timesteps:
            return obs, reward, True, {'sim_time':self.osim_model._sim_timesteps}

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






        


