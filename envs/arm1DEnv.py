import os
import numpy as np
import opensim as osim
from gym import spaces
from rl_utils import print_warning
from rl_utils.plotter import Plotter
from random import uniform

"""
p_shouler = [ -0.013; 0.840];
p_elbow = [-0.013 ;  0.503];
p_wrist = [0.123 ;   0.306];

humerus = 0.33 [m]
radius = 0.24 [m]
"""
# radius: m
# mass: kg
# torque: N.m
# pos: rad
# vel: rad/s
# x,y: meters
# activation: ?
_USER_INSTANCE_PARAM=['radius','mass','torque']

_GOALS ={'state':True, 'upper':np.deg2rad(140), 'lower':np.deg2rad(10)}

_MAX_LIST = {"r_elbow":{'pos':np.deg2rad(150), 'vel': np.deg2rad(180)},
            "user_instance":{'radius':0.40, 'mass':15, 'torque':51.5},
            "TRIlong": {"act":1},
            "BIClong": {"act":1},
            "goal": {"angle":_GOALS['upper']}}

_MIN_LIST = {"r_elbow":{'pos':np.deg2rad(0), 'vel': np.deg2rad(-180)},
            "user_instance":{'radius':0.20, 'mass':1, 'torque':0},
            "TRIlong": {"act":0},
            "BIClong": {"act":0},
            "goal": {"angle":_GOALS['lower']}}

_MUSCLE_LIST = ["TRIlong", "BIClong"]            
_INIT_POS = {'r_shoulder':0, 'r_elbow':np.deg2rad(10)}
_REWARD = {'nan':-5, 'weird_joint_pos':-1, 'goal_achieved':3}     



class Arm1DEnv(object):
    def __init__(self, sim_time=3,
                integrator_accuracy = 5e-5, 
                step_size=0.01, 
                visualize=False,
                show_act_plot=False,
                fixed_init=False,
                with_fes = False):
        # simulation parameters
        self._int_acc = integrator_accuracy
        self._step_size = step_size
        self._show_act_plot = show_act_plot
        self._visualize = visualize
        self._max_sim_timesteps = sim_time/step_size
        self._sim_timesteps = 0 # current simulation steps 
        self.with_fes = with_fes # same activation for all biceps and all triceps muscles

        # RL environemnt parameters
        self._fixed_init = fixed_init    
        self._pos_des = None # desired wrist position
        self._initial_condition=None # initial joint configuration   


        # create model from .osim file
        self._model = osim.Model(os.path.join(os.path.dirname(__file__), '../models/Arm2D/arm2dof6musc.osim') )
        # enable the visualizer
        self._model.setUseVisualizer(self._visualize)  
        # animation of muscle's activation
        if self._visualize and self._show_act_plot:
            self._mus_plot = Plotter(nrows=2, ncols=1,max_simtime=sim_time,headers=_MUSCLE_LIST) 
            self._angle_plot = Plotter(nrows=1, ncols=1,max_simtime=sim_time,headers=['distance'])               
        

        # add bullet to model (extra mass)
        # Body: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Body.html
        # humerus link
        target_object = osim.Body("target_object", # name
                                5, # mass [kg]
                                osim.Vec3(0), # center of mass
                                osim.Inertia(1,1,1,0,0,0)) # inertia's moment [Ixx Iyy Izz Ixy Ixz Iyz]
        # add display geometry
        _geom = osim.Ellipsoid(0.025,0.025,0.025)
        _geom.setColor(osim.Orange)
        target_object.attachGeometry(_geom)      
                                  
        # PlanarJoint: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1PlanarJoint.html
        self.object_joint = osim.PlanarJoint("object_joint", # name
                                self._model.getBodySet().get("r_ulna_radius_hand"), # parent frame
                                osim.Vec3(0,0,0), # location
                                osim.Vec3(0,0,0), # rotation
                                target_object, # child frame
                                osim.Vec3(0,0,0), # location
                                osim.Vec3(0,0,0)) # rotation           

        # add goal body and joint to model
        self._model.addBody(target_object)
        self._model.addJoint(self.object_joint)
                
        # get handles for model parameters
        self._muscles = self._model.getMuscles() # get muscle
        self._bodies = self._model.getBodySet() # get bodies
        self._joints = self._model.getJointSet() # get joints
        self._markers = self._model.getMarkerSet() # get markers          


        # add control to each muscle
        # PrescribedController: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1PrescribedController.html
        # basic ontrol template
        self._brain = osim.PrescribedController()
        # quantity of muscles
        if self.with_fes:
            self._n_muscles = 2
        else:
            self._n_muscles = self._muscles.getSize()

        # Add muscle-actuators with constant-type function
        for idx in range(self._n_muscles):
            self._brain.addActuator(self._muscles.get(idx)) # add actuator for each muscle
            self._brain.prescribeControlForActuator(idx, osim.Constant(1.0)) # add a function to each muscle-actuator

        # add muscle's controllers to model
        self._model.addController(self._brain)
        # get func handle
        self._control_functions=self._brain.get_ControlFunctions()

        # initialize the model and check model consistency (important!)
        self._model.initSystem()
        self._state = None  # model's state
        self._manager = None # model's manager

        # observation space
        self.obs_names = [key+'_'+name for key, val in _MAX_LIST.items() for name in val]
        n_obs = len(self.obs_names)

        high = [var for lvl1 in _MAX_LIST.values() for var in lvl1.values()] 
        high = np.array(high, dtype=np.float32)

        low = [var for lvl1 in _MIN_LIST.values() for var in lvl1.values()] 
        low = np.array(low, dtype=np.float32)  

        self.observation_space = spaces.Box(low=low, \
                                            high=high, \
                                            shape=(n_obs,))
        # action space
        self.action_space = spaces.Box(low=np.zeros((self._n_muscles,), dtype=np.float32), \
                                        high=np.ones((self._n_muscles,), dtype=np.float32), \
                                        shape=(self._n_muscles,))       
                                        
        self.goals_achieved = 0


    def get_user_parameters(self):
        #init_values = [uniform(_MIN_LIST['user_instance']['radius'], _MAX_LIST['user_instance']['radius']),
        #               uniform(_MIN_LIST['user_instance']['mass'], _MAX_LIST['user_instance']['mass'])]
        init_values = [uniform(_MIN_LIST['user_instance']['radius'], _MAX_LIST['user_instance']['radius']),
                       5]

        return dict(zip(['radius', 'mass'], init_values))        
        
    def get_observations(self):
        # compute forces: muscles
        self._model.realizeAcceleration(self._state)
        # bullet
        # - desired pose: x  y
        # marker elbow
        # - pos: x, y
        # - vel: x, y
        # marker wrist
        # - pos: x, y
        # - vel: x, y        
        # muscles
        # - tri
        # - bi
        # observation list
        obs = []

        # joint position
        for joint_name in ['r_elbow']:
            obs.append(self._joints.get(joint_name).getCoordinate().getValue(self._state))
            obs.append(self._joints.get(joint_name).getCoordinate().getSpeedValue(self._state))        

        # user parameters
        self.user_parameters['torque'] = self.user_parameters['radius']*np.sin(obs[0])*self.user_parameters['mass']*9.81
        for name in _USER_INSTANCE_PARAM:
            obs.append(self.user_parameters[name])

        # muscle activation
        for muscle_name in _MUSCLE_LIST:             
            obs.append(self._muscles.get(muscle_name).getActivation(self._state))

        # desired elbow angular position
        obs.append(self.goal_angle)            

        return obs, dict(zip(self.obs_names, obs))

    def normalize_observations(self, obs):    
        return (obs-self.observation_space.low)/(self.observation_space.high-self.observation_space.low)

    def initial_joint_configuration(self):
        if not self._fixed_init:
            init_pos =  [uniform(_MIN_LIST['r_elbow']['pos']+np.deg2rad(10), _MAX_LIST['r_elbow']['pos']-np.deg2rad(10))]
            return dict(zip(['r_elbow'], init_pos))
        else:
            return _INIT_POS


    def actuate(self, action):
        """
        @info: apply stimulus to muscles
        """
        if np.any(np.isnan(action)): # safety condition
            action = np.nan_to_num(action)
            action = np.clip(action, 0, 1)
            
        # apply action
        for idx in range(self._control_functions.getSize()):
            # get control function of actuator "idx"
            func = osim.Constant.safeDownCast(self._control_functions.get(idx))
            # apply action
            func.setValue( float(action[idx]) )

    def get_sim_timesteps(self):
        return self._sim_timesteps

    def integrate(self):
        # update simulation step
        self._sim_timesteps = self._sim_timesteps + 1

        # compute next dynamic configuration
        self._state = self._manager.integrate(self._step_size*self._sim_timesteps)        

    def step_model(self, action):   
        self.actuate(action)
        self.integrate()        

    def reset_manager(self):
        self._manager = osim.Manager(self._model)
        self._manager.setIntegratorAccuracy(self._int_acc)
        self._manager.initialize(self._state)


    def reset_model(self, init_pos=None, bullet_pos=None):
        # set initial position
        if init_pos is not None:
            for joint_name in ['r_elbow']:
                self._joints.get(joint_name).upd_coordinates(0).setRangeMax(_MAX_LIST[joint_name]['pos'])
                self._joints.get(joint_name).upd_coordinates(0).setRangeMin(_MIN_LIST[joint_name]['pos'])                  
                self._joints.get(joint_name).upd_coordinates(0).setDefaultValue(init_pos[joint_name])
                self._joints.get(joint_name).upd_coordinates(0).setDefaultSpeedValue(0.0)
                self._joints.get(joint_name).upd_coordinates(0).setDefaultClamped(True)

        #self._bodies.get("target_object").set_mass(100)

        
        # fixed shoulder
        self._joints.get("r_shoulder").upd_coordinates(0).setDefaultLocked(True)
        
        # compute initial state (important!)
        self._state = self._model.initializeState()

        #_joints = self._model.getJointSet() # get joints
        #goal_joint = self._model.getJointSet().get("goal_joint")
        self.object_joint.get_coordinates(0).setValue(self._state, 0, False)
        self.object_joint.get_coordinates(0).setLocked(self._state, True)
        self.object_joint.get_coordinates(1).setValue(self._state, 0, False)
        self.object_joint.get_coordinates(1).setLocked(self._state, True)
        self.object_joint.get_coordinates(2).setLocked(self._state, False)
        self.object_joint.get_coordinates(2).setValue(self._state, -bullet_pos['radius'], False)
        self.object_joint.get_coordinates(2).setLocked(self._state, True)

        # compute length of fibers based on the state (important!)
        self._model.equilibrateMuscles(self._state)
        # simulation parameters
        self._sim_timesteps = 0
        self._state.setTime(self._sim_timesteps)
        # forward dynamics manager
        self.reset_manager()

    def get_goal_angle(self, metric):
        if abs(metric) <=np.deg2rad(5):
            _GOALS['state']= not _GOALS['state']
            self.goals_achieved+=1
            print(f"goals achieved: {self.goals_achieved}")
        return _GOALS['upper'] if _GOALS['state'] else _GOALS['lower']


    def reset(self, verbose=False):
        self.goals_achieved= 0

        # compute intitial joint configuration
        init_joint_pos = self.initial_joint_configuration()

        # compute wrist position
        self.user_parameters = self.get_user_parameters()

        self.goal_angle = self.get_goal_angle(metric=100)
        # reset model variables
        self.reset_model(init_pos=init_joint_pos, 
                        bullet_pos=self.user_parameters) 
        # get observations
        obs, obs_dict = self.get_observations()
        if verbose:
            print(f"radius: {self.user_parameters['radius']:.3f}")
            #print(f"goal pos: {obs_dict['pos_des_x']:.3f}, {obs_dict['pos_des_y']:.3f}")
            #print(f"shoulder pos: {obs_dict['r_acromion_x']:.3f}, {obs_dict['r_acromion_y']:.3f}")
            #print(f"elbow pos: {obs_dict['r_humerus_epicondyle_x']:.3f}, {obs_dict['r_humerus_epicondyle_y']:.3f}")
            #print(f"wrist pos: {obs_dict['r_radius_styloid_x']:.3f}, {obs_dict['r_radius_styloid_y']:.3f}")
            
        # muscle's activation
        if self._visualize and self._show_act_plot:
            self._mus_plot.reset()
            self._angle_plot.reset()
        # get observations and normalize
        obs = self.normalize_observations(obs)

        return obs


    def gaussian_reward(self, metric, max_error):
        mean = 0
        std = max_error/2
        return np.exp(-(metric-mean)**2/(2*std**2))            


    def step(self, act):
        # mean muscle's activation
        if self.with_fes:
            action = np.zeros((6,),dtype=float)
            action[0:3] = act[0] # triceps
            action[3:5] = act[1] # biceps 
            action[5] = 0
        
        # muscle's activation
        if self._visualize and self._show_act_plot:
            self._mus_plot.add_data(time=self._sim_timesteps*self._step_size, act=action)
            if self._show_act_plot and self._sim_timesteps%100==0:
                self._mus_plot.update_figure()       

        # apply action
        self.step_model(action=action)
        
        # get environemnt observations
        obs, obs_dict = self.get_observations()
        obs=self.normalize_observations(obs)      

        # compute distance from wrist to target point
        distance = obs_dict['goal_angle']-obs_dict['r_elbow_pos']
        # reward system
        reward = self.gaussian_reward(metric=distance, max_error=np.deg2rad(140)) # reward to achieve desired position 
        reward -= 0.001*sum(action) # punishment for inefficient motion
        reward -= 0.005*(obs_dict['r_elbow_vel'])**2 # punishment for high velocity

        # goal condition
        self.goal_angle = self.get_goal_angle(metric=distance)

        # terminal condition: nan observation
        if np.isnan(obs).any(): # check is there are nan values 
            #print_warning(f"terminal state for nan observations")
            obs = np.nan_to_num(obs)
            obs = np.clip(obs, 0, 1)
            return obs, _REWARD['nan'], True, {'sim_timesteps':self._sim_timesteps}

        # terminal condition: max simulation steps reached
        if not self._sim_timesteps < self._max_sim_timesteps:
            return obs, reward, True, {'sim_timesteps':self._sim_timesteps}

        # terminal condition: out  of bounds (joint pos or vel)

        if not np.logical_and(obs[0]<=1, obs[0]>=0):
            return obs, _REWARD['weird_joint_pos'], True, {'sim_timesteps':self._sim_timesteps}
        
        # all fine
        return obs, reward, False, {'sim_timesteps':self._sim_timesteps}
        

        