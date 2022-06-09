import os
import numpy as np
import opensim as osim
from gym import spaces
from rl_utils import print_warning
from rl_utils.plotter import Plotter
from random import uniform

_JOINT_LIST = ["r_shoulder", "r_elbow"] 
_MARKER_LIST =  ["r_radius_styloid"]
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
            "TRIlong": {"act":0},\
            "TRIlat": {"act":0},\
            "TRImed": {"act":0},\
            "BIClong": {"act":0},\
            "BICshort": {"act":0},\
            "BRA": {"act":0}}      

_DES_PARAM = {'theta':np.deg2rad(0), 'radio':0.2370}
_INIT_POS = {'r_shoulder':0, 'r_elbow':np.deg2rad(15)}
_REWARD = {'nan':-5, 'weird_joint_pos':-1}            

class Arm2DEnv(object):
    def __init__(self, sim_time=3,
                integrator_accuracy = 5e-5, 
                step_size=0.01, 
                visualize=False,
                show_goal=False,
                show_act_plot=False,                
                fixed_target=False,
                fixed_init=False):
        # simulation parameters
        self._int_acc = integrator_accuracy
        self._step_size = step_size
        self._show_goal = show_goal
        self._show_act_plot = show_act_plot
        self._visualize = visualize
        self._max_sim_timesteps = sim_time/step_size
        self._sim_timesteps = 0 # current simulation steps  

        # RL environemnt parameters
        self._fixed_target = fixed_target
        self._fixed_init = fixed_init    
        self._pos_des = None # desired wrist position
        self._initial_condition=None # initial joint configuration   


        # create model from .osim file
        self._model = osim.Model(os.path.join(os.path.dirname(__file__), '../models/Arm2D/arm2dof6musc.osim') )
        # enable the visualizer
        self._model.setUseVisualizer(self._visualize)  
        # animation of muscle's activation
        if self._show_act_plot:
            self._mus_plot = Plotter(max_simtime=sim_time,headers=_MUSCLE_LIST)               
        
        # add bullet to model
        if self._visualize and self._show_goal:
            # Body: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Body.html
            # humerus link
            goal_body = osim.Body("goal_body", # name
                                    0.0001, # mass
                                    osim.Vec3(0), # center of mass
                                    osim.Inertia(1,1,1,0,0,0)) # inertia's moment [Ixx Iyy Izz Ixy Ixz Iyz]
            # add display geometry
            _geom = osim.Ellipsoid(0.025,0.025,0.025)
            _geom.setColor(osim.Green)
            goal_body.attachGeometry(_geom)                                
            # PlanarJoint: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1PlanarJoint.html
            self.goal_joint = osim.PlanarJoint("goal_joint", # name
                                    self._model.getGround(), # parent frame
                                    osim.Vec3(0,0,0), # location
                                    osim.Vec3(0,0,0), # rotation
                                    goal_body, # child frame
                                    osim.Vec3(0,0,-0.25), # location
                                    osim.Vec3(0,0,0)) # rotation           

            # add goal body and joint to model
            self._model.addBody(goal_body)
            self._model.addJoint(self.goal_joint)

        # add markers
        # write code
        
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
        high = [var for lvl1 in _MAX_LIST.values() for var in lvl1.values()] 
        n_obs = len(high)
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

        
    def update_joint_limits(self, joint_list, max_list, min_list):
        """
        @info update position range of each joint in "joint_list"
        """
        for joint_name in joint_list:
            self._joints.get(joint_name).upd_coordinates(0).setRangeMax(max_list[joint_name]['pos'])
            self._joints.get(joint_name).upd_coordinates(0).setRangeMin(min_list[joint_name]['pos'])
            self._joints.get(joint_name).upd_coordinates(0).setDefaultSpeedValue(0.0)
            self._joints.get(joint_name).upd_coordinates(0).setDefaultClamped(True)

    def get_observations(self):
        # compute forces: muscles
        self._model.realizeAcceleration(self._state)
        # bullet
        # - desired pose: x  y
        # marker
        # - wrist marker: x y
        # joints
        # - position,  velocity

        # observation list
        obs = []

        # desired wrist position
        for pos in self.pos_des.values():
            obs.append(pos)
        
        # joint position
        for joint_name in _JOINT_LIST:
            obs.append(self._joints.get(joint_name).getCoordinate().getValue(self._state))
            obs.append(self._joints.get(joint_name).getCoordinate().getSpeedValue(self._state))

        # marker position
        for marker_name in _MARKER_LIST:
            for axis_name in range(len(_AXIS_LIST)):
                obs.append(self._markers.get(marker_name).getLocationInGround(self._state)[axis_name])

        # muscle activation
        for idx, muscle_name in enumerate(_MUSCLE_LIST):             
            obs.append(self._muscles.get(muscle_name).getActivation(self._state))
        return obs

    def normalize_observations(self, obs):
        # desired pos: 0 1
        # joint pos and vel: 2 3  4 5
        # marker pos: 6 7
        # muscle activation        
        return (obs-self.observation_space.low)/(self.observation_space.high-self.observation_space.low)

    def initial_joint_configuration(self):
        if not self._fixed_init:
            init_pos =  [uniform(0.001*_MIN_LIST['r_shoulder']['pos'], 0.001*_MAX_LIST['r_shoulder']['pos']), \
                         uniform(_MIN_LIST['r_elbow']['pos']+np.deg2rad(10), _MAX_LIST['r_elbow']['pos']-np.deg2rad(10))]
            return dict(zip(_JOINT_LIST, init_pos))
        else:
            return {name:_INIT_POS[name] for name in _JOINT_LIST}


    def get_goal(self):
        if self._fixed_target:
            theta = _DES_PARAM["theta"]- np.deg2rad(70)
            radio = _DES_PARAM["radio"]
            return  {'x':radio*np.cos(theta), 'y':radio*np.sin(theta) + 0.563}
        else:
            theta = uniform(_MIN_LIST["r_elbow"]["pos"], 0.7*_MAX_LIST["r_elbow"]["pos"]) - np.deg2rad(70)
            radio = _DES_PARAM["radio"]

            return {'x':radio*np.cos(theta), 'y':radio*np.sin(theta) + 0.563}   


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
            for name in _JOINT_LIST:
                self._joints.get(name).upd_coordinates(0).setDefaultValue(init_pos[name])
        
        # compute initial state (important!)
        self._state = self._model.initializeState()
        if self._visualize and self._show_goal and bullet_pos is not None:
            #_joints = self._model.getJointSet() # get joints
            #goal_joint = self._model.getJointSet().get("goal_joint")
            self.goal_joint.get_coordinates(1).setValue(self._state, bullet_pos['x'], False)
            self.goal_joint.get_coordinates(2).setLocked(self._state, False)
            self.goal_joint.get_coordinates(2).setValue(self._state, bullet_pos['y'], False)
            self.goal_joint.get_coordinates(2).setLocked(self._state, True)

        # compute length of fibers based on the state (important!)
        self._model.equilibrateMuscles(self._state)
        # simulation parameters
        self._sim_timesteps = 0
        self._state.setTime(self._sim_timesteps)
        # forward dynamics manager
        self.reset_manager()

    def reset(self, verbose=False):
        # compute intitial joint configuration
        init_joint_pos = self.initial_joint_configuration()

        # compute wrist position
        self.pos_des = self.get_goal() # x, y


        # reset model variables
        self.reset_model(init_pos=init_joint_pos, \
                                bullet_pos=self.pos_des) 

        # get observations
        obs = self.get_observations()
        if verbose:
            print(f"goal pos: {obs[0]:.3f}, {obs[1]:.3f}")
            print(f"wrist pos: {obs[6]:.3f}, {obs[7]:.3f}")
            print(f"elbow pos: {obs[8]:.3f}, {obs[9]:.3f}")

        # muscle's activation
        if self._show_act_plot:
            self._mus_plot.reset()
        # get observations and normalize
        return self.normalize_observations(obs=obs)

    def gaussian_reward(self, metric, max_error):
        mean = 0
        std = max_error/2
        return np.exp(-(metric-mean)**2/(2*std**2))            


    def step(self, action):
        # muscle's activation
        if self._show_act_plot:
            self._mus_plot.add_data(time=self._sim_timesteps*self._step_size, act=action)
            if self._show_act_plot and self._sim_timesteps%100==0:
                self._mus_plot.update_figure()        

        # apply action
        self.step_model(action=action)
        # get environemnt observations
        obs=self.normalize_observations(self.get_observations())

        # compute distance from wrist to target point
        distance = ((obs[6]-obs[0])**2 + (obs[7]-obs[1])**2)**0.5
        # reward system
        reward = self.gaussian_reward(metric=distance, max_error=0.3) # reward to achieve desired position 
        reward -= 0.01*sum(action) # punishment for inefficient motion
       
        # terminal condition: nan observation
        if np.isnan(obs).any(): # check is there are nan values 
            #print_warning(f"terminal state for nan observations")
            obs = np.nan_to_num(obs)
            obs = np.clip(obs, 0, 1)
            return obs, _REWARD['nan'], True, {'sim_time':self._sim_timesteps}
        
        # terminal condition: max simulation steps reached
        if not self._sim_timesteps < self._max_sim_timesteps:
            return obs, reward, True, {'sim_time':self._sim_timesteps}

        # terminal condition: out of bounds (joint pos or vel)
        if not np.logical_and(np.all(np.array(obs)<=1), np.all(np.array(obs)>=0)):
            #print_warning(f"terminal state for weird joint position or velocity")

            return obs, _REWARD['weird_joint_pos'], False, {'sim_time':self._sim_timesteps}

        # all fine
        return obs, reward, False, {'sim_time':self._sim_timesteps}
        