import opensim as osim
import numpy as np
from gym import spaces

_MAX_SHOULDER = {'pos':np.deg2rad(180), 'vel': 1} # rad and rad/s
_MIN_SHOULDER = {'pos':np.deg2rad(-90), 'vel': -1} # rad and rad/s

_MAX_ELBOW = {'pos':np.deg2rad(130), 'vel': 1} # rad and rad/s
_MIN_ELBOW = {'pos':np.deg2rad(0), 'vel': -1} # rad and rad/s

_MAX_WRIST_POS = {'x':2, 'y':4.5}
_MIN_WRIST_POS = {'x':-2, 'y':0.5}

_MAX_FIBER_ACTIVATION = 1
_MIN_FIBER_ACTIVATION = 0

_POS_DES = {'x':0.75, 'y':1}

class ArmModel2D():
    def __init__(self, integrator_accuracy=1e-5, visualize=True, sim_time=3, step_size=0.01):
        # model parameters
        self._global_reference = osim.Vec3(0, 2.5, 0)
        self._mass=1.0
        self._com=osim.Vec3(0,0,0)
        self._inertia=osim.Inertia(0,0,0)
        self._width=0.1
        self._length=0.5
        self._int_acc = integrator_accuracy        
        self._model = osim.Model() # crate a model
        self._model.setUseVisualizer(visualize=visualize) # enable or disable visualizer
        self._step_size = step_size # default 10 ms
        self._max_timesteps = sim_time/self._step_size
        self._sim_timesteps = 0
        

        # Body: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Body.html
        # humerus link
        self._humerus = osim.Body("humerus", # name
                                self._mass, # mass
                                self._com, # center of mass
                                self._inertia) # inertia's moment
        # humerus link
        
        self._radius = osim.Body("radius", # name
                                self._mass, # mass
                                self._com, # center of mass
                                self._inertia) # inertia's moment                                
        
        #_humerus_mesh = osim.Mesh('./models/humerus.vtp')
        #_humerus_mesh.set_scale_factors(osim.Vec3(1,1,1))
        #_radius_mesh = osim.Mesh('./models/radius.vtp')
        #_radius_mesh.set_scale_factors(osim.Vec3(1,1,1))

        # add display geometry
        _geom = osim.Ellipsoid(self._width, self._length, self._width)
        _geom.setColor(osim.Orange)
        self._humerus.attachGeometry(_geom.clone())
        _geom.setColor(osim.Gray)
        self._radius.attachGeometry(_geom.clone())
        

        # PinJoint: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1PinJoint.html
        # shoulder joint
        self._shoulder_joint = osim.PinJoint("shoulder", # name
                                    self._model.getGround(), # parent frame
                                    self._global_reference, # location
                                    osim.Vec3(0,0,0), # rotation
                                    self._humerus, # child frame
                                    osim.Vec3(0,self._length,0), # location
                                    osim.Vec3(0,0,0)) # rotation   

        self._elbow_joint = osim.PinJoint("elbow", # name
                                    self._humerus, # parent frame
                                    osim.Vec3(0,-self._length,0), # location
                                    osim.Vec3(0,0,0), # rotation
                                    self._radius, # child frame
                                    osim.Vec3(0,self._length,0), # location
                                    osim.Vec3(0,0,0)) # rotation   

        # Marker: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Marker.html
        # markers
        #self._elbow_marker = osim.Marker("elbow_maker", #name
        #                                self._humerus, # parent frame
        #                                osim.Vec3(0,-self._length,0)) # location
        self._wrist_marker = osim.Marker("wrist_maker", #name
                                        self._radius, # parent frame
                                        osim.Vec3(0,-self._length,0)) # location

        # Muscle: [add miller muscles]
        # Millard muscle: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Millard2012EquilibriumMuscle.html
        # biceps muscle
        self._biceps = osim.Millard2012EquilibriumMuscle("biceps", # name
                                                800, # max isometric force
                                                0.12, # optimal fiber length
                                                0.3, # tendon slack length
                                                0.0) # pennation angle


        self._biceps.addNewPathPoint("origin", # name
                            self._humerus, # body
                            osim.Vec3(self._width,0.0,0)) # position in body

        self._biceps.addNewPathPoint("insertion", # name
                            self._radius, # body
                            osim.Vec3(self._width,0.3,0)) # position in body 

        # triceps muscle
        self._triceps = osim.Millard2012EquilibriumMuscle("triceps", # name
                                                1, # max isometric force
                                                0.12, # optimal fiber length
                                                0.3, # tendon slack length
                                                0.0) # pennation angle        

        self._triceps.addNewPathPoint("origin", # name
                            self._humerus, # body
                            osim.Vec3(-self._width,0.0,0)) # position in body

        self._triceps.addNewPathPoint("insertion", # name
                            self._radius, # body
                            osim.Vec3(-self._width,0.4,0)) # position in body                             

        # PrescribedController: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1PrescribedController.html
        # specify function to excitate the muscle
        self._brain=osim.PrescribedController() # basic control template
        self._brain.addActuator(self._biceps) # control for this muscle
        self._brain.prescribeControlForActuator("biceps", # actuator's name
                                                osim.Constant(1.0)) # actuator's control function

        self._brain.addActuator(self._triceps) # control for this muscle
        self._brain.prescribeControlForActuator("triceps", # actuator's name
                                                osim.Constant(1.0)) # actuator's control function
        # build model with bodies and joints
        self._model.addBody(self._humerus)
        self._model.addJoint(self._shoulder_joint)
        self._model.addBody(self._radius)
        self._model.addJoint(self._elbow_joint)
        #self._model.addMarker(self._elbow_marker)
        self._model.addMarker(self._wrist_marker)
        self._model.addForce(self._biceps)
        self._model.addForce(self._triceps)
        self._model.addController(self._brain)        
        
        # handles
        self._func_handle = self._brain.get_ControlFunctions()

        # joint limits and default values
        self._shoulder_joint.updCoordinate().setRangeMax(_MAX_SHOULDER['pos'])
        self._shoulder_joint.updCoordinate().setRangeMin(_MIN_SHOULDER['pos'])
        self._shoulder_joint.updCoordinate().setDefaultSpeedValue(0.0)
        self._shoulder_joint.updCoordinate().setDefaultClamped(True)
        

        self._elbow_joint.updCoordinate().setRangeMax(_MAX_ELBOW['pos'])
        self._elbow_joint.updCoordinate().setRangeMin(_MIN_ELBOW['pos'])
        self._elbow_joint.updCoordinate().setDefaultSpeedValue(0.0)
        self._elbow_joint.updCoordinate().setDefaultClamped(True)

        # initialize the model and check model consistency (important!)
        self._state = self._model.initSystem()

        # observation directory
        self.max_obs = {}
        # joint 
        self.max_obs['shoulder_pos']=_MAX_SHOULDER['pos']
        self.max_obs['elbow_pos']=_MAX_ELBOW['pos']
        self.max_obs['shoulder_vel']=_MAX_SHOULDER['vel']
        self.max_obs['elbow_vel']=_MAX_ELBOW['vel']        
        # muscle
        self.max_obs['fiber_activation_biceps']=_MAX_FIBER_ACTIVATION
        self.max_obs['fiber_activation_triceps']=_MAX_FIBER_ACTIVATION
        # marker
        self.max_obs['wrist_marker_x']=_MAX_WRIST_POS['x']
        self.max_obs['wrist_marker_y']=_MAX_WRIST_POS['y']        

        self.min_obs = {}
        # joint 
        self.min_obs['shoulder_pos']=_MIN_SHOULDER['pos']
        self.min_obs['elbow_pos']=_MIN_ELBOW['pos']
        self.min_obs['shoulder_vel']=_MIN_SHOULDER['vel']
        self.min_obs['elbow_vel']=_MIN_ELBOW['vel']        
        # muscle
        self.min_obs['fiber_activation_biceps']=_MIN_FIBER_ACTIVATION
        self.min_obs['fiber_activation_triceps']=_MIN_FIBER_ACTIVATION
        # marker
        self.min_obs['wrist_marker_x']=_MIN_WRIST_POS['x']
        self.min_obs['wrist_marker_y']=_MIN_WRIST_POS['y'] 

        # observation space
        high = [var for var in self.max_obs.values()]
        n_inputs = len(high)
        high = np.array(high, dtype=np.float32)

        low = [var for var in self.min_obs.values()]
        low = np.array(low, dtype=np.float32)     

        self.observation_space = spaces.Box(low=low, \
                                            high=high, \
                                            shape=(n_inputs,))       

        # action_space
        self.action_space = spaces.Box(low=np.zeros((1,), dtype=np.float32), \
                                        high=np.ones((1,), dtype=np.float32), \
                                        shape=(1,))       

        self._model.printToXML("2DArm.osim")             

    def reset_manager(self):
        self._manager = osim.Manager(self._model)
        self._manager.setIntegratorAccuracy(self._int_acc)
        self._manager.initialize(self._state)

    def reset(self, init_pos=None):
        # initial position
        if init_pos is not None:
            self._shoulder_joint.updCoordinate().setDefaultValue(init_pos["shoulder"])
            self._elbow_joint.updCoordinate().setDefaultValue(init_pos["elbow"])
            
        # compute initial state
        self._state = self._model.initializeState()
        self._shoulder_joint.updCoordinate().setLocked(self._state, True)
        # compute length of fibers based on the state
        self._model.equilibrateMuscles(self._state)        
        # simulation parameters
        self._sim_timesteps = 0
        self._state.setTime(self._sim_timesteps)
        # forward dynamics manager
        self.reset_manager()
    
    def integrate(self):
        # update simulation step
        self._sim_timesteps = self._sim_timesteps + 1

        # compute next dynamic configuration
        self._state = self._manager.integrate(self._step_size*self._sim_timesteps)        

    def actuate(self, action):
        if np.any(np.isnan(action)): # safety condition
            action = np.nan_to_num(action)
            action = np.clip(action, 0, 1)
            #print_warning(f"nan action maps to [0 1]")
        
        for idx in range(self._func_handle.getSize()):
            # get control function of actuator "idx"
            func = osim.Constant.safeDownCast(self._func_handle.get(idx)) 
            # apply action
            func.setValue(action[idx])


    def step(self, action, obs_as_dict=False, obs_normalized=False):   
        self.actuate(action)
        self.integrate()

        if obs_normalized:
            return self.normalize_observations(obs=self.get_observations(), obs_as_dict=obs_as_dict)

        return self.get_observations()

    def get_observations(self):
        #self._model.realizeAcceleration(self._state)
        # general dictionary
        obs = {}

        # dictionary for joint data
        obs['shoulder_pos'] = self._shoulder_joint.getCoordinate().getValue(self._state)
        obs['elbow_pos'] = self._elbow_joint.getCoordinate().getValue(self._state)  
        obs['shoulder_vel'] = self._shoulder_joint.getCoordinate().getSpeedValue(self._state)
        obs['elbow_vel'] = self._elbow_joint.getCoordinate().getSpeedValue(self._state)          

        # dictionary for muscle data
        obs['fiber_activation_biceps']=self._biceps.getActivation(s=self._state)
        obs['fiber_activation_triceps']=self._triceps.getActivation(s=self._state)
        
        # dictionary for marker data
        obs['marker_wrist_x'] = self._wrist_marker.getLocationInGround(state=self._state)[0]
        obs['marker_wrist_y'] = self._wrist_marker.getLocationInGround(state=self._state)[1]

        return obs

    def normalize_observations(self, obs, obs_as_dict=False):
        # joint pos and vel
        # muscle activation
        # marker pos
        # desired pos

        flatten_obs = [var for var in obs.values()] 
        normalized_obs = (flatten_obs-self.observation_space.low)/(self.observation_space.high-self.observation_space.low)
        
        if obs_as_dict:
            return dict(zip(obs.keys(), normalized_obs))
        
        return normalized_obs


# observations 
# the muscles has high nonlinear response
# smooth inputs are not enough to avoid overshoot
# maybe "on/off" inputs to avoid overshoot


# create model arm
sim_time = 15
arm_model = ArmModel2D(visualize=True, sim_time=sim_time, step_size=1e-2)

arm_model.reset(init_pos={"shoulder":np.deg2rad(0), "elbow":np.deg2rad(0)})

while arm_model._sim_timesteps*arm_model._step_size < sim_time:
    #action += 1*arm_model._sim_timesteps/(sim_time/arm_model._step_size)
    action=np.array([0.0, 0.0])
    obs = arm_model.step(action=action, obs_as_dict=True, obs_normalized=True)
    if arm_model._sim_timesteps%100 ==0:
        print(f"\n=============")
        print(f"step: {arm_model._sim_timesteps}")
        #print(f"min: {arm_model.}")
        #for var in obs:
        #    print(f"{var}")
        for key, value in obs.items(): 
            print(f"{key}: {value:.2f}")

