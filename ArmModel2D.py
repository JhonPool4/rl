import opensim as osim
import numpy as np


class ArmModel2D():
    def __init__(self, integrator_accuracy=1e-5, visualize=True, sim_time=3, step_size=0.01):
        # model parameters
        self._global_reference = osim.Vec3(0, 1.5, 0)
        self._mass=1.0
        self._com=osim.Vec3(0,0,0)
        self._inertia=osim.Inertia(0,0,0)
        self._width=0.05
        self._length=0.25
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
        self._elbow_marker = osim.Marker("elbow_maker", #name
                                        self._humerus, # parent frame
                                        osim.Vec3(0,-self._length,0)) # location
        self._wrist_marker = osim.Marker("wrist_maker", #name
                                        self._radius, # parent frame
                                        osim.Vec3(0,-self._length,0)) # location

        # Muscle: [add miller muscles]

        # build model with bodies and joints
        self._model.addBody(self._humerus)
        self._model.addJoint(self._shoulder_joint)
        self._model.addBody(self._radius)
        self._model.addJoint(self._elbow_joint)
        self._model.addMarker(self._wrist_marker)
        self._model.addMarker(self._elbow_marker)
        
        # joint limits and default values
        self._shoulder_joint.updCoordinate().setRangeMax(osim.SimTK_PI)
        self._shoulder_joint.updCoordinate().setRangeMin(-0.5*osim.SimTK_PI)
        self._shoulder_joint.updCoordinate().setDefaultSpeedValue(0.0)
        self._shoulder_joint.updCoordinate().setDefaultClamped(True)

        self._elbow_joint.updCoordinate().setRangeMax(1.5*osim.SimTK_PI)
        self._elbow_joint.updCoordinate().setRangeMin(0.0)
        self._elbow_joint.updCoordinate().setDefaultSpeedValue(0.0)
        self._elbow_joint.updCoordinate().setDefaultClamped(True)

        # initialize the model and check model consistency (important!)
        self._state = self._model.initSystem()

        self._model.printToXML("1DArm.osim")

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
        pass

    def step(self, action):   
        self.actuate(action)
        self.integrate()

        return self.get_observations()

    def get_observations(self):
        # general dictionary
        obs = {}

        # dictionary for joint data
        obs['shoulder_pos'] = self._shoulder_joint.getCoordinate().getValue(self._state)
        obs['elbow_pos'] = self._elbow_joint.getCoordinate().getValue(self._state)  

        # dictionary for marker data
        obs['marker_elbow'] = [self._elbow_marker.getLocationInGround(state=self._state)[i] for i in range(3)] 
        obs['marker_wrist'] = [self._wrist_marker.getLocationInGround(state=self._state)[i] for i in range(3) ]

        return obs




# create model arm
sim_time = 10
arm_model = ArmModel2D(visualize=True, sim_time=sim_time)

arm_model.reset(init_pos={"shoulder":np.deg2rad(0), "elbow":np.deg2rad(0)})

while arm_model._sim_timesteps*arm_model._step_size < sim_time:
    
    obs = arm_model.step(0)
    if arm_model._sim_timesteps%100 ==0:
        print(f"\n=============")
        print(f"step: {arm_model._sim_timesteps}")
        for key, value in obs.items(): 
            print(f"{key}: {value}")

