import numpy as np
import opensim as osim
from rl_utils import print_warning

class CustomOsimModel(object):
    def __init__(self, model_path=None, visualize=False, integrator_accuracy = 5e-5, step_size=0.01, add_bullet=False):
        # simulation parameters
        self._int_acc = integrator_accuracy
        self._step_size = step_size
        self._add_bullet = add_bullet
        self._visualize = visualize
        # create model from .osim file
        self._model = osim.Model(model_path)
        # enable the visualizer
        self._model.setUseVisualizer(self._visualize)

        # get handles for model parameters
        self._muscles = self._model.getMuscles() # get muscle
        self._joints = self._model.getJointSet() # get joints
        self._markers = self._model.getMarkerSet() # get markers

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
        self._control_functions= self._brain.get_ControlFunctions()
        

        if self._visualize and self._add_bullet:
            # Body: https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Body.html
            # humerus link
            goal_body = osim.Body("goal_body", # name
                                    0.0001, # mass
                                    osim.Vec3(0), # center of mass
                                    osim.Inertia(1,1,1,0,0,0)) # inertia's moment [Ixx Iyy Izz Ixy Ixz Iyz]
            # add display geometry
            _geom = osim.Ellipsoid(0.025,0.025,0.025)
            _geom.setColor(osim.Orange)
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
            
        # initialize the model and check model consistency (important!)
        self._model.initSystem()
        self._state = None  # model's state
        self._manager = None # model's manager
        self._sim_timesteps = 0 # number of simulation steps      
        
    def update_joint_limits(self, joint_list, max_list, min_list):
        for joint_name in joint_list:
            self._joints.get(joint_name).upd_coordinates(0).setRangeMax(max_list[joint_name]['pos'])
            self._joints.get(joint_name).upd_coordinates(0).setRangeMin(min_list[joint_name]['pos'])
            self._joints.get(joint_name).upd_coordinates(0).setDefaultSpeedValue(0.0)
            self._joints.get(joint_name).upd_coordinates(0).setDefaultClamped(True)

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

    def reset_manager(self):
        self._manager = osim.Manager(self._model)
        self._manager.setIntegratorAccuracy(self._int_acc)
        self._manager.initialize(self._state)


    def reset(self, init_pos=None, bullet_pos=None):
        # set initial position
        if init_pos is not None:
            self._joints.get("r_shoulder").upd_coordinates(0).setDefaultValue(init_pos["r_shoulder"])
            self._joints.get("r_elbow").upd_coordinates(0).setDefaultValue(init_pos["r_elbow"])
        
        # compute initial state (important!)
        self._state = self._model.initializeState()
        if self._visualize and self._add_bullet and bullet_pos is not None:
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


    def integrate(self):
        # update simulation step
        self._sim_timesteps = self._sim_timesteps + 1

        # compute next dynamic configuration
        self._state = self._manager.integrate(self._step_size*self._sim_timesteps)        


    def step(self, action):   
        self.actuate(action)
        self.integrate()


        