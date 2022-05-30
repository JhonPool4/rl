import numpy as np
import opensim
from rl_utils import print_warning




class CustomOsimModel(object):

    def __init__(self, model_path=None, visualize=False, integrator_accuracy = 5e-5, step_size=0.01):
        # simulation parameters
        self.integrator_accuracy = integrator_accuracy
        self.step_size = step_size
        self.max_forces = []
        # create model from .osim file
        self.model = opensim.Model(model_path)
        # object type control
        self.brain = opensim.PrescribedController()

        # enable the visualizer
        self.model.setUseVisualizer(visualize)
        
        # handle model parameters
        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()

        # Add actuators as constant functions.
        for j in range(self.muscleSet.getSize()):
            func = opensim.Constant(1.0)
            self.brain.addActuator(self.muscleSet.get(j))
            self.brain.prescribeControlForActuator(j, func)

            self.max_forces.append(self.muscleSet.get(j).getMaxIsometricForce())

        # add controllers to model
        self.model.addController(self.brain)
        # initialize the model and check model consistency
        #self.state= self.model.initSystem()
        self.state = None # initial model's state
        self.sim_steps = 0 # number of simulation steps
        self.manager = None 
        self.n_inputs = self.muscleSet.getSize()
        

    def actuate(self, action):
        """
        @info: apply action to model
        """
        if np.any(np.isnan(action)):
            action = np.nan_to_num(action)
            action = np.clip(action, 0, 1)
            print_warning(f"nan action maps to [0 1]")
            
        brain = opensim.PrescribedController.safeDownCast(self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        # apply action
        for j in range(functionSet.getSize()):
            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue( float(action[j]) )

    def compute_model_states(self):
        self.model.realizeAcceleration(self.state)

        res = {} # dictionary with model states

        ## Joints
        res["joint_pos"] = {}
        #res["joint_vel"] = {}
        #res["joint_acc"] = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            res["joint_pos"][name] = [joint.get_coordinates(i).getValue(self.state) for i in range(joint.numCoordinates())]
            #res["joint_vel"][name] = [joint.get_coordinates(i).getSpeedValue(self.state) for i in range(joint.numCoordinates())]
            #res["joint_acc"][name] = [joint.get_coordinates(i).getAccelerationValue(self.state) for i in range(joint.numCoordinates())]

        ## Muscles
        res["muscles"] = {}
        for i in range(self.muscleSet.getSize()):
            muscle = self.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {}
            res["muscles"][name]["activation"] = muscle.getActivation(self.state)
            res["muscles"][name]["fiber_length"] = muscle.getFiberLength(self.state)
            #res["muscles"][name]["fiber_velocity"] = muscle.getFiberVelocity(self.state)
            #res["muscles"][name]["fiber_force"] = muscle.getFiberForce(self.state)
        
        ## Markers
        res["markers"] = {}
        for i in range(self.markerSet.getSize()):
            marker = self.markerSet.get(i)
            name = marker.getName()
            res["markers"][name] = {}
            res["markers"][name]["pos"] = [marker.getLocationInGround(self.state)[i] for i in range(3)]
            #res["markers"][name]["vel"] = [marker.getVelocityInGround(self.state)[i] for i in range(3)]
            #res["markers"][name]["acc"] = [marker.getAccelerationInGround(self.state)[i] for i in range(3)]

        return res

    def reset_manager(self):
        self.manager = opensim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def set_state(self, state):
        self.state = state
        self.sim_steps = int(self.state.getTime() / self.step_size) # TODO: remove istep altogether
        self.reset_manager()

    def reset(self, initial_condition=None):
        self.state = self.model.initializeState()
        if initial_condition is not None:
            # get pre-defined position and velocity
            qpos = self.state.getQ()
            #qvel = self.state.getQDot()
            # new joint position and velocity
            qpos[0] = initial_condition["pos"][0]
            qpos[1] = initial_condition["pos"][1]
            #qvel[0] = initial_condition["vel"][0]
            #qvel[1] = initial_condition["vel"][1]
            # update joint position and velocity
            self.state.setQ(qpos)
            #self.state.setU(qvel)
        
        self.model.equilibrateMuscles(self.state)
        self.state.setTime(0)
        self.sim_steps = 0

        self.reset_manager()

    def integrate(self):
        # update simulation step
        self.sim_steps = self.sim_steps + 1

        # compute next dynamic configuration
        self.state = self.manager.integrate(self.step_size*self.sim_steps)
        """
        # clip observations
        qpos = self.state.getQ() 
        print_warning(f" obs: {np.rad2deg(qpos[0])}, {np.rad2deg(qpos[1])}")
        
        # for normal motion
        if not (_MIN_ELBOW_POS<=qpos[1]<=_MAX_ELBOW_POS) or not (_MIN_SHOULDER_POS<=qpos[0]<=_MAX_SHOULDER_POS): 
            print_warning(f"clipping obs: {qpos}")
            qpos[1] = np.clip(qpos[1], _MIN_ELBOW_POS, _MAX_ELBOW_POS)
            qpos[0] = np.clip(qpos[0], _MIN_SHOULDER_POS, _MAX_SHOULDER_POS)

            self.state.setQ(qpos)
            #self.model.equilibrateMuscles(self.state)
            self.reset_manager()
        
        """

    def step(self, action):   
        self.actuate(action)
        self.integrate()


        