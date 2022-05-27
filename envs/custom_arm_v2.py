import os
import numpy as np
from gym import spaces
from .custom_osim_model import CustomOsimModel
from rl_utils import print_warning
import opensim

# joint range
_MAX_SHOULDER_POS = np.deg2rad(180)
_MIN_SHOULDER_POS = np.deg2rad(-90)

_MAX_ELBOW_POS = np.deg2rad(130)
_MIN_ELBOW_POS = np.deg2rad(0)


class CustomArmEnv():

    def __init__(self, max_sim_time=300, fixed_target=True, visualize=False, integrator_accuracy = 5e-5, step_size=0.01):
        # description of arm model
        model_path = os.path.join(os.path.dirname(__file__), '../models/arm2dof6musc.osim')    
        # create osim model
        self.osim_model = CustomOsimModel(model_path, visualize, integrator_accuracy, step_size)

        blockos = opensim.Body('target', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,1,0,0,0) )
        self.target_joint = opensim.PlanarJoint('target-joint',
                                  self.osim_model.model.getGround(), # PhysicalFrame
                                  opensim.Vec3(0, 0, 0),
                                  opensim.Vec3(0, 0, 0),
                                  blockos, # PhysicalFrame
                                  opensim.Vec3(0, 0, -0.25),
                                  opensim.Vec3(0, 0, 0))

        geometry = opensim.Ellipsoid(0.02, 0.02, 0.02)
        geometry.setColor(opensim.Green)
        blockos.attachGeometry(geometry)  

        self.osim_model.model.addJoint(self.target_joint)
        self.osim_model.model.addBody(blockos)           
        self.osim_model.model.initSystem()


        # simulation parameters
        self.max_sim_time = max_sim_time
        self.fixed_target = fixed_target
        self.pos_des = np.array([0.1, 0.4])
        self.initial_condition={"pos":np.array([np.pi/10, np.pi/10]), "vel":np.zeros((2,))}

        # RL environment parameters
        self.action_space = spaces.Box(low=np.zeros((self.osim_model.n_inputs,)), \
                                        high=np.ones((self.osim_model.n_inputs,)), \
                                        shape=(self.osim_model.n_inputs,))

        # requires normalization
        self.observation_space = spaces.Box(low=np.zeros((18,)), \
                                        high=np.ones((18,)), \
                                        shape=(18,))


    def get_observations(self, state_dict):
       
        obs = [self.pos_des[0], self.pos_des[1]]

        for joint in ["r_shoulder","r_elbow",]:
            obs += state_dict["joint_pos"][joint]
            #obs += state_dict["joint_vel"][joint]

        for muscle in sorted(state_dict["muscles"].keys()):
            obs += [state_dict["muscles"][muscle]["activation"]]
            obs += [state_dict["muscles"][muscle]["fiber_length"]]
        obs += state_dict["markers"]["r_radius_styloid"]["pos"][:2]

        return obs

    def step(self, action):
        # apply action
        self.osim_model.step(action=action)
        # get model states
        state_dict = self.osim_model.compute_model_states()
        # get environemnt observations
        obs=self.get_observations(state_dict=state_dict)

        # compute distance from wrist to target point
        distance = ((state_dict["markers"]["r_radius_styloid"]["pos"][0]-self.pos_des[0])**2 + (state_dict["markers"]["r_radius_styloid"]["pos"][1]-self.pos_des[1])**2)**0.5
        # compute gaussian reward function
        mean = 0 # desired distance
        std = 0.09 # s*std is equal to maximum distance
        reward = 5*(1/(2*np.pi*std**2)**0.5)*np.exp(-(distance-mean)**2/(2*std**2))



        if np.isnan(obs).any(): # check is there are nan values
            print_warning(f"terminal state for NAN observations")
            print_warning(f"{obs}")
            obs = np.nan_to_num(obs)
            
            return obs, -10, True, {}
        
        if not self.osim_model.sim_steps < self.max_sim_time:
            return obs, reward, True, {}

        if not (_MIN_ELBOW_POS<=state_dict["joint_pos"]["r_elbow"]<=_MAX_ELBOW_POS) or not (_MIN_SHOULDER_POS<=state_dict["joint_pos"]["r_shoulder"]<=_MAX_SHOULDER_POS):
            #print_warning(f"terminal state for weird position")
            return obs, -1, False, {}


        return obs, reward, False, {}

    def reset(self):
        # reset model variables
        self.osim_model.reset(initial_condition=self.initial_condition)      
        # get model states
        state_dict = self.osim_model.compute_model_states()
        # get environemnt observations
        obs=self.get_observations(state_dict=state_dict)



        state = opensim.State(self.osim_model.state)

        self.target_joint.getCoordinate(1).setValue(state, self.pos_des[0], False)

        self.target_joint.getCoordinate(2).setLocked(state, False)
        self.target_joint.getCoordinate(2).setValue(state, self.pos_des[1], False)
        self.target_joint.getCoordinate(2).setLocked(state, True)
        self.osim_model.set_state(state)

        return obs


        


