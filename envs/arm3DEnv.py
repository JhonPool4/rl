import os
import numpy as np
import opensim as osim
from gym import spaces
from rl_utils import print_warning
from rl_utils.plotter import Plotter
from random import uniform


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
       

class Arm3DEnv(object):
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
        self._model = osim.Model(os.path.join(os.path.dirname(__file__), '../models/Arm3D/completearm.osim') )
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
            return {'x':0.5, 'y':0.5, 'z':0.5}   
        else:
            return {'x':0.5, 'y':0.5, 'z':0.5}   


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
        t0 = self._step_size*self._sim_timesteps
        print(f"t0: {t0}")
        self._sim_timesteps = self._sim_timesteps + 1
        t1 = self._step_size*self._sim_timesteps
        print(f"t1: {t1}")
        print(f"bool: {t1>t0}")
        print(f"")
        #print(f"time: {self._sim_timesteps}")
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
        #if init_pos is not None:
        #    for name in _JOINT_LIST:
        #        self._joints.get(name).upd_coordinates(0).setDefaultValue(init_pos[name])
        
        # compute initial state (important!)
        self._state = self._model.initializeState()
        #if self._visualize and self._show_goal and bullet_pos is not None:
        #    #_joints = self._model.getJointSet() # get joints
        #    #goal_joint = self._model.getJointSet().get("goal_joint")
        #    self.goal_joint.get_coordinates(1).setValue(self._state, bullet_pos['x'], False)
        #    self.goal_joint.get_coordinates(2).setLocked(self._state, False)
        #    self.goal_joint.get_coordinates(2).setValue(self._state, bullet_pos['y'], False)
        #    self.goal_joint.get_coordinates(2).setLocked(self._state, True)

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
        """
        # get observations
        obs = self.get_observations()
        if verbose:
            print(f"goal pos: {obs[0]:.3f}, {obs[1]:.3f}")
            print(f"wrist pos: {obs[6]:.3f}, {obs[7]:.3f}")
            print(f"elbow pos: {obs[8]:.3f}, {obs[9]:.3f}")
        """
        # muscle's activation
        if self._show_act_plot:
            self._mus_plot.reset()
        # get observations and normalize
        #return self.normalize_observations(obs=obs)
        return np.zeros((6,))

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
        """
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
        """
        return np.zeros((6,)), 0, False, {'sim_time':0}
        