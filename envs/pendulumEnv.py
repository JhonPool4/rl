#import gym
from gym import spaces
#from gym.utils import seeding
import numpy as np
from os import path
from random import uniform


class PendulumEnv():
    def __init__(self, render_flag=False, sim_time=3, step_size=1e-2):
        self.max_speed = 8.0
        self.max_torque = 20.0
        self.m = 1.0
        self.l = 1.0
        self.g = 9.81
        self.viewer = None
        self.th_des = np.deg2rad(145)

        high = np.array([1.0, 1.0, self.max_speed, 1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.max_sim_timesteps = int(sim_time/step_size)
        self.sim_timesteps = 0
        self.step_size = step_size  
        self.render_flag = render_flag


    def gaussian_reward(self, max_error, x, mu=0):
        sigma = max_error/3
        return np.exp(-(x-mu)**2/(2*sigma**2))


    def step(self, u):
        # get angular position and velocity
        th, thdot = self.state  # th := theta
        # limit control signal
        u = np.clip(u, -self.max_torque, self.max_torque)#[0]
        self.last_u = u
        # reward system
        reward = 5*self.gaussian_reward(max_error=np.deg2rad(20), x=th, mu=self.th_des) + \
                 2*self.gaussian_reward(max_error=np.deg2rad(self.max_speed), x=thdot, mu=0) + \
                 2*self.gaussian_reward(max_error=np.deg2rad(self.max_torque), x=u, mu=0)  


        # compute new states
        newthdot = thdot + (-3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l ** 2) * u) * self.step_size
        newth = th + newthdot * self.step_size

        self.state = np.array([newth, newthdot])

        # update sim timesteps
        self.sim_timesteps += 1
        # terminal conditions
        done = bool (self.sim_timesteps >= self.max_sim_timesteps \
                    or thdot > self.max_speed \
                    or thdot < -self.max_speed)

        if self.render_flag:
            self.render()          
            if done:
                self.close()            

        return self._get_obs(), reward,  done, {'sim_timesteps':self.sim_timesteps, 'max_sim_timesteps': self.max_sim_timesteps}

    def reset(self):
        self.sim_timesteps = 0
        self.state = [uniform(-np.pi, np.pi), 0]
        self.th_des = uniform(self.state[0]-np.deg2rad(20), self.state[0]+np.deg2rad(20))
        self.last_u = None
        if self.render_flag:
            self.render()        
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta) , thetadot/self.max_speed, np.cos(self.th_des), np.sin(self.th_des)], dtype=np.float32)
        #return np.array([np.cos(theta), np.sin(theta) , thetadot, theta_des], dtype=np.float32)        

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] - np.pi/2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 5, np.abs(self.last_u) / 5)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
