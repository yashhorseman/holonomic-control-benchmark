import gymnasium as gym 
import numpy as np
from gymnasium import spaces

from sim.disturbance import Scenario, step_disturbance
from sim.plant import act, k, plant_step
from sim.trajectories import circle_ref, figure8_ref, waypoint_ref

F_MAX = 4 * act["wheel_force_max_n"]
M_MAX = F_MAX*k

def observe(state, ref):
    px, py, theta,  vx, vy, omega = state
    c, s = np.cos(theta), np.sin(theta)
    ex = c*(ref[0] - px) + s*(ref[1] - py)
    ey = -s*(ref[0] - px) + c*(ref[1] - py)
    eth = ref[2] - theta
    return np.array([ex, ey, np.sin(eth), np.cos(eth), vx, vy, omega], dtype=np.float32)

class TrackingEnv(gym.Env):
    def __init__(self, dt = 0.02, episode_time = 20.0):
        self.dt = dt
        self.n_steps = int(episode_time / dt)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype = np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype = np.float32)

    def _make_ref(self):
        t = np.arange(self.n_steps +1)*self.dt
        choice = self.np_random.integers(3)
        offset = self.np_random.uniform(0.0, 20.0)
        if choice==0:
            return circle_ref(t + offset)
        if choice == 1:
            return  figure8_ref(t+offset)
        return waypoint_ref(t, hold=8.0)
    
    def reset(self, seed = None, options= None ):
        super().reset(seed=seed)
        self.ref = self._make_ref()
        self.i = 0
        self.state = np.zeros(6)
        self.state[:3] = self.ref[0]
        return observe(self.state, self.ref[0]), {}
    def step(self, action):
        wrench = np.array([action[0]*F_MAX, action[1]*F_MAX, action[2]*M_MAX])
        self.state, applied = plant_step(self.state, wrench, self.dt)
        self.i+=1
        obs = observe(self.state, self.ref[self.i])
        pos_err2 = obs[0]**2 + obs[1]**2
        eth = np.arctan2(obs[2], obs[3])
        reward = float(-(pos_err2 + 0.5 * eth**2 + 1e-5 * float(np.sum(applied**2))))
        terminated = bool(pos_err2 > 25.0)
        if terminated:
            reward -= 50.0
        truncated = self.i >= self.n_steps
        return obs, reward, terminated, truncated, {}

class TrackingEnvDR(TrackingEnv):
    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)
        r = self.np_random
        mu_s = float(r.uniform(0.09, 0.9))
        self.sc = Scenario(
            name="dr",
            mass=float(r.uniform(70.0, 115.0)),
            izz=float(r.uniform(9.0, 16.0)),
            mu_s=mu_s,
            mu_k=mu_s * float(r.uniform(0.7, 0.95)),
            com_x=float(r.uniform(-0.1, 0.1)),
            action_noise=float(r.uniform(0.0, 3.0)),
        )
        self.dr_rng = np.random.default_rng(int(r.integers(2**31)))
        return obs, info

    def step(self, action):
        wrench = np.array([action[0]*F_MAX, action[1]*F_MAX, action[2]*M_MAX])
        self.state, applied = step_disturbance(self.state, wrench, self.dt, self.sc, self.i * self.dt, self.dr_rng)
        self.i += 1
        obs = observe(self.state, self.ref[self.i])
        pos_err2 = obs[0]**2 + obs[1]**2
        eth = np.arctan2(obs[2], obs[3])
        reward = float(-(pos_err2 + 0.5 * eth**2 + 1e-5 * float(np.sum(applied**2))))
        terminated = bool(pos_err2 > 25.0)
        if terminated:
            reward -= 50.0
        truncated = self.i >= self.n_steps
        return obs, reward, terminated, truncated, {}