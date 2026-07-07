import numpy as np 
from stable_baselines3 import PPO, SAC

from sim.rl import F_MAX, M_MAX, observe

ALGOS = {"ppo": PPO, "sac": SAC}

class RlController:
    def __init__(self, model_path, algo):
        self.model = ALGOS[algo].load(model_path, device = "cpu")
    def reset(self):
        pass
    def compute(self, state, ref):
        obs = observe(state, ref)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.array([action[0]*F_MAX, action[1]*F_MAX, action[2]*M_MAX])