from pathlib import Path

import numpy as np
import yaml

_config = Path(__file__).resolve().parents[1] / "config" / "robot_params.yaml"
params = yaml.safe_load(_config.read_text())
robot = params["robot"]
act =params["actuator"]
fric = params["friction"]

m = robot["mass_kg"]
izz = robot["izz_kgm2"]
k = robot["half_wheelbase_m"] + robot["half_track_m"]

M = np.array([[1.0, 1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, -1.0], [-k, k, -k, k],])
M_pinv = np.linalg.pinv(M)

def wrench_to_wheel_forces(wrench):
    return M_pinv @ wrench

def wheel_forces_to_wrench(forces):
    return M @ forces

def apply_actuator_limits(forces, f_max):
    return np.clip(forces, -f_max, f_max)

def apply_friction(forces, mu_s, mu_k, normal_force):
    grip = mu_s * normal_force
    slipping = np.abs(forces) > grip
    capped = forces.copy()
    capped[slipping] = np.sign(forces[slipping]) * mu_k * normal_force
    return capped

def dynamics(state, wrench):
    _, _, theta, vx, vy, omega = state
    fx, fy, mz = wrench
    return np.array([
        vx * np.cos(theta) - vy * np.sin(theta),
        vx * np.sin(theta) + vy * np.cos(theta),
        omega,
        fx / m + vy * omega,
        fy / m - vx * omega,
        mz / izz,
    ])

def rk4_step(state, wrench, dt):
    k1 = dynamics(state, wrench)
    k2 = dynamics(state + 0.5 * dt * k1, wrench)
    k3 = dynamics(state + 0.5 * dt * k2, wrench)
    k4 = dynamics(state + dt * k3, wrench)
    return state + (dt/ 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def plant_step(state, wrench_cmd, dt):
    forces = wrench_to_wheel_forces(np.asarray(wrench_cmd, dtype=float))
    forces = apply_actuator_limits(forces, act["wheel_force_max_n"])
    forces = apply_friction(forces, fric["mu_static"], fric["mu_kinetic"], m*9.81/4)
    wrench = wheel_forces_to_wrench(forces)
    return rk4_step(state, wrench, dt), wrench