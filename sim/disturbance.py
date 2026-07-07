import time
from dataclasses import dataclass
import numpy as np
from sim.plant import(
    M,
    act,
    fric,
    izz as IZZ,
    m as MASS,
    wrench_to_wheel_forces,
    wheel_forces_to_wrench,
    apply_actuator_limits,
    apply_friction,
)       

@dataclass
class Scenario:
    name : str
    mass : float = MASS
    izz : float = IZZ
    mu_s : float = fric["mu_static"]
    mu_k : float = fric["mu_kinetic"]
    f_max : float = act["wheel_force_max_n"]
    com_x : float = 0.0
    action_noise : float = 0.0
    ext_force : tuple = (0.0, 0.0)
    ext_window : tuple = (0.0, 0.0)

def _dynamics(state, wrench, mass, izz):
    _, _, theta, vx, vy, omega = state
    fx, fy, mz = wrench
    return np.array([
        vx * np.cos(theta) - vy * np.sin(theta),
        vx * np.sin(theta) + vy * np.cos(theta),
        omega,
        fx / mass + vy * omega,
        fy / mass - vx * omega,
        mz / izz,
    ])

def _rk4(state, wrench, dt, mass, izz):
    k1 = _dynamics(state, wrench, mass, izz)
    k2 = _dynamics(state + 0.5 * dt * k1, wrench, mass, izz)
    k3 = _dynamics(state + 0.5 * dt * k2, wrench, mass, izz)
    k4 = _dynamics(state + dt * k3, wrench, mass, izz)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def step_disturbance(state, wrench_cmd, dt, sc, t, rng):
    forces = wrench_to_wheel_forces(np.asarray(wrench_cmd, dtype=float))
    if sc.action_noise > 0.0:
        forces = forces + rng.normal(0.0, sc.action_noise, size=4)
    forces = apply_actuator_limits(forces, sc.f_max)
    forces = apply_friction(forces, sc.mu_s, sc.mu_k, sc.mass * 9.81 / 4)
    wrench = wheel_forces_to_wrench(forces)
    wrench = wrench + np.array([0.0, 0.0, -sc.com_x * wrench[1]])
    if sc.ext_window[0] <= t < sc.ext_window[1]:
        c, s = np.cos(state[2]), np.sin(state[2])
        fwx, fwy = sc.ext_force
        wrench = wrench + np.array([c * fwx + s * fwy, -s * fwx + c * fwy, 0.0])
    return _rk4(state, wrench, dt, sc.mass, sc.izz), wrench


def run_tracking_disturbance(controller, ref_fn, duration, sc, seed=0, dt=0.02):
    n = int(duration / dt)
    t = np.arange(n) * dt
    ref = ref_fn(t)
    rng = np.random.default_rng(seed)
    state = np.zeros(6)
    state[:3] = ref[0]
    states = np.zeros((n, 6))
    wrenches = np.zeros((n, 3))
    compute = np.zeros(n)
    controller.reset()
    for i in range(n):
        tic = time.perf_counter()
        cmd = controller.compute(state, ref[i])
        compute[i] = time.perf_counter() - tic
        state, applied = step_disturbance(state, cmd, dt, sc, t[i], rng)
        states[i] = state
        wrenches[i] = applied
    return {"t": t, "ref": ref, "states": states, "wrenches": wrenches, "compute": compute}