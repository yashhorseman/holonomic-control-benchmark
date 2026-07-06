import time
import numpy as np
from sim.plant import plant_step


def run_tracking(controller, ref_fn, duration, dt=0.02):
    n = int(duration / dt)
    t = np.arange(n) * dt
    ref = ref_fn(t)
    state = np.zeros(6)
    state[:3] = ref[0]
    states = np.zeros((n, 6))
    wrenches = np.zeros((n, 3))
    compute = np.zeros(n)
    for i in range(n):
        tic = time.perf_counter()
        cmd = controller.compute(state, ref[i])
        compute[i] = time.perf_counter() - tic
        state, actual = plant_step(state, cmd, dt)
        states[i] = state
        wrenches[i] = actual
    return {"t": t, "ref": ref, "states": states, "wrenches": wrenches, "compute": compute}


def tracking_metrics(result):
    t = result["t"]
    dt = t[1] - t[0]
    pos_err = np.linalg.norm(result["states"][:, :2] - result["ref"][:, :2], axis=1)
    dtheta = result["states"][:, 2] - result["ref"][:, 2]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    return {
        "pos_rmse": float(np.sqrt(np.mean(pos_err**2))),
        "heading_rmse": float(np.sqrt(np.mean(dtheta**2))),
        "effort": float(np.sum(result["wrenches"] ** 2) * dt),
        "compute_ms_mean": float(np.mean(result["compute"]) * 1e3),
        "compute_ms_max": float(np.max(result["compute"]) * 1e3),
    }


def settling_times(result, tol=0.05):
    t = result["t"]
    dt = t[1] - t[0]
    err = np.linalg.norm(result["states"][:, :2] - result["ref"][:, :2], axis=1)
    ref_xy = result["ref"][:, :2]
    jumps = np.flatnonzero(np.linalg.norm(np.diff(ref_xy, axis=0), axis=1) > 1e-9) + 1
    starts = np.concatenate([[0], jumps])
    ends = np.concatenate([jumps, [len(t)]])
    times = []
    for a, b in zip(starts, ends):
        above = np.flatnonzero(err[a:b] >= tol)
        if len(above) == 0:
            times.append(0.0)
        elif above[-1] == b - a - 1:
            times.append(np.nan)
        else:
            times.append((above[-1] + 1) * dt)
    return times