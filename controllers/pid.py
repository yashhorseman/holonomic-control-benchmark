import numpy as np


class PidController:
    def __init__(self, gains, dt, f_limit, m_limit, i_max=5.0):
        self.kp = np.asarray(gains["kp"], dtype=float)
        self.ki = np.asarray(gains["ki"], dtype=float)
        self.kd = np.asarray(gains["kd"], dtype=float)
        self.dt = dt
        self.limits = np.array([f_limit, f_limit, m_limit])
        self.i_max = i_max
        self.reset()

    def reset(self):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)

    def compute(self, state, ref):
        px, py, theta = state[:3]
        ex = ref[0] - px
        ey = ref[1] - py
        etheta = np.arctan2(np.sin(ref[2] - theta), np.cos(ref[2] - theta))
        c, s = np.cos(theta), np.sin(theta)
        error = np.array([c * ex + s * ey, -s * ex + c * ey, etheta])
        self.integral = np.clip(self.integral + error * self.dt, -self.i_max, self.i_max)
        deriv = (error - self.prev_error) / self.dt
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        return np.clip(out, -self.limits, self.limits)     