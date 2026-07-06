import casadi as ca 
import do_mpc
import numpy as np

from sim.plant import M,act,izz, m

def build_model():
    model = do_mpc.model.Model("continuous")
    px = model.set_variable("_x", "px")
    py = model.set_variable("_x","py")
    theta = model.set_variable("_x", "theta")
    vx = model.set_variable("_x", "vx")
    vy = model.set_variable("_x", "vy")
    omega = model.set_variable("_x", "omega")
    u = model.set_variable("_u", "wheel_forces", (4,1)) # _u = decision input of four wheel forces
    x_ref = model.set_variable("_tvp", "x_ref") #tvp = time varying param
    y_ref = model.set_variable("_tvp", "y_ref")
    th_ref = model.set_variable("_tvp", "th_ref")
    wrench = ca.DM(M) @ u
    model.set_rhs("px", vx*ca.cos(theta) - vy*ca.sin(theta))
    model.set_rhs("py", vx*ca.sin(theta) + vy*ca.cos(theta))
    model.set_rhs("theta", omega)
    model.set_rhs("vx", wrench[0]/m+vy*omega)
    model.set_rhs("vy", wrench[1]/m-vx*omega)
    model.set_rhs("omega", wrench[2]/izz)
    model.setup()
    return model

class MpcController:
    def __init__(self, ref_fn, dt, horizon = 20, mpc_dt = 0.1, w_pos = 1.0, w_th = 0.5, w_du = 1e-3):
        self.ref_fn = ref_fn
        self.dt = dt
        self.mpc_dt = mpc_dt
        self.horizon = horizon
        self.resolve_every = max(1, round(mpc_dt / dt))
        model = build_model()
        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon = horizon,
            t_step = mpc_dt,
            store_full_solution = False,
            nlpsol_opts = {"ipopt.print_level" : 0, "print_time" : 0},
        )
        lterm = (
            w_pos * ((model.x["px"] - model.tvp["x_ref"]) ** 2 + (model.x["py"] - model.tvp["y_ref"]) ** 2)
            + w_th * 2.0 * (1.0 - ca.cos(model.x["theta"] - model.tvp["th_ref"]))
        )
        mpc.set_objective(lterm=lterm,mterm=lterm)
        mpc.set_rterm(wheel_forces=w_du)
        f_max = act["wheel_force_max_n"]
        mpc.bounds["lower", "_u", "wheel_forces"] = -f_max
        mpc.bounds["upper", "_u", "wheel_forces"] = f_max
        self.tvp_template = mpc.get_tvp_template()
        mpc.set_tvp_fun(self._tvp_fun)
        mpc.setup()
        self.mpc = mpc
        self.reset()

    def _tvp_fun(self, t_now):
        times = float(np.squeeze(t_now)) + self.mpc_dt * np.arange(self.horizon + 1)
        ref = self.ref_fn(times)
        for i in range(self.horizon + 1):
            self.tvp_template["_tvp", i, "x_ref"] = ref[i, 0]
            self.tvp_template["_tvp", i, "y_ref"] = ref[i, 1]
            self.tvp_template["_tvp", i, "th_ref"] = ref[i, 2]
        return self.tvp_template
    
    def reset(self):
        self.step_count = 0
        self.last_wrench = np.zeros(3)
        self.mpc.reset_history()
        self.initialized = False

    def compute(self, state, ref):
        if self.step_count % self.resolve_every == 0:
            x0 = np.asarray(state, dtype = float).reshape(-1,1)
            if not self.initialized:
                self.mpc.x0 = x0
                self.mpc.set_initial_guess()
                self.initialized = True
            u = np.asarray(self.mpc.make_step(x0)).flatten()
            self.last_wrench = M @ u
        self.step_count += 1
        return self.last_wrench

