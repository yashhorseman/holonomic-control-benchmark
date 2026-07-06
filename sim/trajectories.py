import numpy as np 

def circle_ref(t, radius = 2.0, period = 20.0):
    w = 2*np.pi/period
    x = radius*np.cos(w*t)
    y = radius*np.sin(w*t)
    heading = np.arctan2(np.gradient(y,t), np.gradient(x,t))
    return np.column_stack([x, y, heading])

def figure8_ref(t, scale = 2.0, period = 30.0):
    w = 2*np.pi/period
    x = scale * np.sin(w*t)
    y =0.5*scale*np.sin(2*w*t)
    heading = np.arctan2(np.gradient(y,t), np.gradient(x,t))
    return np.column_stack([x, y, heading])

def waypoint_ref(t, points=((2.0, 0.0),(2.0, 2.0), (0.0, 2.0), (0.0, 0.0)), hold = 5.0):
    pts = np.asarray(points)
    idx = np.minimum((t/hold).astype(int), len(pts) - 1)
    return np.column_stack([pts[idx], np.zeros(len(t))])

