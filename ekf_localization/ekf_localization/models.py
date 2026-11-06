"""
Motion and measurement models for the EKF.

The nonlinear models are written symbolically with SymPy, differentiated
automatically and converted to NumPy functions once at start-up. All the
returned callables are pure NumPy, so they can be unit tested without ROS.

State layouts:
    3D: [x, y, theta]
    5D: [x, y, theta, v, omega]   (velocities are part of the state)
"""

import numpy as np
import sympy

# Below this angular velocity the arc model is replaced by straight-line motion
OMEGA_THRESHOLD = 1e-6


def _lambdify(args, expr):
    f = sympy.lambdify(args, expr, 'numpy')
    return lambda *a: np.array(f(*a), dtype=float)


def velocity_motion_model(state_dim=3):
    """
    Build the velocity motion model g(mu, u, dt) and its Jacobians G and V.

    With ``state_dim=3`` the velocities come from the control u = [v, omega].
    With ``state_dim=5`` they are read from the state itself (constant velocity
    model) and u only drives the control-noise term V M V^T.

    Returns
    -------
    tuple of callable
        ``(eval_gux, eval_Gt, eval_Vt)``, each with signature ``f(mu, u, dt)``.

    """
    if state_dim not in (3, 5):
        raise ValueError('state_dim must be 3 or 5')

    x, y, theta, v, w, dt = sympy.symbols('x y theta v w dt')
    r = v / w
    arc = [x - r * sympy.sin(theta) + r * sympy.sin(theta + w * dt),
           y + r * sympy.cos(theta) - r * sympy.cos(theta + w * dt),
           theta + w * dt]
    line = [x + v * sympy.cos(theta) * dt,
            y + v * sympy.sin(theta) * dt,
            theta]

    if state_dim == 5:
        arc += [v, w]
        line += [v, w]
        state = [x, y, theta, v, w]
    else:
        state = [x, y, theta]

    args = (x, y, theta, v, w, dt)
    models = {}
    for name, g in (('arc', sympy.Matrix(arc)), ('line', sympy.Matrix(line))):
        models[name] = (
            _lambdify(args, g),
            _lambdify(args, g.jacobian(sympy.Matrix(state))),
            _lambdify(args, g.jacobian(sympy.Matrix([v, w]))),
        )

    def _velocities(mu, u):
        return (mu[3], mu[4]) if state_dim == 5 else (u[0], u[1])

    def _evaluate(index):
        def f(mu, u, dt):
            v_val, w_val = _velocities(mu, u)
            kind = 'line' if abs(w_val) < OMEGA_THRESHOLD else 'arc'
            out = models[kind][index](mu[0], mu[1], mu[2], v_val, w_val, dt)
            return out.flatten() if index == 0 else out
        return f

    return _evaluate(0), _evaluate(1), _evaluate(2)


def landmark_measurement_model(state_dim=3):
    """
    Build the range-bearing landmark model h(mu, m) and its Jacobian H.

    ``m`` is the landmark position; only its first two components [mx, my] are
    used, because turtlebot3_perception publishes the range in the ground plane,
    both from the landmark simulator and from the AprilTag detections.

    Returns
    -------
    tuple of callable
        ``(eval_hx, eval_Ht)``, each with signature ``f(mu, m)``.

    """
    x, y, theta, mx, my = sympy.symbols('x y theta mx my')
    h = sympy.Matrix([
        sympy.sqrt((mx - x) ** 2 + (my - y) ** 2),
        sympy.atan2(my - y, mx - x) - theta,
    ])
    args = (x, y, theta, mx, my)
    eval_h = _lambdify(args, h)
    eval_H = _lambdify(args, h.jacobian(sympy.Matrix([x, y, theta])))

    def eval_hx(mu, m):
        return eval_h(mu[0], mu[1], mu[2], m[0], m[1]).flatten()

    def eval_Ht(mu, m):
        # The landmark does not depend on the velocity states: pad with zeros
        H = np.zeros((2, state_dim))
        H[:, :3] = eval_H(mu[0], mu[1], mu[2], m[0], m[1])
        return H

    return eval_hx, eval_Ht


def odometry_velocity_model():
    """Linear model of a wheel-odometry reading [v, omega] for the 5D state."""
    H = np.zeros((2, 5))
    H[0, 3] = 1.0
    H[1, 4] = 1.0
    return (lambda mu: H @ mu), (lambda mu: H)


def imu_gyro_model():
    """Linear model of a gyroscope reading [omega] for the 5D state."""
    H = np.zeros((1, 5))
    H[0, 4] = 1.0
    return (lambda mu: H @ mu), (lambda mu: H)
