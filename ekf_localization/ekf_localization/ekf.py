"""Generic Extended Kalman Filter for mobile-robot localization (pure NumPy, no ROS)."""

import numpy as np

from .utils import residual


class RobotEKF:
    """
    Extended Kalman Filter with a pluggable motion model.

    The motion model is given as three callables with signature ``f(mu, u, dt)``:
    ``eval_gux`` returns the predicted state, ``eval_Gt`` the Jacobian w.r.t. the
    state and ``eval_Vt`` the Jacobian w.r.t. the control input. Measurement
    models are passed to :meth:`update` already evaluated, so the same filter can
    fuse heterogeneous sensors (landmarks, wheel odometry, IMU, ...).
    """

    def __init__(self, dim_x, dim_u, eval_gux, eval_Gt, eval_Vt):
        self.mu = np.zeros(dim_x)       # state mean
        self.Sigma = np.eye(dim_x)      # state covariance
        self.Mt = np.eye(dim_u)         # control noise covariance

        self.eval_gux = eval_gux
        self.eval_Gt = eval_Gt
        self.eval_Vt = eval_Vt

        self._I = np.eye(dim_x)

    def predict(self, u, dt):
        """
        Propagate mean and covariance through the motion model.

        Sigma' = G Sigma G^T + V M V^T, with the Jacobians evaluated at the
        previous mean.
        """
        Gt = self.eval_Gt(self.mu, u, dt)
        Vt = self.eval_Vt(self.mu, u, dt)
        self.mu = self.eval_gux(self.mu, u, dt)
        self.Sigma = Gt @ self.Sigma @ Gt.T + Vt @ self.Mt @ Vt.T

    def update(self, z, z_hat, Ht, Qt, angle_idx=None):
        """
        Correct the state with a measurement.

        Parameters
        ----------
        z : np.ndarray
            Actual measurement.
        z_hat : np.ndarray
            Expected measurement h(mu).
        Ht : np.ndarray
            Jacobian of the measurement model evaluated at mu.
        Qt : np.ndarray
            Measurement noise covariance.
        angle_idx : int, optional
            Index of an angular component of z, wrapped to [-pi, pi) in the
            innovation.

        Returns
        -------
        np.ndarray
            The innovation z - z_hat.

        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        z_hat = np.atleast_1d(np.asarray(z_hat, dtype=float))
        Ht = np.atleast_2d(Ht)
        Qt = np.atleast_2d(Qt)

        S = Ht @ self.Sigma @ Ht.T + Qt
        # K = Sigma H^T S^-1, computed with a linear solve (S is symmetric)
        K = np.linalg.solve(S, Ht @ self.Sigma).T

        y = residual(z, z_hat, angle_idx)
        self.mu = self.mu + K @ y

        # Joseph form: numerically more stable than (I - KH) Sigma
        I_KH = self._I - K @ Ht
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ Qt @ K.T
        return y
