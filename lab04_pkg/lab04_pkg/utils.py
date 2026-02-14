import math
import numpy as np
from math import sin,cos
import matplotlib.pyplot as plt
from scipy.stats import norm
def sample_velocity_motion_model(x, u, a, dt):
    """Sample velocity motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- velocity reading obtained from the robot [v, w]
    sigma -- noise parameters of the motion model [a1, a2, a3, a4, a5, a6] or [std_dev_v, std_dev_w]
    dt -- time interval of prediction
    """

    if x is list:
        x = np.array(x)

    if x.ndim == 1:  # manage the case of a single pose
        x = x.reshape(1, -1)

    if u is list:
        u = np.array(u)

    sigma = np.ones((3))
    if a.shape == u.shape:
        sigma[:-1] = a[:]
        sigma[-1] = a[1] * 0.5
    else:
        sigma[0] = a[0] * u[0] ** 2 + a[1] * u[1] ** 2
        sigma[1] = a[2] * u[0] ** 2 + a[3] * u[1] ** 2
        sigma[2] = a[4] * u[0] ** 2 + a[5] * u[1] ** 2

    v_hat = np.ones(x.shape[0]) * u[0] + np.random.normal(0, sigma[0], x.shape[0])
    w_hat = np.ones(x.shape[0]) * u[1] + np.random.normal(0, sigma[1], x.shape[0])
    gamma_hat = np.random.normal(0, sigma[2], x.shape[0])

    r = v_hat / w_hat

    x_prime = x[:, 0] - r * np.sin(x[:, 2]) + r * np.sin(x[:, 2] + w_hat * dt)
    y_prime = x[:, 1] + r * np.cos(x[:, 2]) - r * np.cos(x[:, 2] + w_hat * dt)
    theta_prime = x[:, 2] + w_hat * dt + gamma_hat * dt
    return np.squeeze(np.stack([x_prime, y_prime, theta_prime], axis=-1))

def velocity_mm_Gt(x, u, dt):
    """
    Evaluate Jacobian Gt w.r.t state x=[x, y, theta]
    """
    theta = x[2]
    v, w = u[0], u[1]
    r = v / w
    Gt = np.array(
        [
            [1, 0, -r * cos(theta) + r * cos(theta + w * dt)],
            [0, 1, -r * sin(theta) + r * sin(theta + w * dt)],
            [0, 0, 1],
        ]
    )

    return Gt


def velocity_mm_Vt(x, u, dt):
    """
    Evaluate Jacobian Vt w.r.t command u=[v,w]
    """
    theta = x[2]
    v, w = u[0], u[1]
    r = v / w
    Vt = np.array(
        [
            [
                -sin(theta) / w + sin(theta + w * dt) / w,
                dt * v * cos(theta + w * dt) / w + v * sin(theta) / w**2 - v * sin(theta + w * dt) / w**2,
            ],
            [
                -cos(theta) / w - cos(theta + w * dt) / w,
                dt * v * sin(theta + w * dt) / w - v * cos(theta) / w**2 + v * cos(theta + w * dt) / w**2,
            ],
            [0, dt],
        ]
    )

    return Vt

def evaluate_sampling_dist(mu, sigma, n_samples, sample_function):

    n_bins = 100
    samples = []

    for i in range(n_samples):
        samples.append(sample_function(mu, sigma))

    print("%30s : mean = %.3f, std_dev = %.3f" % ("Normal", np.mean(samples), np.std(samples)))

    count, bins, ignored = plt.hist(samples, n_bins)
    plt.plot(bins, norm(mu, sigma).pdf(bins), linewidth=2, color='r')
    plt.xlim([mu - 5*sigma, mu + 5*sigma])
    plt.title("Normal distribution of samples")
    plt.grid()
    plt.savefig("gaussian_dist.pdf")
    plt.show()

def landmark_range_bearing_model(robot_pose, landmark, sigma):
    """""
    Sampling z from landmark model for range and bearing
    robot pose: can be the estimated robot pose or the particles
    """ ""
    if robot_pose is list:
        robot_pose = np.array(robot_pose)

    if robot_pose.ndim == 1:  # manage the case of a single pose
        robot_pose = robot_pose.reshape(1, -1)

    r_ = np.linalg.norm(robot_pose[:, 0:2] - landmark, axis=1) + np.random.normal(0.0, sigma[0], robot_pose.shape[0])
    phi_ = (
        np.arctan2(landmark[1] - robot_pose[:, 1], landmark[0] - robot_pose[:, 0])
        - robot_pose[:, 2]
        + np.random.normal(0.0, sigma[1], robot_pose.shape[0])
    )
    return np.squeeze(np.stack([r_, phi_], axis=-1))

def residual(a, b, **kwargs):
    """
    Compute the residual between expected and sensor measurements, normalizing angles between [-pi, pi)
    If passed, angle_indx should indicate the positional index of the angle in the measurement arrays a and b

    Returns:
        y [np.array] : the residual between the two states
    """
    y = a - b

    if 'angle_idx' in kwargs:
        angle_idx = kwargs["angle_idx"]
        theta = y[angle_idx]
        y[angle_idx] = normalize_angle(theta)
        
    return y

def normalize_angle(theta):
    """
    Normalize angles between [-pi, pi)
    """
    theta = theta % (2 * np.pi)  # force in range [0, 2 pi)
    if theta > np.pi:  # move to [-pi, pi)
        theta -= 2 * np.pi
    
    return theta
