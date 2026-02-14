import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from math import degrees
from math import sqrt, sin, cos
arrow = u'$\u2191$'

def sample_measurement_landmark_model(x0,z, landmark, sigma_r, sigma_phi):
    """
    Infere the correct robot pose knowing its initial pose, landmark position and measurement z w.r.t. the landmark

    """
    r = z[0]+np.random.normal(0, sigma_r)
    phi=z[1]+np.random.normal(0, sigma_phi)
    theta = x0[2] #+ np.random.normal(0, sigma_theta)
    theta_absolute= phi + theta

    #gamma_hat = np.random.uniform(0, 2*math.pi)
    x = landmark[0] - r * cos(theta_absolute)
    y = landmark[1] - r * sin(theta_absolute)
    #theta_ = gamma_hat - math.pi - phi_hat

    return np.array([x, y, theta])


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

def measurement_jacobian_H(x,landmark):

    delta_x = (x[0]-landmark[0])
    delta_y=(x[1]-landmark[1])
    q = delta_x**2 + delta_y**2
    r = sqrt(q)

    H = np.array([[delta_x/r , delta_y/r , 0],[-delta_y/q , delta_x/q, -1]])

    return H


def main():

    np.random.seed(61)

    x0 = np.array([2.0,1.0,np.pi/6])
    landmark = np.array([5.0,3.0])
    sigma_r= 0.2
    sigma_phi=0.2

    n_samples=1000

    Ht = measurement_jacobian_H(x0, landmark)
    print(Ht)
    #Forward measurement model
    z_measured= landmark_range_bearing_model(x0, landmark, np.array([sigma_r, sigma_phi]))

    
    #Inverse measurement model with known landmark
    samples_with_landmark=[]
    for i in range(n_samples):
        
        x_sample = sample_measurement_landmark_model(x0, z_measured, landmark, sigma_r, sigma_phi)
        samples_with_landmark.append(x_sample)
    samples_with_landmark = np.array(samples_with_landmark)

    #Statistics
    mean_with_landmark = np.mean(samples_with_landmark, axis=0)
    std_with_landmark = np.std(samples_with_landmark, axis=0)


    print("With Known Landmark:")
    print(f"Mean: {mean_with_landmark}, Std Dev: {std_with_landmark}")



    fig = plt.figure(figsize=(14, 9))
    
    # ========== PLOT 1: Spatial distribution with landmark ==========
    ax1 = plt.subplot(2, 2, 1)
    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x0[2])-90)

    # Plot samples
    for x_ in samples_with_landmark[:n_samples]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_[2])-90)
        ax1.scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='r')
    ax1.scatter(x0[0], x0[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')

    ax1.scatter(landmark[0], landmark[1], 
                c='red', s=250, marker='*',
                edgecolors='black', linewidths=2, zorder=5)
    legend_elements = [
        Line2D([0], [0], marker=arrow, color='b', label='Current pose',
               markerfacecolor='none', markeredgecolor='b', markersize=15, linestyle='None'),
        Line2D([0], [0], marker=arrow, color='w', label='Samples',
               markerfacecolor='none', markeredgecolor='red', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='*', color='w', label='Landmark',
               markerfacecolor='red', markeredgecolor='black', markersize=15, linestyle='None')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')

    ax1.set_xlabel('x [m]', fontsize=11)
    ax1.set_ylabel('y [m]', fontsize=11)
    ax1.set_title('Sampling with measurement model', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    
    # ========== PLOT 2: Theta distribution with landmark ==========
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(samples_with_landmark[:, 0], bins=30, alpha=0.7, 
             color='blue', edgecolor='black', density=True)
    ax2.axvline(mean_with_landmark[0], color='red', linestyle='--', 
                linewidth=2, label=f'Samples mean={mean_with_landmark[0]:.3f}')
    ax2.axvline(x0[0], color='green', linestyle='--', 
                linewidth=2, label=f'Current x position={x0[0]:.3f}')
    ax2.set_xlabel('x position [m]', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Samples x position distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=9, loc='upper right')

    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(samples_with_landmark[:, 1], bins=30, alpha=0.7, 
             color='blue', edgecolor='black', density=True)
    ax3.axvline(mean_with_landmark[1], color='red', linestyle='--', 
                linewidth=2, label=f'Samples mean={mean_with_landmark[1]:.3f}')
    ax3.axvline(x0[1], color='green', linestyle='--', 
                linewidth=2, label=f'Current y position={x0[1]:.3f}')
    ax3.set_xlabel('y position [m]', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Samples y position distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(fontsize=9, loc='upper right')

    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(samples_with_landmark[:, 2], bins=30, alpha=0.7, 
             color='blue', edgecolor='black', density=True)
    ax4.axvline(mean_with_landmark[2], color='red', linestyle='--', 
                linewidth=2, label=f'Samples mean={mean_with_landmark[2]:.3f}')
    ax4.axvline(x0[2], color='green', linestyle='--', 
                linewidth=2, label=f'Current y position={x0[2]:.3f}')
    ax4.set_xlabel('Absolute angle [rad]', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('Samples absolute orientation', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig('measurement_model_full.png', dpi=300, bbox_inches='tight')
    print("Plot saved: 'measurement_model_full.png'")
    
    plt.show()

    plt.figure(figsize=(10, 7))

    for x_ in samples_with_landmark[:n_samples]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_[2])-90)
        plt.scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='r')
    plt.scatter(x0[0], x0[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')

    plt.scatter(landmark[0], landmark[1], 
                c='red', s=250, marker='*',
                edgecolors='black', linewidths=2, zorder=5)
    legend_elements = [
        Line2D([0], [0], marker=arrow, color='b', label='Current pose',
               markerfacecolor='none', markeredgecolor='b', markersize=15, linestyle='None'),
        Line2D([0], [0], marker=arrow, color='w', label='Samples',
               markerfacecolor='none', markeredgecolor='red', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='*', color='w', label='Landmark',
               markerfacecolor='red', markeredgecolor='black', markersize=15, linestyle='None')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.xlabel('x [m]', fontsize=11)
    plt.ylabel('y [m]', fontsize=11)
    plt.title('Sampling with measurement model', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('measurement_model.png', dpi=300, bbox_inches='tight')
    print("Plot saved: 'measurement_model.png'")
    
    plt.show()



main()
