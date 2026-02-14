from utils import sample_velocity_motion_model, velocity_mm_Gt, velocity_mm_Vt,evaluate_sampling_dist
import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from math import degrees
arrow = u'$\u2191$'

def main():

    np.random.seed(25)

    x0=np.array([0.0,0.0,0.0])
    u=np.array([1.0,0.5])
    dt=1.0
    n_samples=500
    samples_angular = []
    samples_linear = []
    #High linear velocity noise
    a_linear = np.array([0.2, 0.2, 0.02, 0.02, 0.1, 0.1])
    #High angular velocity noise
    a_angular = np.array([0.02, 0.02, 0.2, 0.2, 0.1, 0.1])


    #sampling 
    for i in range(n_samples):
        x_angular = sample_velocity_motion_model(np.array([x0]), u, a_angular, dt)
        x_linear = sample_velocity_motion_model(np.array([x0]), u, a_linear, dt)
        samples_angular.append(x_angular)
        samples_linear.append(x_linear)

    samples_angular = np.array(samples_angular)
    samples_linear = np.array(samples_linear)

    #jacobians
    Gt = velocity_mm_Gt(x0, u, dt)
    Vt = velocity_mm_Vt(x0, u, dt)

    #statistics
    mean_angular = np.mean(samples_angular, axis=0)
    mean_linear = np.mean(samples_linear, axis=0)

    std_angular = np.std(samples_angular, axis=0)
    std_linear = np.std(samples_linear, axis=0)
    print("High Angular Velocity Noise:")
    print(f"Mean: {mean_angular}, Std Dev: {std_angular}")

    print("High Linear Velocity Noise:")
    print(f"Mean: {mean_linear}, Std Dev: {std_linear}")

    '''
    #Evaluate sampling distribution for angular noise
    print("\nEvaluating sampling distribution for high angular velocity noise:")
    for i in range(3):
        print(f"State dimension {i}:")
        evaluate_sampling_dist(mean_angular[i], a_angular[i], n_samples,np.random.normal)

    #Evaluate sampling distribution for linear noise
    print("\nEvaluating sampling distribution for high linear velocity noise:")
    
    for i in range(3):
        print(f"State dimension {i}:")
        evaluate_sampling_dist(mean_linear[i], a_linear[i], n_samples,np.random.normal) '''


    print("\n[PASSO 6] Generazione plot...")

    # Plot code:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x0[2])-90)
    axes[0,0].scatter(x0[0], x0[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')
    for x_ in samples_linear[:n_samples]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_[2])-90)
        axes[0,0].scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='orange')
    axes[0,0].set_xlabel("x-position [m]")
    axes[0,0].set_ylabel("y-position [m]")
    axes[0,0].set_title("Velocity motion model sampling with high linear velocity noise", fontweight='bold')
    legend_elements = [
        Line2D([0], [0], marker=arrow, color='w', label='Current pose',
               markerfacecolor='none', markeredgecolor='b', markersize=15, linestyle='None'),
        Line2D([0], [0], marker=arrow, color='w', label='Samples',
               markerfacecolor='none', markeredgecolor='orange', markersize=10, linestyle='None')
    ]
    axes[0,0].legend(handles=legend_elements, loc='lower right')
    axes[0,1].hist(samples_linear[:, 0], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0,1].axvline(mean_linear[0], color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_linear[0]:.3f}')
    axes[0,1].set_xlabel('x position [m]', fontsize=12)
    axes[0,1].set_ylabel('Frequency', fontsize=12)
    axes[0,1].set_title('y position distribution - High linear uncertainty', fontsize=13, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3, axis='y')
    axes[0,1].legend(fontsize=11)
    axes[1,0].hist(samples_linear[:, 1], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].axvline(mean_linear[1], color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_linear[1]:.3f}')
    axes[1,0].set_xlabel('y position [m]', fontsize=12)
    axes[1,0].set_ylabel('Frequency', fontsize=12)
    axes[1,0].set_title('y position distribution - High linear uncertainty', fontsize=13, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3, axis='y')
    axes[1,0].legend(fontsize=11)
    axes[1,1].hist(samples_linear[:, 2], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1,1].axvline(mean_linear[2], color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_linear[2]:.3f}')
    axes[1,1].set_xlabel('θ [rad]', fontsize=12)
    axes[1,1].set_ylabel('Frequency', fontsize=12)
    axes[1,1].set_title('θ distribution - High linear uncertainty', fontsize=13, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3, axis='y')
    axes[1,1].legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('velocity_motion_high_linear_uncertainty_full.png', dpi=300, bbox_inches='tight')
    print("  ✓ Plot salvato: 'velocity_motion_high_linear_uncertainty_full.png'")
    plt.show()


    plt.figure(figsize=(10,7))
    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x0[2])-90)
    plt.scatter(x0[0], x0[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')
    for x_ in samples_linear[:n_samples]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_[2])-90)
        plt.scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='orange')
    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("Velocity motion model sampling with high linear velocity noise", fontweight='bold')
    legend_elements = [
        Line2D([0], [0], marker=arrow, color='w', label='Current pose',
               markerfacecolor='none', markeredgecolor='b', markersize=15, linestyle='None'),
        Line2D([0], [0], marker=arrow, color='w', label='Samples',
               markerfacecolor='none', markeredgecolor='orange', markersize=10, linestyle='None')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig('velocity_motion_high_linear_uncertainty.png', dpi=300, bbox_inches='tight')
    print("  ✓ Plot salvato: 'velocity_motion_high_linear_uncertainty.png'")
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x0[2])-90)
    axes[0,0].scatter(x0[0], x0[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')
    for x_ in samples_angular[:n_samples]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_[2])-90)
        axes[0,0].scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='g')
    axes[0,0].set_xlabel("x-position [m]")
    axes[0,0].set_ylabel("y-position [m]")
    axes[0,0].set_title("Velocity motion model sampling with high angular velocity noise", fontweight='bold')
    legend_elements = [
        Line2D([0], [0], marker=arrow, color='w', label='Current pose',
               markerfacecolor='none', markeredgecolor='b', markersize=15, linestyle='None'),
        Line2D([0], [0], marker=arrow, color='w', label='Samples',
               markerfacecolor='none', markeredgecolor='g', markersize=10, linestyle='None')
    ]
    axes[0,0].legend(handles=legend_elements, loc='lower right')
    axes[0,1].hist(samples_angular[:, 0], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0,1].axvline(mean_angular[0], color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_angular[0]:.3f}')
    axes[0,1].set_xlabel('x position [m]', fontsize=12)
    axes[0,1].set_ylabel('Frequency', fontsize=12)
    axes[0,1].set_title('x position distribution - High angular uncertainty', fontsize=13, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3, axis='y')
    axes[0,1].legend(fontsize=11)
    axes[1,0].hist(samples_angular[:, 1], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1,0].axvline(mean_angular[1], color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_angular[1]:.3f}')
    axes[1,0].set_xlabel('y position [m]', fontsize=12)
    axes[1,0].set_ylabel('Frequency', fontsize=12)
    axes[1,0].set_title('y position distribution - High angular uncertainty', fontsize=13, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3, axis='y')
    axes[1,0].legend(fontsize=11)
    axes[1,1].hist(samples_angular[:, 2], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1,1].axvline(mean_angular[2], color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_angular[2]:.3f}')
    axes[1,1].set_xlabel('θ [rad]', fontsize=12)
    axes[1,1].set_ylabel('Frequency', fontsize=12)
    axes[1,1].set_title('θ distribution - High angular uncertainty', fontsize=13, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3, axis='y')
    axes[1,1].legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('velocity_motion_high_angular_uncertainty_full.png', dpi=300, bbox_inches='tight')
    print("  ✓ Plot salvato: 'velocity_motion_high_angular_uncertainty_full.png'")
    plt.show()
     

    plt.figure(figsize=(10,7)) 
    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x0[2])-90)
    plt.scatter(x0[0], x0[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')
    for x_ in samples_angular[:n_samples]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_[2])-90)
        plt.scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='g')
    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("Velocity motion model sampling with high angular velocity noise", fontweight='bold')
    legend_elements = [
        Line2D([0], [0], marker=arrow, color='w', label='Current pose',
               markerfacecolor='none', markeredgecolor='b', markersize=15, linestyle='None'),
        Line2D([0], [0], marker=arrow, color='w', label='Samples',
               markerfacecolor='none', markeredgecolor='g', markersize=10, linestyle='None')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig('velocity_motion_high_angular_uncertainty.png', dpi=300, bbox_inches='tight')
    print("  ✓ Plot salvato: 'velocity_motion_high_angular_uncertainty.png'")
    plt.show()

    print(Gt)
    print(Vt)

main()