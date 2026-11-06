from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ekf_localization'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sebastiano Comai',
    maintainer_email='sebastiano.comai@outlook.it',
    description='EKF localization with landmarks, wheel odometry and IMU fusion.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ekf_node = ekf_localization.ekf_node:main',
            'ekf_fusion_node = ekf_localization.ekf_fusion_node:main',
            'ekf_demo_recorder = ekf_localization.demo_recorder:main',
        ],
    },
)
