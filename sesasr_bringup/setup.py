from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'sesasr_bringup'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sebastiano Comai',
    maintainer_email='sebastiano.comai@outlook.it',
    description='Starts the Gazebo simulation shared by the other packages.',
    license='Apache-2.0',
    tests_require=['pytest'],
)
