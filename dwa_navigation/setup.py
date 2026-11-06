from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'dwa_navigation'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sebastiano Comai',
    maintainer_email='sebastiano.comai@outlook.it',
    description='Dynamic Window Approach local planner for the TurtleBot3.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dwa_node = dwa_navigation.dwa_node:main',
            'demo_recorder = dwa_navigation.demo_recorder:main',
        ],
    },
)
