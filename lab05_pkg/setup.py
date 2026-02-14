from setuptools import find_packages, setup

package_name = 'lab05_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='edo',
    maintainer_email='edo@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'DWA_controller = lab05_pkg.controller:main',
            'DWA_action = lab05_pkg.controller_action:main',
            'DWA_controller_task1 = lab05_pkg.controller_task1:main',
            'DWA_controller_vecchio = lab05_pkg.controller_vecchio:main',
        ],
    },
)

