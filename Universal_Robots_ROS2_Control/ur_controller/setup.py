import os
from setuptools import find_packages, setup
from glob import glob

package_name = 'ur_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tien',
    maintainer_email='49854272+Tien-Jirattanun@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'force_motion_controller = ur_controller.force_motion_controller:main',
            'inverse_kinematics = ur_controller.inverse_kinematics:main'
        ],
    },
)
