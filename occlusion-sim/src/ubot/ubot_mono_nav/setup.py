import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'ubot_mono_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'behavior_trees'), glob('behavior_trees/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chibueze',
    maintainer_email='praiseorji4@gmail.com',
    description='LiDAR-less navigation for the ubot: RGB -> metric depth -> LaserScan -> nav2.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mono_depth_node = ubot_mono_nav.mono_depth_node:main',
            'depth_to_scan = ubot_mono_nav.depth_to_scan_node:main',
            'scan_watchdog = ubot_mono_nav.scan_watchdog_node:main',
            'oak_rgb_node = ubot_mono_nav.oak_rgb_node:main',
        ],
    },
)
