import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'ubot_eval'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'behavior_trees'), glob('behavior_trees/*.xml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chibueze',
    maintainer_email='praiseorji4@gmail.com',
    description='Navigation experiment harness: episodes, readiness gating, versioned run dirs.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'episode_runner = ubot_eval.episode_runner:main',
            'campaign = ubot_eval.campaign:main',
        ],
    },
)
