from setuptools import find_packages, setup

package_name = 'cam_ros_node'

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
    maintainer='sourav',
    maintainer_email='sourav.hawaldar@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cam_ros_node = cam_ros_node.cam_ros_node:main',
            'stereo_pointcloud_node = cam_ros_node.stereo_pointcloud_node:main',
        ],
    },
)
