from setuptools import setup

package_name = 'wheel_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fini',
    maintainer_email='fini@kakao.com',
    description='Navigation with deep reinforcement learning',
    license='GNU General Public License v2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim0 = wheel_navigation.sim0:main',
            'sim1 = wheel_navigation.sim1:main',
            'learn0 = wheel_navigation.learn0:main',            
            'learn1 = wheel_navigation.learn1:main',
        ],
    },
)
