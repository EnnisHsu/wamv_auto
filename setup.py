from setuptools import find_packages, setup

package_name = 'wamv_auto'

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
    maintainer='zenon',
    maintainer_email='zxudx@connect.ust.hk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wamv_auto_sys.py = wamv_auto.wamv_auto_sys:main'
            # 'wamv_nav2_sys.py = wamv_auto.wamv_nav2_sys:main'
        ],
    },
)
