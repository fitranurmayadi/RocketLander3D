from setuptools import setup, find_packages

setup(
    name="rocket_lander",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "pybullet",
        "numpy"
    ],
    package_data={
        "rocket_lander": ["assets/*", "assets/meshes/*"]
    }
)
