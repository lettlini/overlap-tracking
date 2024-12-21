from setuptools import find_packages, setup

setup(
    name="overlap-tracking",
    version="1",
    author="Kolya Lettl",
    author_email="kolya.lettl@studserv.uni-leipzig.de",
    description="library for overlap-based object tracking",
    packages=find_packages(),
    install_requires=["numpy", "polars"],
)
