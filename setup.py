from setuptools import find_packages, setup

setup(
    name="driftbench",
    version="1.0.0",
    description="Offline recommender metric drift and A/B definition consistency diagnostics",
    python_requires=">=3.11",
    packages=find_packages(include=["driftbench", "driftbench.*"]),
    install_requires=["numpy>=1.24,<2"],
)
