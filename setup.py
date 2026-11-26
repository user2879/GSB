from setuptools import find_packages, setup

setup(
    name="mdet-render",
    version="0.1.0",
    description="Deterministic fixed-point multimodal condition renderer",
    packages=find_packages(include=["mdet_render", "mdet_render.*"]),
    python_requires=">=3.11",
    install_requires=["numpy>=1.24"],
    entry_points={"console_scripts": ["mdet-render=mdet_render.cli:main"]},
)
