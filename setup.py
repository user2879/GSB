from setuptools import find_packages, setup


setup(
    name="ckptsim",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.11",
    install_requires=["numpy>=1.26"],
    entry_points={"console_scripts": ["ckptsim=ckptsim.cli:main"]},
)
