from setuptools import find_packages, setup


setup(
    name="ins-diag",
    version="0.1.0",
    description="Deterministic inertial-sensor window clustering and density diagnostics",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["numpy>=1.24", "PyYAML>=6.0"],
    entry_points={"console_scripts": ["ins-diag=ins_diag.cli:main"]},
)

