from setuptools import find_packages, setup


setup(
    name="gsb-diagnostics",
    version="0.1.0",
    description="Deterministic local shard diagnostics for fixed-point quota watermarks",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"gsb": ["py.typed"]},
    python_requires=">=3.10",
    install_requires=["numpy>=1.24"],
    entry_points={"console_scripts": ["gsb-demo=gsb.cli:main"]},
)
