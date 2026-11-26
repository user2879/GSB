from setuptools import find_packages, setup

setup(
    name="reco-shard-diag",
    version="0.1.0",
    description="Deterministic shard delivery diagnostics for recommendation serving",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.11",
    install_requires=["numpy>=1.26"],
    entry_points={"console_scripts": ["reco-shard-diag=reco_shard_diag.cli:main"]},
)
