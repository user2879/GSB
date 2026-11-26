from setuptools import find_packages, setup


setup(
    name="affinity-scheduler",
    version="1.0.0",
    description="Deterministic Kubernetes affinity scheduling and multi-version ledger",
    packages=find_packages(include=["affinity_scheduler", "affinity_scheduler.*"]),
    python_requires=">=3.11",
    install_requires=["numpy==1.26.4"],
    entry_points={
        "console_scripts": ["affinity-scheduler=affinity_scheduler.cli:main"]
    },
)
