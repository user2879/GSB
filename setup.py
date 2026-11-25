from setuptools import find_packages, setup


setup(
    name="versioned-lease-arbiter",
    version="1.0.0",
    description="Version-tag driven multi-node lease arbitration middleware",
    packages=find_packages(include=["lease_arbiter", "lease_arbiter.*"]),
    python_requires=">=3.11",
    entry_points={"console_scripts": ["lease-arbiter=lease_arbiter.cli:main"]},
)
