"""Legacy editable-install compatibility; metadata also lives in pyproject.toml."""

from setuptools import find_packages, setup


setup(
    name="probelease",
    version="0.1.0",
    description=(
        "Deterministic loopback UDP probe leases with pause, resume, replay, "
        "and splice verification"
    ),
    packages=find_packages(include=["probelease", "probelease.*"]),
    python_requires=">=3.11",
    install_requires=["numpy>=1.24"],
    entry_points={"console_scripts": ["probelease=probelease.cli:main"]},
)

