"""Compatibility shim for pip versions without PEP 660 editable support."""

from setuptools import find_packages, setup


setup(
    name="timbre-workbench",
    version="1.0.0",
    description="Deterministic speaker-timbre analysis and diagnostics workbench",
    packages=find_packages(include=["timbre_workbench", "timbre_workbench.*"]),
    python_requires=">=3.11",
    install_requires=["numpy>=1.24"],
    entry_points={"console_scripts": ["timbre-workbench=timbre_workbench.cli:main"]},
)

