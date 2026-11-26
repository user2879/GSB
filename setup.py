"""Legacy-compatible setup entry point for offline setuptools installations."""

from setuptools import find_packages, setup


setup(
    name="gsb-ancient-page",
    version="0.1.0",
    description="Deterministic fixed-point ancient-page digitization demo",
    python_requires=">=3.11",
    packages=find_packages(include=["gsb", "gsb.*"]),
    install_requires=["numpy>=1.24", "Pillow>=9.0", "PyYAML>=6.0"],
    extras_require={"dev": ["pytest>=8.0"]},
    entry_points={"console_scripts": ["gsb=gsb.__main__:main"]},
)
