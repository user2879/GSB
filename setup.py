from setuptools import find_packages, setup


setup(
    name="rope-extrapolation-diagnostics",
    version="1.0.0",
    description="Deterministic RoPE extrapolation diagnostics",
    packages=find_packages(include=["rope_diag", "rope_diag.*"]),
    python_requires=">=3.11",
    install_requires=["numpy>=1.24"],
    entry_points={"console_scripts": ["rope-diag=rope_diag.cli:main"]},
)
