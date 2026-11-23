from setuptools import find_packages, setup


setup(
    name="gateway-diag",
    version="0.1.0",
    description="Deterministic loopback inference gateway routing diagnostics",
    python_requires=">=3.11",
    packages=find_packages(include=("gateway_diag", "gateway_diag.*")),
    install_requires=("numpy>=1.26,<3",),
    entry_points={"console_scripts": ("gateway-diag=gateway_diag.cli:main",)},
)
