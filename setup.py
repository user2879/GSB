from setuptools import find_packages, setup


setup(
    name="inference-attribution",
    version="0.1.0",
    description="Deterministic local inference batch-attribution diagnostics",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["numpy==1.26.4"],
    entry_points={
        "console_scripts": [
            "inference-attribution=inference_attribution.cli:main",
        ]
    },
)
