from setuptools import find_packages, setup

setup(
    name="batch-receipt-ledger",
    version="1.0.0",
    description="Deterministic local batch attribution receipt ledger",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.11",
    install_requires=["numpy==1.26.4"],
    entry_points={
        "console_scripts": [
            "batch-ledger=batch_ledger.cli:main",
            "batch-ledger-worker=batch_ledger.worker:main",
        ]
    },
)
