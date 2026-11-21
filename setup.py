"""Compatibility shim for older pip/setuptools editable installs.

Project metadata and package discovery remain canonical in pyproject.toml.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
