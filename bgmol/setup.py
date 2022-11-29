"""
ProjectName
Collection of OpenMM systems
"""
import sys
from setuptools import setup, find_packages

setup(
    # Self-descriptive entries which should always be present
    name='bgmol',
    packages=find_packages(),
    include_package_data=True,
    package_data={'bgmol': ['bgmol/data']}
)

