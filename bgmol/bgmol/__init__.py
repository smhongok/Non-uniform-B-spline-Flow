"""
ProjectName
Collection of OpenMM systems
"""

# package-wide paths
import os

SYSTEMPATH = os.path.join(os.path.dirname(os.path.normpath(__file__)), "systems")
DATASETSPATH = os.path.join(os.path.dirname(os.path.normpath(__file__)), "datasets")
DATAPATH = os.path.join(os.path.dirname(os.path.normpath(__file__)), "data")


# Add imports here
from . import datasets
from . import systems


