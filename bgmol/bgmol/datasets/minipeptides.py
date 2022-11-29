
import os
import numpy as np

from simtk import unit
# from simtk.openmm.app import HBonds
from simtk.openmm import LangevinIntegrator, Platform

from .base import DataSet
from ..systems.minipeptides import MiniPeptide
from ..tpl.hdf5 import load_hdf5, HDF5TrajectoryFile
from bgmol.util import get_data_file

__all__ = ["AImplicitUnconstrained"]



class AImplicitUnconstrained(DataSet):
    """Capped alanine in implicit solvent without bond constraints.
    1 microsecond samples spaced in 1 ps intervals.
    The dataset contains positions, forces, and energies.
    """
    md5 = "f18b9a9c06f3590f1632ca99161c6553"
    num_frames = 1000000
    size = 461080  # in bytes
    selection = "all"
    openmm_version = "7.5.0"

    def __init__(self, root=get_data_file("../data"), read: bool = False):
        super().__init__(root=root, read=read)
        self._system = MiniPeptide(
            "A",
            solvated=False,
            constraints=None,
        )
        self._temperature = 300.

    @property
    def trajectory_file(self):
        return os.path.join(self.root, "AImplicitUnconstrained/traj0.h5")

    def read(self, n_frames=None, stride=None, atom_indices=None):
        self.trajectory = load_hdf5(self.trajectory_file)
        f = HDF5TrajectoryFile(self.trajectory_file)
        frames = f.read(n_frames=n_frames, stride=stride, atom_indices=atom_indices)
        self._energies = frames.potentialEnergy
        self._forces = frames.forces
        f.close()

    @property
    def integrator(self):
        integrator = LangevinIntegrator(self.temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond)
        return integrator

    @property
    def platform(self):
        platform = Platform.getPlatformByName("CUDA")
        platform.setPropertyDefaultValue("Precision", "mixed")
        return platform
