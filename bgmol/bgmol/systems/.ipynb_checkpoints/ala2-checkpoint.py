import os
import tempfile
import numpy as np
from simtk.openmm import app
import mdtraj as md


def compute_phi_psi(traj):
    """Compute backbone dihedrals.

    Parameters
    ----------
    traj : mdtraj.Trajectory
    """
    phi_atoms = [4, 6, 8, 14]
    phi = md.compute_dihedrals(traj, indices=[phi_atoms])[:, 0]
    psi_atoms = [6, 8, 14, 16]
    psi = md.compute_dihedrals(traj, indices=[psi_atoms])[:, 0]
    return phi, psi


DEFAULT_RIGID_BLOCK = np.array([6, 8, 9, 10, 14])


DEFAULT_Z_MATRIX = np.array([
    [0, 1, 4, 6],
    [1, 4, 6, 8],
    [2, 1, 4, 0],
    [3, 1, 4, 0],
    [4, 6, 8, 14],
    [5, 4, 6, 8],
    [7, 6, 8, 4],
    [11, 10, 8, 6],
    [12, 10, 8, 11],
    [13, 10, 8, 11],
    [15, 14, 8, 16],
    [16, 14, 8, 6],
    [17, 16, 14, 15],
    [18, 16, 14, 8],
    [19, 18, 16, 14],
    [20, 18, 16, 19],
    [21, 18, 16, 19]
])


