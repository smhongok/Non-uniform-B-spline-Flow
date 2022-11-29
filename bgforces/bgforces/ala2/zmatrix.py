
import numpy as np
from bgmol.systems.ala2 import DEFAULT_RIGID_BLOCK, DEFAULT_Z_MATRIX


def get_z_matrix(z_id="z1"):
    if z_id == "z1":
        return DEFAULT_Z_MATRIX, DEFAULT_RIGID_BLOCK
    elif z_id == "z2":
        zmatrix = np.row_stack([DEFAULT_Z_MATRIX, np.array([[9, 8, 6, 14]])])
        rigid_block = np.array([6, 8, 10, 14])
        return zmatrix, rigid_block
    elif z_id == "z3":
        zmatrix = np.row_stack([DEFAULT_Z_MATRIX, np.array([[9, 8, 6, 14], [10, 8, 14, 6]])])
        rigid_block = np.array([6, 8, 14])
        return zmatrix, rigid_block
    elif z_id == "z4":
        zmatrix = np.row_stack([
            DEFAULT_Z_MATRIX, 
            np.array([
                [9, 8, 6, 14],
                [10, 8, 14, 6],
                [6, 8, 14, -1],
                [8, 14, -1, -1],
                [14, -1, -1, -1]
            ])
        ])
        rigid_block = np.array([])
        return zmatrix, rigid_block
    else:
        raise UserException(f"invalid z_id {z_id}")
