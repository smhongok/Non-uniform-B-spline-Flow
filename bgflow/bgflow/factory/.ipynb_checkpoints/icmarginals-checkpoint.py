
import numpy as np
import torch

import bgflow as bg
from .tensor_info import BONDS, ANGLES, TORSIONS, FIXED, AUGMENTED

__all__ = ["InternalCoordinateMarginals"]


class InternalCoordinateMarginals(dict):
    def __init__(
            self,
            current_dims,
            ctx,
            bond_mu=1.0,
            bond_sigma=1.0,
            bond_lower=1e-5,
            bond_upper=np.infty,
            angle_mu=0.5,
            angle_sigma=1.0,
            angle_lower=1e-5,
            angle_upper=1.0,
            torsion_lower=0.0,
            torsion_upper=1.0,
            fixed_scale=20.0,
            bonds=BONDS,
            angles=ANGLES,
            torsions=TORSIONS,
            fixed=FIXED,
            augmented=AUGMENTED,
    ):
        self.current_dims = current_dims
        self.ctx = ctx
        super().__init__()
        if bonds in current_dims:
            self[bonds] = bg.TruncatedNormalDistribution(
                mu=bond_mu*torch.ones(current_dims[bonds], **ctx),
                sigma=bond_sigma*torch.ones(current_dims[bonds], **ctx),
                lower_bound=torch.tensor(bond_lower, **ctx),
                upper_bound=torch.tensor(bond_upper, **ctx),
            )

        # angles
        if angles in current_dims:
            self[angles] = bg.TruncatedNormalDistribution(
                mu=angle_mu*torch.ones(current_dims[angles], **ctx),
                sigma=angle_sigma*torch.ones(current_dims[angles], **ctx),
                lower_bound=torch.tensor(angle_lower, **ctx),
                upper_bound=torch.tensor(angle_upper, **ctx),
            )

        # angles
        if torsions in current_dims:
            self[torsions] = bg.SloppyUniform(
                low=torsion_lower*torch.ones(current_dims[torsions], **ctx),
                high=torsion_upper*torch.ones(current_dims[torsions], **ctx)
            )

        # fixed
        if fixed in current_dims:
            self[fixed] = torch.distributions.Normal(
                loc=torch.zeros(current_dims[fixed], **ctx),
                scale=fixed_scale*torch.ones(current_dims[fixed], **ctx)
            )

        # augmented
        if augmented in current_dims:
            self[augmented] = torch.distributions.Normal(
                loc=torch.zeros(current_dims[augmented], **ctx),
                scale=torch.ones(current_dims[augmented], **ctx)
            )

