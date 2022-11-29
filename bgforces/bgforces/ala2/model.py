from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl

import bgflow as bg
from bgflow.factory import (
    BoltzmannGeneratorBuilder,
    InternalCoordinateMarginals,
    BONDS, ANGLES, TORSIONS, FIXED, ShapeInfo,
    LinLogCutEnergy
)
from .data import Ala2Data
from .plot import *
from .zmatrix import get_z_matrix
from bgmol.datasets import AImplicitUnconstrained

__all__ = ["Ala2Generator"]


class Ala2Generator(pl.LightningModule):

    def __init__(
            self,
            force_weight=0.0,
            nll_weight=1.0,
            kld_weight=0.0,
            kld_batchsize=0,
            use_inverse_transformer=True,
            transformer_type="spline",
            activation_type="silu",
            hidden=(64, 64),
            n_torsion_blocks=2,
            n_angle_blocks=0,
            siren_scale_first_weights=False,
            siren_initialize=False,
            use_informed_marginals=True,
            lr=5e-4,
            lr_decay=0.7,
            z_id="z4",
            high_energy=1e3,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # build model
        self.model = self._build_bg(
            transformer_type,
            use_inverse_transformer,
            activation_type,
            hidden,
            n_torsion_blocks=n_torsion_blocks,
            n_angle_blocks=n_angle_blocks,
            siren_scale_first_weights=siren_scale_first_weights,
            siren_initialize=siren_initialize,
            use_informed_marginals=use_informed_marginals,
            z_id=z_id,
            high_energy=high_energy
        )

        # optimization settings
        weight_sum = force_weight + nll_weight + kld_weight
        self.force_weight = force_weight / weight_sum
        self.nll_weight = nll_weight / weight_sum
        self.kld_weight = kld_weight / weight_sum
        self.kld_batchsize = kld_batchsize
        self.lr = lr
        self.lr_decay = lr_decay

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Ala2Generator")
        parser.add_argument("--force-weight", type=float, default=0.0)
        parser.add_argument("--nll-weight", type=float, default=1.0)
        parser.add_argument("--kld-weight", type=float, default=0.0)
        parser.add_argument("--kld-batchsize", type=int, default=0)
        parser.add_argument("--inverse", dest="use_inverse_transformer", action="store_true")
        parser.add_argument("--no-inverse", dest="use_inverse_transformer", action="store_false")
        parser.set_defaults(use_inverse_transformer=True)
        parser.add_argument("--siren_scale_first_weights",
                            dest="siren_scale_first_weights", action="store_true")
        parser.add_argument("--no-siren_scale_first_weights",
                            dest="siren_scale_first_weights", action="store_false")
        parser.set_defaults(siren_scale_first_weights=False)
        parser.add_argument("--siren_initialize", dest="siren_initialize", action="store_true")
        parser.add_argument("--no-siren_initialize", dest="siren_initialize", action="store_false")
        parser.set_defaults(siren_initialize=False)
        parser.add_argument("--transformer-type", default="spline", choices=["spline", "mixture", "bspline"])
        parser.add_argument("--activation-type", default="silu", choices=["silu", "sin"])
        parser.add_argument("--hidden", type=int, default=(64, 64), nargs='*')
        parser.add_argument("--n-torsion-blocks", type=int, default=2)
        parser.add_argument("--n-angle-blocks", type=int, default=0)
        parser.add_argument("--informed-marginals",
                            dest="use_informed_marginals", action="store_true")
        parser.add_argument("--no-informed-marginals",
                            dest="use_informed_marginals", action="store_false")
        parser.set_defaults(use_informed_marginals=True)
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--lr_decay", type=float, default=0.7)
        parser.add_argument("--z-id", default="z4", choices=["z1", "z2", "z3", "z4"])
        parser.add_argument("--high-energy", type=float, default=1e3)
        return parent_parser

    def training_step(self, batch, batch_idx):
        xyz, forces = batch
        force_mse = self._force_mse(xyz, forces)
        nll = self._nll(xyz)
        kld = self._kld()
        loss = self._weighted_loss(force_mse, nll, kld)
        self.log("train_force_mse", force_mse)
        self.log("train_nll", nll)
        if kld is not None:
            self.log("train_kld", kld)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        xyz, forces = batch
        force_mse = self._force_mse(xyz, forces)
        nll = self._nll(xyz)
        kld = self._kld()
        loss = self._weighted_loss(force_mse, nll, kld)
        self.log("val_force_mse", force_mse)
        self.log("val_nll", nll)
        if kld is not None:
            self.log("val_kld", kld)
            self.log("val_sampling_efficiency", self._sampling_efficiency())
        self.log("val_loss", loss)
        # self.logger.experiment.add_image("samples_and_energies", self._plot())

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=self.lr_decay)
        return [optim, ], [scheduler, ]

    # Builder
    @staticmethod
    def _build_trafo_from_data(z_id="z4"):

        # get z-matrix
        zmatrix, rigid_block = get_z_matrix(z_id)

        if len(rigid_block) > 0:

            # read dataset to construct energy model and whitening layer
            data = Ala2Data()
            data.prepare_data()
            data.setup()

            target_energy = data.dataset.get_energy_model()

            # throw away 6 degrees of freedom (rotation and translation)
            dim_cartesian = len(rigid_block) * 3 - 6

            coordinate_transform = bg.MixedCoordinateTransformation(
                data=data.train_xyz,
                z_matrix=zmatrix,
                fixed_atoms=rigid_block,
                keepdims=dim_cartesian,
                normalize_angles=True,
            )
        else:
            coordinate_transform = bg.GlobalInternalCoordinateTransformation(
                z_matrix=zmatrix,
                normalize_angles=True,
            )

            target_energy = AImplicitUnconstrained(read=False).get_energy_model()

        return coordinate_transform, target_energy

    def _build_bg(
            self,
            transformer_type,
            use_inverse_transformer,
            activation_type,
            hidden,
            ctx={"device": torch.device("cpu"), "dtype": torch.float32},
            n_torsion_blocks=6,
            n_angle_blocks=2,
            siren_scale_first_weights=False,
            siren_initialize=False,
            use_informed_marginals=True,
            z_id="z1",
            high_energy=1e3,
    ):

        # parse data
        transformer_type = {
            "spline": bg.ConditionalSplineTransformer,
            "mixture": bg.MixtureCDFTransformer,
            "bspline": bg.ConditionalBSplineTransformer
        }[transformer_type]
        activation = {"silu": torch.nn.SiLU(), "sin": bg.nn.dense.Sin()}[activation_type]
        hidden = hidden

        coordinate_transform, target_energy = self._build_trafo_from_data(z_id)
        target_energy = LinLogCutEnergy(target_energy, high_energy=high_energy)

        shape_info = ShapeInfo.from_coordinate_transform(coordinate_transform)
        builder = BoltzmannGeneratorBuilder(shape_info, target_energy, **ctx)
        builder.DEFAULT_TRANSFORMER_TYPE = transformer_type
        builder.DEFAULT_TRANSFORMER_KWARGS = {"inverse": use_inverse_transformer}
        builder.DEFAULT_CONDITIONER_KWARGS = {
            "hidden": hidden,
            "activation": activation,
        }
        if activation_type == "sin":
            builder.DEFAULT_CONDITIONER_KWARGS["siren_scale_first_weights"] = siren_scale_first_weights,
            builder.DEFAULT_CONDITIONER_KWARGS["siren_initialize"] = siren_initialize

        if isinstance(coordinate_transform, bg.MixedCoordinateTransformation):
            for i in range(n_torsion_blocks):
                builder.add_condition(TORSIONS, on=FIXED)
                builder.add_condition(FIXED, on=TORSIONS)
            for i in range(n_angle_blocks):
                builder.add_condition(BONDS, on=ANGLES)
                builder.add_condition(ANGLES, on=BONDS)
            builder.add_condition(ANGLES, on=(TORSIONS, FIXED))
            builder.add_condition(BONDS, on=(ANGLES, TORSIONS, FIXED))
        else:
            # del shape_info[ORIGIN]
            # del shape_info[ROTATION]
            n_torsions = builder.current_dims[TORSIONS][-1]
            TORSIONS1, TORSIONS2 = builder.add_split(
                TORSIONS,
                ("TORSIONS1", "TORSIONS2"),
                (n_torsions // 2, n_torsions - n_torsions // 2),
            )
            for i in range(n_torsion_blocks):
                builder.add_condition(TORSIONS1, on=TORSIONS2)
                builder.add_condition(TORSIONS2, on=TORSIONS1)
            builder.add_merge((TORSIONS1, TORSIONS2), TORSIONS)
            for i in range(n_angle_blocks):
                builder.add_condition(BONDS, on=ANGLES)
                builder.add_condition(ANGLES, on=BONDS)
            builder.add_condition(ANGLES, on=(TORSIONS))
            builder.add_condition(BONDS, on=(ANGLES, TORSIONS))

        if use_informed_marginals:
            marginals = InternalCoordinateMarginals(builder.current_dims, builder.ctx)
            marginals[BONDS] = bg.SloppyUniform(
                0.05 * torch.ones(builder.current_dims[BONDS][-1], **builder.ctx),
                0.3 * torch.ones(builder.current_dims[BONDS][-1], **builder.ctx),
                validate_args=False
            )
            marginals[ANGLES] = bg.SloppyUniform(
                0.15 * torch.ones(builder.current_dims[ANGLES][-1], **builder.ctx),
                1.0 * torch.ones(builder.current_dims[ANGLES][-1], **builder.ctx),
                validate_args=False
            )
            if FIXED in builder.current_dims:
                marginals[FIXED] = bg.SloppyUniform(
                    -5.0 * torch.ones(builder.current_dims[FIXED][-1], **builder.ctx),
                    5.0 * torch.ones(builder.current_dims[FIXED][-1], **builder.ctx),
                    validate_args=False
                )
            builder.add_map_to_ic_domains(marginals)
        else:
            builder.add_map_to_ic_domains()
        builder.add_map_to_cartesian(coordinate_transform)

        return builder.build_generator()

        # Losses

    def _force_mse(self, xyz, reference_force):
        force = self.model.force(xyz)
        return ((force - reference_force) ** 2).mean()

    def _nll(self, xyz):
        energy = self.model.energy(xyz)
        return energy.mean()

    def _kld(self):
        if self.kld_batchsize != 0:
            kld = self.model.kldiv(self.kld_batchsize)
            return kld.mean()
        else:
            return None
        
    def _kld_for_samples(self,numsamples):
        kld = self.model.kldiv(numsamples)
        return kld.mean()


    def _weighted_loss(self, force_mse, nll, kld):
        loss = self.force_weight * force_mse
        loss = loss + self.nll_weight * nll
        if kld is not None:
            loss = loss + self.kld_weight * kld
        return loss

    def _sampling_efficiency(self):
        x = self.model.sample(self.kld_batchsize)
        log_weights = self.model.log_weights(x)
        ess = torch.exp(2 * torch.logsumexp(log_weights, dim=-1) - torch.logsumexp(2 * log_weights, dim=-1))
        return ess / self.kld_batchsize

    # Logging

    def _plot(self, data, target_energy, n_samples=5000):
        samples = self.model.sample(n_samples)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        fig.tight_layout()

        plot_phi_psi(axes[0], samples, data.dataset.system)
        plot_energies(axes[1], samples, target_energy, data.val_xyz)
        del samples

        return fig
        # fig.savefig(f"ala2_fm_bump.png", dpi=300, bbox_inches="tight")
