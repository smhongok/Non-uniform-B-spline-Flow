import torch
from torch.nn import functional as F
import math
import random
from .base import Transformer
from nflows.transforms.base import InputOutsideDomain
from nflows.utils import torchutils
from matplotlib import pyplot as plt
import time
import numpy as np
__all__ = [
    "ConditionalBSplineTransformer_with_plot",
]

DEFAULT_MIN_BIN_WIDTH = 1e-4
DEFAULT_MIN_BIN_HEIGHT = 1e-4
class ConditionalBSplineTransformer_with_plot(Transformer):
    def __init__(
        self,
        params_net: torch.nn.Module,
        is_circular: bool = False,
        left: float = 0.0,
        right: float = 1.0,
        bottom: float = 0.0,
        top: float = 1.0,
    ):
        """
        Spline transformer transforming variables in [left, right) into variables in [bottom, top).

        Uses `n_bins` spline nodes to define a inverse CDF transform on the interval.

        Uses bayesiains/nflows under the hood.

        Parameters
        ----------
        params_net: torch.nn.Module
            Computes the transformation parameters for `y` conditioned
            on `x`. Input dimension must be `x.shape[-1]`. Output
            dimension must be
                - `y.shape[-1] * n_bins * 3` if splines are circular,
                - `y.shape[-1] * (n_bins * 3 + 1)` if not
                - `y.shape[-1] * n_bins * 3 + n_circular if some transformed variables are circular
            The integer `n_bins` is inferred implicitly from the network output shape.
        is_circular : bool or torch.Tensor
            If True, the boundaries are treated as periodic boundaries, i.e. the pdf is enforced to be continuous.
            If is_circular is a boolean tensor, only the indices at which the tensor is True are treated as periodic
            (tensor shape has to be (y.shape[-1], ) ).

        Raises
        ------
        RuntimeError
            If the `params_net` output does not have the correct shape.

        Notes
        -----
        This transform requires the nflows package to be installed.
        It is available from PyPI and can be installed with `pip install nflows`.

        References
        ----------
        C. Durkan, A. Bekasov, I. Murray, G. Papamakarios, Neural Spline Flows, NeurIPS 2019,
        https://arxiv.org/abs/1906.04032.
        """
        super().__init__()
        self._params_net = params_net
        self._is_circular = torch.tensor(is_circular, dtype=torch.bool)
        self._left = left
        self._right = right
        self._bottom = bottom
        self._top = top

    def _compute_params(self, x, y_dim):
        """Compute widths, heights, and slopes from x through the params_net.

        Parameters
        ----------
        x : torch.Tensor
            Conditioner input.
        y_dim : int
            Number of transformed degrees of freedom.

        Returns
        -------
        widths : torch.Tensor
            unnormalized bin widths for the monotonic spline interpolation
            shape ( ... , y_dim, n_bins), where ... represents batch dims
        heights : torch.Tensor
            unnormalized bin heights for the monotonic spline interpolation
            shape ( ... , y_dim, n_bins)
        slopes : torch.Tensor
            unnormalized slopes for the monotonic spline interpolation
            shape (... , y_dim, n_bins + 1)
        """
        params = self._params_net(x)
        # assume that all but the last dim of the params tensor are batch dims
        batch_shape = params.shape[:-1]
        n_bins = (params.shape[-1] - (2 + 4) * self._n_noncircular(y_dim)) // (y_dim * 2)
        #print(self._n_noncircular(y_dim))
        pieces, noncircular_pieces, noncircular_pieces2, widths,noncircular_widths1,noncircular_widths2,noncircular_widths3, noncircular_widths4 = torch.split(
            params,
            [n_bins * y_dim, self._n_noncircular(y_dim), self._n_noncircular(y_dim),n_bins * y_dim, self._n_noncircular(y_dim), self._n_noncircular(y_dim), self._n_noncircular(y_dim), self._n_noncircular(y_dim)],
            dim=-1
        )

        pieces = pieces.reshape(*batch_shape, y_dim, n_bins)
        noncircular_pieces = noncircular_pieces.reshape(*batch_shape, -1)
        noncircular_pieces2 = noncircular_pieces2.reshape(*batch_shape, -1)
        widths = widths.reshape(*batch_shape, y_dim, n_bins)
        noncircular_widths1 = noncircular_widths1.reshape(*batch_shape, -1)
        noncircular_widths2 = noncircular_widths2.reshape(*batch_shape, -1)
        noncircular_widths3 = noncircular_widths3.reshape(*batch_shape, -1)
        noncircular_widths4 = noncircular_widths4.reshape(*batch_shape, -1)
        # make periodic
        pieces = torch.cat([pieces, pieces[..., [0]], pieces[..., [1]]], dim=-1)
        widths = torch.cat([widths, widths[..., [0]], widths[..., [1]], widths[..., [-2]], widths[..., [-1]]], dim=-1)
        # make noncircular indices non-periodic
        pieces[..., self._noncircular_indices(y_dim), -2] = noncircular_pieces
        pieces[..., self._noncircular_indices(y_dim), -1] = noncircular_pieces2
        widths[..., self._noncircular_indices(y_dim), -4] = noncircular_widths1
        widths[..., self._noncircular_indices(y_dim), -3] = noncircular_widths2
        widths[..., self._noncircular_indices(y_dim), -2] = noncircular_widths3
        widths[..., self._noncircular_indices(y_dim), -1] = noncircular_widths4
        return pieces, widths

    def _forward(self, x, y, *args, elementwise_jacobian=False, **kwargs):

        pieces,widths = self._compute_params(x, y.shape[-1])
        z, dlogp = cubic_B_spline_with_plot(
            y,
            pieces,
            widths,
            inverse=True, # ?? why this was True in SNF
            left=self._left,
            right=self._right,
            top=self._top,
            bottom=self._bottom,
        )
        if not elementwise_jacobian:
            dlogp = dlogp.sum(dim=-1, keepdim=True)
        return z, dlogp

    def _inverse(self, x, y, *args, elementwise_jacobian=False, **kwargs):

        pieces, widths= self._compute_params(x, y.shape[-1])
        z, dlogp = cubic_B_spline_with_plot(
            y,
            pieces,
            widths,
            inverse=False, # ?? why this was False in SNF
            left=self._left,
            right=self._right,
            top=self._top,
            bottom=self._bottom,
        )
        if not elementwise_jacobian:
            dlogp = dlogp.sum(dim=-1, keepdim=True)
        return z, dlogp

    def _n_noncircular(self, y_dim):
        if self._is_circular.all():
            return 0
        elif not self._is_circular.any():
            return y_dim
        else:
            return self._is_circular.sum()

    def _noncircular_indices(self, y_dim):
        if self._is_circular.all():
            return slice(0)
        elif not self._is_circular.any():
            return slice(None)
        else:
            return torch.logical_not(self._is_circular)


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[...,-1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def cbrt(x, eps=0):
    ans = torch.sign(x)*torch.exp(torch.log(torch.abs(x))/3.0)
    return ans


def sqrt(x, eps=1e-9):
    ans = torch.exp((torch.log(torch.abs(x))) / 2.0)
    return ans
"""
def plot_B_spline( #nonuniform, # w\o identity
    knots,
    t,
    cumwidths,
    cumheights,
    num_bins,
    eps,
    linear_threshold,
    quadratic_threshold,
    inverse,
):
    # assume all tensors are 1d as like as inputs
    # assume all tensors are in CPU
    inputs = torch.linspace(1e-6, 1-1e-6, steps=100)
    inputs = torch.reshape(inputs, (100, 1))
    #print(knots.size(dim=-1))
    knots = knots.repeat(100,1, 1)
    t=t.repeat(100, 1,1)
    cumwidths=cumwidths.repeat(100, 1, 1)
    cumheights=cumheights.repeat(100, 1,1 )
    #print(cumwidths.shape)
    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]
    #print(bin_idx.shape)
    i0 = bin_idx
    im1 = torch.remainder(bin_idx - 1, num_bins + 3)
    im2 = torch.remainder(bin_idx - 2, num_bins + 3)
    im3 = torch.remainder(bin_idx - 3, num_bins + 3)

    j3 = bin_idx + 3
    j2 = bin_idx + 2
    j1 = bin_idx + 1
    j0 = bin_idx
    jm1 = torch.remainder(bin_idx - 1, num_bins + 5)
    jm2 = torch.remainder(bin_idx - 2, num_bins + 5)

    # print("checkpoint1")
    #print(knots.shape)
    #print(inputs.shape)
    km0 = knots.gather(-1, i0)[..., 0]
    km1 = knots.gather(-1, im1)[..., 0]
    km2 = knots.gather(-1, im2)[..., 0]
    km3 = knots.gather(-1, im3)[..., 0]

    t3 = t.gather(-1, j3)[..., 0]
    t2 = t.gather(-1, j2)[..., 0]
    t1 = t.gather(-1, j1)[..., 0]
    t0 = t.gather(-1, j0)[..., 0]
    tm1 = t.gather(-1, jm1)[..., 0]
    tm2 = t.gather(-1, jm2)[..., 0]

    inputs_a1 = km0 * (
            1 / ((t3 - t0) * (t2 - t0) * (t1 - t0))
    ) + km1 * (
                        - 1 / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                        - 1 / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                        - 1 / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                ) + km2 * (
                        1 / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                        + 1 / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                        + 1 / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                ) + km3 * (
                        -1 / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                )

    inputs_b1 = km0 * (
            (-3 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
    ) + km1 * (
                        (2 * tm1 + t1) / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                        + (tm1 + t2 + t0) / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                        + (t3 + 2 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                ) + km2 * (
                        (-2 * t1 - tm2) / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                        + (-2 * t2 - t0) / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                        + (-t2 - t1 - tm1) / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                ) + km3 * (
                        (3 * t1) / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                )

    inputs_c1 = km0 * (
            (3 * t0 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
    ) + km1 * (
                        (- tm1 * tm1 - 2 * tm1 * t1) / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                        + (- tm1 * t2 - tm1 * t0 - t2 * t0) / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                        + (- t0 * t0 - 2 * t3 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                ) + km2 * (
                        (t1 * t1 + 2 * t1 * tm2) / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                        + (t2 * t2 + 2 * t0 * t2) / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                        + (t2 * t1 + tm1 * t1 + t2 * tm1) / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                ) + km3 * (
                        (-3 * t1 * t1) / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                )

    inputs_d1 = km0 * (
            (- t0 * t0 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
    ) + km1 * (
                        (tm1 * tm1 * t1) / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                        + (tm1 * t2 * t0) / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                        + (t3 * t0 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                ) + km2 * (
                        - (t1 * t1 * tm2) / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                        - (t0 * t2 * t2) / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                        - (t2 * tm1 * t1) / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                ) + km3 * (
                        (t1 * t1 * t1) / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                )

    input_left_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_right_cumwidths = cumwidths.gather(-1, bin_idx + 1)[..., 0]
    input_low_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_high_cumheights = cumheights.gather(-1, bin_idx + 1)[..., 0]

    if inverse:

        inputs = torch.as_tensor(inputs,dtype=torch.double)
        #outputs = inputs
        outputs = torch.zeros_like(inputs)
        inputs_b_ = torch.as_tensor(inputs_b1 / inputs_a1 / 3.,dtype=torch.double)
        inputs_c_ = torch.as_tensor(inputs_c1 / inputs_a1 / 3.,dtype=torch.double)
        inputs_d_ = torch.as_tensor((inputs_d1 - inputs) / inputs_a1,dtype=torch.double)
        delta_1 = -inputs_b_.pow(2) + inputs_c_
        delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
        delta_3 = inputs_b_ * inputs_d_ - inputs_c_.pow(2)

        discriminant = 4. * delta_1 * delta_3 - delta_2.pow(2)

        depressed_1 = -2. * inputs_b_ * delta_1 + delta_2
        depressed_2 = delta_1


        three_roots_mask = discriminant >= 0  # Discriminant == 0 might be a problem in practice.
        one_root_mask = discriminant < 0


        # Deal with one root cases.
        p_ = torch.zeros_like(inputs)
        p_[one_root_mask] = cbrt((-depressed_1[one_root_mask] + sqrt(-discriminant[one_root_mask])) / 2.)

        p = p_[one_root_mask]
        q = cbrt((-depressed_1[one_root_mask] - sqrt(-discriminant[one_root_mask])) / 2.)

        outputs_one_root = ((p+q) - inputs_b_[one_root_mask])

        outputs[one_root_mask] = torch.as_tensor(outputs_one_root,dtype=outputs.dtype)

        # Deal with three root cases.

        theta = torch.atan2(sqrt(discriminant[three_roots_mask]), -depressed_1[three_roots_mask])
        theta /= 3.

        cubic_root_1 = torch.cos(theta)
        cubic_root_2 = torch.sin(theta)

        root_1 = cubic_root_1
        root_2 = -0.5 * cubic_root_1 - 0.5 * math.sqrt(3) * cubic_root_2
        root_3 = -0.5 * cubic_root_1 + 0.5 * math.sqrt(3) * cubic_root_2

        root_scale = 2 * sqrt(-depressed_2[three_roots_mask])
        root_shift = -inputs_b_[three_roots_mask]

        root_1 = root_1 * root_scale + root_shift
        root_2 = root_2 * root_scale + root_shift
        root_3 = root_3 * root_scale + root_shift

        root1_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_1).float()
        root1_mask *= (root_1 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        root2_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_2).float()
        root2_mask *= (root_2 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        root3_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_3).float()
        root3_mask *= (root_3 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        roots = torch.stack([root_1, root_2, root_3], dim=-1)

        masks = torch.stack([root1_mask, root2_mask, root3_mask], dim=-1)
        mask_index = torch.argsort(masks, dim=-1, descending=True)[..., 0][..., None]
        output_three_roots = torch.gather(roots, dim=-1, index=mask_index).view(-1)
        outputs[three_roots_mask] = torch.as_tensor(output_three_roots,dtype=outputs.dtype)

        # Deal with a -> 0 (almost quadratic) cases.

        quadratic_mask = inputs_a1.abs() < quadratic_threshold
        a = inputs_b1[quadratic_mask]
        b = inputs_c1[quadratic_mask]
        c = (inputs_d1[quadratic_mask] - inputs[quadratic_mask])
        # assert (b.pow(2)-4*a*c>0).all()
        # alpha = (-b + sqrt(b.pow(2) - 4 * a * c)) / 2 * reciprocal(a)
        alpha = (-b + sqrt(b.pow(2) - 4 * a * c)) / (2 * a)
        #alpha = 2 * c / ( - b - torch.sqrt(b.pow(2) - 4 * a * c))
        outputs[quadratic_mask] = torch.as_tensor(alpha,dtype=outputs.dtype) # + input_left_cumwidths[quadratic_mask]

        # Deal with b-> 0 (almost linear) cases.
        linear_mask = inputs_b1.abs() < linear_threshold
        linear_mask = linear_mask * quadratic_mask
        b = inputs_c1[linear_mask]
        c = (inputs_d1[linear_mask] - inputs[linear_mask])
        alpha = c / b
        outputs[linear_mask] = torch.as_tensor(alpha, dtype=outputs.dtype)

        outputs = torch.clamp(outputs, input_left_cumwidths, input_right_cumwidths)
        # shifted_outputs = (outputs - input_left_cumwidths)/m
        logabsdet = -torch.log(
            (
                (3 * inputs_a1 * outputs.pow(2)
                    + 2 * inputs_b1 * outputs
                    + inputs_c1)
            )
        )

    else:
        outputs = (inputs_a1 * inputs.pow(3) +
                   inputs_b1 * inputs.pow(2) +
                   inputs_c1 * inputs +
                   inputs_d1)
        outputs = torch.clamp(outputs, input_low_cumheights, input_high_cumheights)


        logabsdet = torch.log((3 * inputs_a1 * inputs.pow(2) +
                               2 * inputs_b1 * inputs +
                               inputs_c1))

    outputs = torch.as_tensor(outputs, dtype=torch.float).numpy()
    logabsdet = torch.as_tensor(logabsdet, dtype=torch.float).numpy()
    #print(outputs)
    plt.plot(inputs,outputs)
    now = (time.time())
    plt.savefig('bsplineTransformation//bsplineTransformation'+str(now) + '.png')
    plt.savefig('bsplineTransformation//bsplineTransformation'+str(now) + '.pdf')
    #plt.show()
    #plt.tight_layout()
    #plt.plot(inputs, logabsdet)
    #plt.show()
    #plt.savefig('savefig_default.png')
    
    
    #plt.savefig('bsplineTransformation'+str(now) + '.png')
"""
    
def plot_B_spline(
    knots,
    t,
    cumwidths,
    cumheights,
    num_bins,
    eps,
    linear_threshold,
    quadratic_threshold,
    inverse,
):
    inputs = torch.linspace(1e-6, 1-1e-6, steps=100)
    inputs = torch.reshape(inputs, (100, 1))
    #print(knots.size(dim=-1))
    knots = knots.repeat(100,1, 1)
    t=t.repeat(100, 1,1)
    cumwidths=cumwidths.repeat(100, 1, 1)
    cumheights=cumheights.repeat(100, 1,1 )
    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
        input_low_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        input_high_cumheights = cumheights.gather(-1, bin_idx + 1)[..., 0]

        i0 = bin_idx
        im1 = torch.remainder(bin_idx - 1, num_bins + 3)
        im2 = torch.remainder(bin_idx - 2, num_bins + 3)
        im3 = torch.remainder(bin_idx - 3, num_bins + 3)

        j3 = bin_idx + 3
        j2 = bin_idx + 2
        j1 = bin_idx + 1
        j0 = bin_idx
        jm1 = torch.remainder(bin_idx - 1, num_bins + 5)
        jm2 = torch.remainder(bin_idx - 2, num_bins + 5)

        km0 = knots.gather(-1, i0)[..., 0]
        km1 = knots.gather(-1, im1)[..., 0]
        km2 = knots.gather(-1, im2)[..., 0]
        km3 = knots.gather(-1, im3)[..., 0]

        t3 = t.gather(-1, j3)[..., 0]
        t2 = t.gather(-1, j2)[..., 0]
        t1 = t.gather(-1, j1)[..., 0]
        t0 = t.gather(-1, j0)[..., 0]
        tm1 = t.gather(-1, jm1)[..., 0]
        tm2 = t.gather(-1, jm2)[..., 0]
        
        input_left_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_right_cumwidths = cumwidths.gather(-1, bin_idx + 1)[..., 0]
        
        inputs_a1 = km0 * (
                1 / ((t3 - t0) * (t2 - t0) * (t1 - t0))
        ) + km1 * (
                            - 1 / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                            - 1 / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                            - 1 / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                    ) + km2 * (
                            1 / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                            + 1 / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                            + 1 / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                    ) + km3 * (
                            -1 / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                    )

        inputs_b1 = km0 * (
                (-3 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
        ) + km1 * (
                            (2 * tm1 + t1) / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                            + (tm1 + t2 + t0) / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                            + (t3 + 2 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                    ) + km2 * (
                            (-2 * t1 - tm2) / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                            + (-2 * t2 - t0) / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                            + (-t2 - t1 - tm1) / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                    ) + km3 * (
                            (3 * t1) / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                    )

        inputs_c1 = km0 * (
                (3 * t0 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
        ) + km1 * (
                            (- tm1 * tm1 - 2 * tm1 * t1) / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                            + (- tm1 * t2 - tm1 * t0 - t2 * t0) / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                            + (- t0 * t0 - 2 * t3 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                    ) + km2 * (
                            (t1 * t1 + 2 * t1 * tm2) / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                            + (t2 * t2 + 2 * t0 * t2) / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                            + (t2 * t1 + tm1 * t1 + t2 * tm1) / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                    ) + km3 * (
                            (-3 * t1 * t1) / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                    )

        inputs_d1 = km0 * (
                (- t0 * t0 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
        ) + km1 * (
                            (tm1 * tm1 * t1) / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                            + (tm1 * t2 * t0) / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                            + (t3 * t0 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                    ) + km2 * (
                            - (t1 * t1 * tm2) / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                            - (t0 * t2 * t2) / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                            - (t2 * tm1 * t1) / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                    ) + km3 * (
                            (t1 * t1 * t1) / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                    )            
    
        inputs = torch.as_tensor(inputs, dtype=torch.double)
        outputs = torch.zeros_like(inputs)
        inputs_b_ = torch.as_tensor(inputs_b1 / inputs_a1 / 3., dtype=torch.double, device=inputs_b1.device)
        inputs_c_ = torch.as_tensor(inputs_c1 / inputs_a1 / 3., dtype=torch.double, device=inputs_b1.device)
        inputs_d_ = torch.as_tensor((inputs_d1 - inputs) / inputs_a1, dtype=torch.double, device=inputs_b1.device)
        delta_1 = -inputs_b_.pow(2) + inputs_c_
        delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
        delta_3 = inputs_b_ * inputs_d_ - inputs_c_.pow(2)

        discriminant = 4. * delta_1 * delta_3 - delta_2.pow(2)

        depressed_1 = -2. * inputs_b_ * delta_1 + delta_2
        depressed_2 = delta_1

        three_roots_mask = discriminant >= 0  # Discriminant == 0 might be a problem in practice.
        one_root_mask = discriminant < 0

        # Deal with one root cases.
        p_ = torch.zeros_like(inputs)
        p_[one_root_mask] = cbrt((-depressed_1[one_root_mask] + sqrt(-discriminant[one_root_mask])) / 2.)

        p = p_[one_root_mask]
        q = cbrt((-depressed_1[one_root_mask] - sqrt(-discriminant[one_root_mask])) / 2.)

        outputs_one_root = ((p + q) - inputs_b_[one_root_mask])

        outputs[one_root_mask] = torch.as_tensor(outputs_one_root, dtype=outputs.dtype)

        # Deal with three root cases.

        theta = torch.atan2(sqrt(discriminant[three_roots_mask]), -depressed_1[three_roots_mask])
        theta /= 3.

        cubic_root_1 = torch.cos(theta)
        cubic_root_2 = torch.sin(theta)

        root_1 = cubic_root_1
        root_2 = -0.5 * cubic_root_1 - 0.5 * math.sqrt(3) * cubic_root_2
        root_3 = -0.5 * cubic_root_1 + 0.5 * math.sqrt(3) * cubic_root_2

        root_scale = 2 * sqrt(-depressed_2[three_roots_mask])
        root_shift = -inputs_b_[three_roots_mask]

        root_1 = root_1 * root_scale + root_shift
        root_2 = root_2 * root_scale + root_shift
        root_3 = root_3 * root_scale + root_shift

        root1_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_1).float()
        root1_mask *= (root_1 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        root2_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_2).float()
        root2_mask *= (root_2 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        root3_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_3).float()
        root3_mask *= (root_3 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        roots = torch.stack([root_1, root_2, root_3], dim=-1)

        masks = torch.stack([root1_mask, root2_mask, root3_mask], dim=-1)
        mask_index = torch.argsort(masks, dim=-1, descending=True)[..., 0][..., None]
        output_three_roots = torch.gather(roots, dim=-1, index=mask_index).view(-1)
        outputs[three_roots_mask] = torch.as_tensor(output_three_roots, dtype=outputs.dtype)

        # Deal with a -> 0 (almost quadratic) cases.

        quadratic_mask = inputs_a1.abs() < quadratic_threshold
        a = inputs_b1[quadratic_mask]
        b = inputs_c1[quadratic_mask]
        c = (inputs_d1[quadratic_mask] - inputs[quadratic_mask])
        alpha = (-b + sqrt(b.pow(2) - 4 * a * c)) / (2 * a)
        outputs[quadratic_mask] = torch.as_tensor(alpha, dtype=outputs.dtype)  # + input_left_cumwidths[quadratic_mask]

        # Deal with b-> 0 (almost linear) cases.
        linear_mask = inputs_b1.abs() < linear_threshold
        linear_mask = linear_mask * quadratic_mask
        b = inputs_c1[linear_mask]
        c = (inputs_d1[linear_mask] - inputs[linear_mask])
        alpha = c / b
        outputs[linear_mask] = torch.as_tensor(alpha, dtype=outputs.dtype)

        outputs = torch.clamp(outputs, input_left_cumwidths, input_right_cumwidths)
        logabsdet = -torch.log(
            (torch.abs(
                (3 * inputs_a1 * outputs.pow(2)
                 + 2 * inputs_b1 * outputs
                 + inputs_c1))
            )
        )
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

        i0 = bin_idx
        im1 = torch.remainder(bin_idx - 1, num_bins + 3)
        im2 = torch.remainder(bin_idx - 2, num_bins + 3)
        im3 = torch.remainder(bin_idx - 3, num_bins + 3)

        j3 = bin_idx + 3
        j2 = bin_idx + 2
        j1 = bin_idx + 1
        j0 = bin_idx
        jm1 = torch.remainder(bin_idx - 1, num_bins + 5)
        jm2 = torch.remainder(bin_idx - 2, num_bins + 5)

        km0 = knots.gather(-1, i0)[..., 0]
        km1 = knots.gather(-1, im1)[..., 0]
        km2 = knots.gather(-1, im2)[..., 0]
        km3 = knots.gather(-1, im3)[..., 0]

        t3 = t.gather(-1, j3)[..., 0]
        t2 = t.gather(-1, j2)[..., 0]
        t1 = t.gather(-1, j1)[..., 0]
        t0 = t.gather(-1, j0)[..., 0]
        tm1 = t.gather(-1, jm1)[..., 0]
        tm2 = t.gather(-1, jm2)[..., 0]

        input_left_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_right_cumwidths = cumwidths.gather(-1, bin_idx + 1)[..., 0]

        w_j_2 = (inputs - t0) / (t1 - t0)
        w_j_3 = (inputs - t0) / (t2 - t0) # (x-t_j)/(t_j+2 - t_j)
        w_jm1_3 = (inputs - tm1) / (t1 - tm1)

        B_jm2_3 = (1-w_jm1_3)*(1-w_j_2)
        B_jm1_3 = w_jm1_3 * (1-w_j_2) + (1-w_j_3) * w_j_2
        B_j_3 = w_j_3 * w_j_2
        D_jm2_3 = (km2-km3) / (t1-tm2) 
        D_jm1_3 = (km1-km2) / (t2-tm1) 
        D_j_3 = (km0-km1) / (t3-t0) 
        
        absdet = 3*(D_jm2_3*B_jm2_3+D_jm1_3*B_jm1_3+D_j_3*B_j_3)
        logabsdet = torch.log(absdet)
        
        outputs = (km3 + (inputs - tm2)*D_jm2_3)*B_jm2_3 + (km2 + (inputs - tm1)*D_jm1_3)*B_jm1_3 + (km1 + (inputs - t0)*D_j_3)*B_j_3
        #outputs = outputs * (top - bottom) + bottom
        #logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)        
        
    outputs = torch.as_tensor(outputs, dtype=torch.float).numpy()
    logabsdet = torch.as_tensor(logabsdet, dtype=torch.float).numpy()
    #print(outputs)
    plt.plot(inputs,outputs)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.axis('square')
    plt.tight_layout()
    now = (time.time())
    plt.savefig('bsplineTransformation//bsplineTransformation'+str(now) + '.png')
    plt.savefig('bsplineTransformation//bsplineTransformation'+str(now) + '.pdf')
        
        
def cubic_B_spline_with_plot(  # nonuniform , # w/o identity
        inputs,
        unnormalized_var1,
        unnormalized_widths,
        inverse=False,
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
):
    inverse = not inverse # if this line is on : easy (fwd)
    #print("inv")
    eps = 1e-4
    quadratic_threshold = 1e-7
    linear_threshold = 1e-7
    num_d = unnormalized_var1.shape[-1]
    num_bins = num_d - 2
    m = 1.0 / num_bins

    if torch.is_tensor(left):
        lim_tensor = True
    else:
        lim_tensor = False

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')
    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
        widths = torch.softmax(unnormalized_widths, dim=-1)  # num_bins+4 dim initialize
        widths = min_bin_width + (1 - (num_bins + 4) * min_bin_width) * widths
        widths = (widths / torch.sum(widths[..., 0:num_bins], dim=-1, keepdim=True))
        widths = torch.as_tensor(widths, dtype = torch.double)
        cumwidths = torch.cumsum(widths[..., 0:num_bins], dim=-1)
        cumwidths[..., -1] = 1.
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)

        t = F.pad(cumwidths, pad=(0, 4), mode='constant', value=0.0)  # num_bins + 5 dim initialize

        t[..., -1] = t[..., 0] - widths[..., -1]
        t[..., -2] = t[..., -1] - widths[..., -2]
        t[..., num_bins + 1] = t[..., num_bins] + widths[..., num_bins]
        t[..., num_bins + 2] = t[..., num_bins + 1] + widths[..., num_bins + 1]


        var2 = torch.softmax(unnormalized_var1[...,0:num_bins+2], dim=-1)  # num_bins + 2 dim initialize
        var2 = min_bin_height + (1 - (num_bins+2) * min_bin_height) * var2

        knots = torch.cumsum(var2, dim=-1) # r-2,  ... , s-1
        knots = knots[...,:-1]
        knots = torch.as_tensor(knots, dtype=torch.double)

        w_m1_3 = widths[...,-1] / (widths[...,-1] + widths[...,0])
        w_m1_4 = widths[...,-1] / (widths[...,-1] + widths[...,0] + widths[...,1])
        w_m2_4 = (widths[...,-1] + widths[...,-2]) / (widths[...,-2] + widths[...,-1] + widths[...,0])
        w_km1_3 = widths[...,num_bins-1] / (widths[...,num_bins-1] + widths[...,num_bins])
        w_km1_4 = widths[...,num_bins-1] / (widths[...,num_bins-1] + widths[...,num_bins] + widths[...,num_bins+1])
        w_km2_4 = (widths[...,num_bins-1] + widths[...,num_bins-2]) / (widths[...,num_bins-2] + widths[...,num_bins-1] + widths[...,num_bins])

        f_r = knots[...,0] * ( (1-w_m1_3)*w_m2_4 + w_m1_3*(1-w_m1_4)) + knots[...,1]*w_m1_3*w_m1_4
        f_s = knots[...,-2] * ((1-w_km1_3)*(1-w_km2_4)) + knots[...,-1]*((1-w_km1_3)*w_km2_4 + w_km1_3*(1-w_km1_4)) +  w_km1_3*w_km1_4

        a_coeff = 1 / (f_s - f_r)
        b_coeff = -f_r / (f_s - f_r)

        knots = F.pad(knots, pad=(0,1),mode='constant', value=1.)
        knots = F.pad(knots, pad=(0,1),mode='constant', value=0.)
        knots = a_coeff[...,None] * knots + b_coeff[...,None]

        knots = torch.roll(knots, shifts=-2, dims=-1)
        knots3 = torch.roll(knots, shifts=3, dims=-1)
        widths2 = torch.roll(widths, shifts=2, dims=-1)
        
        cumheights = torch.zeros_like(cumwidths)  # (num_bin + 1) dim initialize
        
        cumheights = knots3[..., 0:num_bins+1] * (torch.square(widths2[...,2:num_bins+3]) / (
            (widths2[..., 0:num_bins+1] +  widths2[..., 1:num_bins+2]+  widths2[..., 2:num_bins+3])
            * (widths2[..., 1:num_bins+2]+  widths2[..., 2:num_bins+3])
        )
                                                 ) \
        + knots3[..., 1:num_bins+2] * (
            ( widths2[..., 2:num_bins+3] * (widths2[..., 0:num_bins+1] +  widths2[..., 1:num_bins+2]))
            / (( widths2[..., 1:num_bins+2] +  widths2[..., 2:num_bins+3]) * (
                widths2[..., 0:num_bins+1] + widths2[..., 1:num_bins+2] + widths2[..., 2:num_bins+3]))
            + (widths2[..., 1:num_bins+2] * (widths2[..., 2:num_bins+3] +  widths2[..., 3:num_bins+4]))
            / ((widths2[..., 1:num_bins+2] + widths2[..., 2:num_bins+3]) * (
                widths2[..., 1:num_bins+2] + widths2[..., 2:num_bins+3] +  widths2[..., 3:num_bins+4]))
        ) \
        + knots3[..., 2:num_bins+3] * (
            torch.square(widths2[..., 1:num_bins+2]) / (
                (widths2[..., 1:num_bins+2] + widths2[..., 2:num_bins+3] +  widths2[..., 3:num_bins+4])
                * (widths2[..., 1:num_bins+2] + widths2[..., 2:num_bins+3])
            )
            )

        cumheights[..., 0] = 0.0
        cumheights[..., -1] = 1.0
        bin_idx = searchsorted(cumheights, inputs)[..., None]
        input_low_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        input_high_cumheights = cumheights.gather(-1, bin_idx + 1)[..., 0]

        i0 = bin_idx
        im1 = torch.remainder(bin_idx - 1, num_bins + 3)
        im2 = torch.remainder(bin_idx - 2, num_bins + 3)
        im3 = torch.remainder(bin_idx - 3, num_bins + 3)

        j3 = bin_idx + 3
        j2 = bin_idx + 2
        j1 = bin_idx + 1
        j0 = bin_idx
        jm1 = torch.remainder(bin_idx - 1, num_bins + 5)
        jm2 = torch.remainder(bin_idx - 2, num_bins + 5)

        km0 = knots.gather(-1, i0)[..., 0]
        km1 = knots.gather(-1, im1)[..., 0]
        km2 = knots.gather(-1, im2)[..., 0]
        km3 = knots.gather(-1, im3)[..., 0]

        t3 = t.gather(-1, j3)[..., 0]
        t2 = t.gather(-1, j2)[..., 0]
        t1 = t.gather(-1, j1)[..., 0]
        t0 = t.gather(-1, j0)[..., 0]
        tm1 = t.gather(-1, jm1)[..., 0]
        tm2 = t.gather(-1, jm2)[..., 0]
        
        input_left_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_right_cumwidths = cumwidths.gather(-1, bin_idx + 1)[..., 0]
        
        inputs_a1 = km0 * (
                1 / ((t3 - t0) * (t2 - t0) * (t1 - t0))
        ) + km1 * (
                            - 1 / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                            - 1 / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                            - 1 / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                    ) + km2 * (
                            1 / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                            + 1 / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                            + 1 / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                    ) + km3 * (
                            -1 / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                    )

        inputs_b1 = km0 * (
                (-3 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
        ) + km1 * (
                            (2 * tm1 + t1) / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                            + (tm1 + t2 + t0) / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                            + (t3 + 2 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                    ) + km2 * (
                            (-2 * t1 - tm2) / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                            + (-2 * t2 - t0) / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                            + (-t2 - t1 - tm1) / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                    ) + km3 * (
                            (3 * t1) / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                    )

        inputs_c1 = km0 * (
                (3 * t0 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
        ) + km1 * (
                            (- tm1 * tm1 - 2 * tm1 * t1) / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                            + (- tm1 * t2 - tm1 * t0 - t2 * t0) / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                            + (- t0 * t0 - 2 * t3 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                    ) + km2 * (
                            (t1 * t1 + 2 * t1 * tm2) / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                            + (t2 * t2 + 2 * t0 * t2) / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                            + (t2 * t1 + tm1 * t1 + t2 * tm1) / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                    ) + km3 * (
                            (-3 * t1 * t1) / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                    )

        inputs_d1 = km0 * (
                (- t0 * t0 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
        ) + km1 * (
                            (tm1 * tm1 * t1) / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                            + (tm1 * t2 * t0) / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                            + (t3 * t0 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                    ) + km2 * (
                            - (t1 * t1 * tm2) / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                            - (t0 * t2 * t2) / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                            - (t2 * tm1 * t1) / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                    ) + km3 * (
                            (t1 * t1 * t1) / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                    )            
    
        inputs = torch.as_tensor(inputs, dtype=torch.double)
        outputs = torch.zeros_like(inputs)
        inputs_b_ = torch.as_tensor(inputs_b1 / inputs_a1 / 3., dtype=torch.double, device=inputs_b1.device)
        inputs_c_ = torch.as_tensor(inputs_c1 / inputs_a1 / 3., dtype=torch.double, device=inputs_b1.device)
        inputs_d_ = torch.as_tensor((inputs_d1 - inputs) / inputs_a1, dtype=torch.double, device=inputs_b1.device)
        delta_1 = -inputs_b_.pow(2) + inputs_c_
        delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
        delta_3 = inputs_b_ * inputs_d_ - inputs_c_.pow(2)

        discriminant = 4. * delta_1 * delta_3 - delta_2.pow(2)

        depressed_1 = -2. * inputs_b_ * delta_1 + delta_2
        depressed_2 = delta_1

        three_roots_mask = discriminant >= 0  # Discriminant == 0 might be a problem in practice.
        one_root_mask = discriminant < 0

        # Deal with one root cases.
        p_ = torch.zeros_like(inputs)
        p_[one_root_mask] = cbrt((-depressed_1[one_root_mask] + sqrt(-discriminant[one_root_mask])) / 2.)

        p = p_[one_root_mask]
        q = cbrt((-depressed_1[one_root_mask] - sqrt(-discriminant[one_root_mask])) / 2.)

        outputs_one_root = ((p + q) - inputs_b_[one_root_mask])

        outputs[one_root_mask] = torch.as_tensor(outputs_one_root, dtype=outputs.dtype)

        # Deal with three root cases.

        theta = torch.atan2(sqrt(discriminant[three_roots_mask]), -depressed_1[three_roots_mask])
        theta /= 3.

        cubic_root_1 = torch.cos(theta)
        cubic_root_2 = torch.sin(theta)

        root_1 = cubic_root_1
        root_2 = -0.5 * cubic_root_1 - 0.5 * math.sqrt(3) * cubic_root_2
        root_3 = -0.5 * cubic_root_1 + 0.5 * math.sqrt(3) * cubic_root_2

        root_scale = 2 * sqrt(-depressed_2[three_roots_mask])
        root_shift = -inputs_b_[three_roots_mask]

        root_1 = root_1 * root_scale + root_shift
        root_2 = root_2 * root_scale + root_shift
        root_3 = root_3 * root_scale + root_shift

        root1_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_1).float()
        root1_mask *= (root_1 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        root2_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_2).float()
        root2_mask *= (root_2 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        root3_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_3).float()
        root3_mask *= (root_3 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        roots = torch.stack([root_1, root_2, root_3], dim=-1)

        masks = torch.stack([root1_mask, root2_mask, root3_mask], dim=-1)
        mask_index = torch.argsort(masks, dim=-1, descending=True)[..., 0][..., None]
        output_three_roots = torch.gather(roots, dim=-1, index=mask_index).view(-1)
        outputs[three_roots_mask] = torch.as_tensor(output_three_roots, dtype=outputs.dtype)

        # Deal with a -> 0 (almost quadratic) cases.

        quadratic_mask = inputs_a1.abs() < quadratic_threshold
        a = inputs_b1[quadratic_mask]
        b = inputs_c1[quadratic_mask]
        c = (inputs_d1[quadratic_mask] - inputs[quadratic_mask])
        alpha = (-b + sqrt(b.pow(2) - 4 * a * c)) / (2 * a)
        outputs[quadratic_mask] = torch.as_tensor(alpha, dtype=outputs.dtype)  # + input_left_cumwidths[quadratic_mask]

        # Deal with b-> 0 (almost linear) cases.
        linear_mask = inputs_b1.abs() < linear_threshold
        linear_mask = linear_mask * quadratic_mask
        b = inputs_c1[linear_mask]
        c = (inputs_d1[linear_mask] - inputs[linear_mask])
        alpha = c / b
        outputs[linear_mask] = torch.as_tensor(alpha, dtype=outputs.dtype)

        outputs = torch.clamp(outputs, input_left_cumwidths, input_right_cumwidths)
        logabsdet = -torch.log(
            (torch.abs(
                (3 * inputs_a1 * outputs.pow(2)
                 + 2 * inputs_b1 * outputs
                 + inputs_c1))
            )
        )
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - math.log(top - bottom) + math.log(right - left)
        outputs = torch.as_tensor(outputs, dtype=torch.float)
        logabsdet = torch.as_tensor(logabsdet, dtype=torch.float)        
    else:
        inputs = (inputs - left) / (right - left)

        #inputs = torch.clamp(inputs, 1e-6, 1 - (1e-6))

        widths = torch.softmax(unnormalized_widths, dim=-1)  # num_bins+4 dim initialize
        widths = min_bin_width + (1 - (num_bins + 4) * min_bin_width) * widths
        widths = (widths / torch.sum(widths[..., 0:num_bins], dim=-1, keepdim=True))
        #widths = torch.as_tensor(widths, dtype = torch.double)
        cumwidths = torch.cumsum(widths[..., 0:num_bins], dim=-1)
        cumwidths[..., -1] = 1.
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)

        t = F.pad(cumwidths, pad=(0, 4), mode='constant', value=0.0)  # num_bins + 5 dim initialize

        t[..., -1] = t[..., 0] - widths[..., -1]
        t[..., -2] = t[..., -1] - widths[..., -2]
        t[..., num_bins + 1] = t[..., num_bins] + widths[..., num_bins]
        t[..., num_bins + 2] = t[..., num_bins + 1] + widths[..., num_bins + 1]


        var2 = torch.softmax(unnormalized_var1[...,0:num_bins+2], dim=-1)  # num_bins + 2 dim initialize
        var2 = min_bin_height + (1 - (num_bins+2) * min_bin_height) * var2

        knots = torch.cumsum(var2, dim=-1) # r-2,  ... , s-1
        knots = knots[...,:-1]
        #knots = torch.as_tensor(knots, dtype=torch.double)

        w_m1_3 = widths[...,-1] / (widths[...,-1] + widths[...,0])
        w_m1_4 = widths[...,-1] / (widths[...,-1] + widths[...,0] + widths[...,1])
        w_m2_4 = (widths[...,-1] + widths[...,-2]) / (widths[...,-2] + widths[...,-1] + widths[...,0])
        w_km1_3 = widths[...,num_bins-1] / (widths[...,num_bins-1] + widths[...,num_bins])
        w_km1_4 = widths[...,num_bins-1] / (widths[...,num_bins-1] + widths[...,num_bins] + widths[...,num_bins+1])
        w_km2_4 = (widths[...,num_bins-1] + widths[...,num_bins-2]) / (widths[...,num_bins-2] + widths[...,num_bins-1] + widths[...,num_bins])

        f_r = knots[...,0] * ( (1-w_m1_3)*w_m2_4 + w_m1_3*(1-w_m1_4)) + knots[...,1]*w_m1_3*w_m1_4
        f_s = knots[...,-2] * ((1-w_km1_3)*(1-w_km2_4)) + knots[...,-1]*((1-w_km1_3)*w_km2_4 + w_km1_3*(1-w_km1_4)) +  w_km1_3*w_km1_4

        a_coeff = 1 / (f_s - f_r)
        b_coeff = -f_r / (f_s - f_r)

        knots = F.pad(knots, pad=(0,1),mode='constant', value=1.)
        knots = F.pad(knots, pad=(0,1),mode='constant', value=0.)
        knots = a_coeff[...,None] * knots + b_coeff[...,None]

        knots = torch.roll(knots, shifts=-2, dims=-1)
        knots3 = torch.roll(knots, shifts=3, dims=-1)
        widths2 = torch.roll(widths, shifts=2, dims=-1)
        
        cumheights = torch.zeros_like(cumwidths)  # (num_bin + 1) dim initialize
        
        cumheights = knots3[..., 0:num_bins+1] * (torch.square(widths2[...,2:num_bins+3]) / (
            (widths2[..., 0:num_bins+1] +  widths2[..., 1:num_bins+2]+  widths2[..., 2:num_bins+3])
            * (widths2[..., 1:num_bins+2]+  widths2[..., 2:num_bins+3])
        )
                                                 ) \
        + knots3[..., 1:num_bins+2] * (
            ( widths2[..., 2:num_bins+3] * (widths2[..., 0:num_bins+1] +  widths2[..., 1:num_bins+2]))
            / (( widths2[..., 1:num_bins+2] +  widths2[..., 2:num_bins+3]) * (
                widths2[..., 0:num_bins+1] + widths2[..., 1:num_bins+2] + widths2[..., 2:num_bins+3]))
            + (widths2[..., 1:num_bins+2] * (widths2[..., 2:num_bins+3] +  widths2[..., 3:num_bins+4]))
            / ((widths2[..., 1:num_bins+2] + widths2[..., 2:num_bins+3]) * (
                widths2[..., 1:num_bins+2] + widths2[..., 2:num_bins+3] +  widths2[..., 3:num_bins+4]))
        ) \
        + knots3[..., 2:num_bins+3] * (
            torch.square(widths2[..., 1:num_bins+2]) / (
                (widths2[..., 1:num_bins+2] + widths2[..., 2:num_bins+3] +  widths2[..., 3:num_bins+4])
                * (widths2[..., 1:num_bins+2] + widths2[..., 2:num_bins+3])
            )
            )

        cumheights[..., 0] = 0.0
        cumheights[..., -1] = 1.0
        
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

        i0 = bin_idx
        im1 = torch.remainder(bin_idx - 1, num_bins + 3)
        im2 = torch.remainder(bin_idx - 2, num_bins + 3)
        im3 = torch.remainder(bin_idx - 3, num_bins + 3)

        j3 = bin_idx + 3
        j2 = bin_idx + 2
        j1 = bin_idx + 1
        j0 = bin_idx
        jm1 = torch.remainder(bin_idx - 1, num_bins + 5)
        jm2 = torch.remainder(bin_idx - 2, num_bins + 5)

        km0 = knots.gather(-1, i0)[..., 0]
        km1 = knots.gather(-1, im1)[..., 0]
        km2 = knots.gather(-1, im2)[..., 0]
        km3 = knots.gather(-1, im3)[..., 0]

        t3 = t.gather(-1, j3)[..., 0]
        t2 = t.gather(-1, j2)[..., 0]
        t1 = t.gather(-1, j1)[..., 0]
        t0 = t.gather(-1, j0)[..., 0]
        tm1 = t.gather(-1, jm1)[..., 0]
        tm2 = t.gather(-1, jm2)[..., 0]

        input_left_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_right_cumwidths = cumwidths.gather(-1, bin_idx + 1)[..., 0]

        w_j_2 = (inputs - t0) / (t1 - t0)
        w_j_3 = (inputs - t0) / (t2 - t0) # (x-t_j)/(t_j+2 - t_j)
        w_jm1_3 = (inputs - tm1) / (t1 - tm1)

        B_jm2_3 = (1-w_jm1_3)*(1-w_j_2)
        B_jm1_3 = w_jm1_3 * (1-w_j_2) + (1-w_j_3) * w_j_2
        B_j_3 = w_j_3 * w_j_2
        D_jm2_3 = (km2-km3) / (t1-tm2) 
        D_jm1_3 = (km1-km2) / (t2-tm1) 
        D_j_3 = (km0-km1) / (t3-t0) 
        
        absdet = 3*(D_jm2_3*B_jm2_3+D_jm1_3*B_jm1_3+D_j_3*B_j_3)
        logabsdet = torch.log(absdet)
        
        outputs = (km3 + (inputs - tm2)*D_jm2_3)*B_jm2_3 + (km2 + (inputs - tm1)*D_jm1_3)*B_jm1_3 + (km1 + (inputs - t0)*D_j_3)*B_j_3
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)

    
    r = random.randint(0, 1000)
    if r == 0:
        cd = random.randint(0, 10)
        plot_B_spline(knots[cd, 0, :].cpu().detach(), t[cd, 0, :].cpu().detach(),
                      cumwidths[cd, 0, :].cpu().detach(), cumheights[cd, 0, :].cpu().detach(),
                      num_bins, eps, linear_threshold, quadratic_threshold, inverse=inverse)
    
    return outputs, logabsdet
