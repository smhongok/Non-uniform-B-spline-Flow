import numpy as np
import bgflow as bg
from collections import OrderedDict, namedtuple

__all__ = ["TensorInfo", "ShapeInfo", "BONDS", "ANGLES", "TORSIONS", "FIXED", "ORIGIN", "ROTATION", "AUGMENTED",
           "TARGET"]


TensorInfo = namedtuple(
    "TensorInfo",
    ["name", "is_circular"],
    defaults=(False, )
)

BONDS = TensorInfo("BONDS", False)
ANGLES = TensorInfo("ANGLES", False)
TORSIONS = TensorInfo("TORSIONS", True)
FIXED = TensorInfo("FIXED", False)  # in relative/mixed trafo
ORIGIN = TensorInfo("ORIGIN", False)  # in global trafo
ROTATION = TensorInfo("ROTATION", False)
AUGMENTED = TensorInfo("AUGMENTED", False)
TARGET = TensorInfo("TARGET", False)


class ShapeInfo(OrderedDict):
    # TODO: support multiple event dimensions
    def __init__(self):
        super().__init__()

    @staticmethod
    def from_coordinate_transform(coordinate_transform, dim_augmented=0):
        shape_info = ShapeInfo()
        if coordinate_transform.dim_angles > 0:
            shape_info[BONDS] = (coordinate_transform.dim_bonds, )
        if coordinate_transform.dim_angles > 0:
            shape_info[ANGLES] = (coordinate_transform.dim_angles, )
        if coordinate_transform.dim_torsions > 0:
            shape_info[TORSIONS] = (coordinate_transform.dim_torsions, )
        if coordinate_transform.dim_fixed > 0:
            shape_info[FIXED] = (coordinate_transform.dim_fixed, )
        if dim_augmented > 0:
            shape_info[AUGMENTED] = (dim_augmented, )
        if isinstance(coordinate_transform, bg.GlobalInternalCoordinateTransformation):
            shape_info[ORIGIN] = (1, 3)
            shape_info[ROTATION] = (1, 3, 3)
        return shape_info

    def split(self, field, into, sizes, dim=-1):
        # remove one
        index = self.index(field)
        if not sum(sizes) == self[field][dim]:
            raise ValueError(f"split sizes {sizes} do not sum up to total ({self[field]})")
        all_sizes = list(self[field])
        del self[field]
        # insert multiple
        for f in into:
            assert f not in self
        for el, size in zip(reversed(into), reversed(sizes)):
            all_sizes[dim] = size
            self.insert(el, index, tuple(all_sizes))

    def merge(self, fields, to, index=None, dim=-1):
        # remove multiple
        size = sum(self[f][dim] for f in fields)
        all_sizes = list(self[fields[0]])
        # TODO: check that other dimensions are compatible
        all_sizes[dim] = size
        first_index = min(self.index(f) for f in fields)
        for f in fields:
            del self[f]
        # insert one
        assert to not in self
        if index is None:
            index = first_index
        self.insert(to, index, tuple(all_sizes))

    def replace(self, field, other):
        if isinstance(other, str):
            other = field._replace(name=other)
        self.insert(other, self.index(field), self[field])
        del self[field]
        return other

    def copy(self):
        clone = ShapeInfo()
        for field in self:
            clone[field] = self[field]
        return clone

    def insert(self, field, index, size):
        if index < 0:
            index = len(self) - index
        assert field not in self
        self[field] = size  # append
        for i, key in enumerate(list(self)):
            if index <= i < len(self) -1:
                self.move_to_end(key)

    def index(self, field, fields=None):
        fields = self if fields is None else fields
        return list(fields).index(field)

    def names(self, fields=None):
        fields = self if fields is None else fields
        return (field.name for field in fields)

    def dim_all(self, fields=None, dim=-1):
        fields = self if fields is None else fields
        return sum(self[field][dim] for field in fields)

    def dim_circular(self, fields=None, dim=-1):
        fields = self if fields is None else fields
        return sum(self[field][dim] for field in fields if field.is_circular)

    def dim_noncircular(self, fields=None, dim=-1):
        fields = self if fields is None else fields
        return sum(self[field][dim] for field in fields if not field.is_circular)

    def is_circular(self, fields=None, dim=-1):
        fields = self if fields is None else fields
        return np.concatenate([np.ones(self[field][dim])*field.is_circular for field in fields]).astype(bool)

    def circular_indices(self, fields=None, dim=-1):
        fields = self if fields is None else fields
        return np.arange(self.dim_all(fields, dim))[self.is_circular(fields, dim)]

