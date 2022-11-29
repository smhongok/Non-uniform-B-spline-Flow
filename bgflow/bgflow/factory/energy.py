
from bgflow.utils.train import linlogcut
from bgflow.distribution.energy import Energy


__all__ = ["LinLogCutEnergy"]


class LinLogCutEnergy(Energy):
    def __init__(self, delegate, high_energy=1e3, max_energy=1e9):
        super().__init__(delegate.dim)
        self.delegate = delegate
        self.high_energy = high_energy
        self.max_energy = max_energy
    
    def _energy(self, x, temperature=None):
        u = self.delegate.energy(x[:, :self.delegate.dim])
        if self.high_energy is not None or self.max_energy is not None:
            return linlogcut(u, high_val=self.high_energy, max_val=self.max_energy, inplace=True)
        else:
            return u


