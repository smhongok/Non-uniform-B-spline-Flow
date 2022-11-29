
import torch
from simtk.unit import MOLAR_GAS_CONSTANT_R, kelvin, kilojoules_per_mole, amu, nanometer, picosecond


__all__ = ["FlowPotentialWrapper", "Integrator", "VerletIntegrator", "LangevinIntegrator", "Simulation"]

class FlowPotentialWrapper(torch.nn.Module):
    def __init__(self, flow, temperature):
        super().__init__()
        self.flow = flow
        self.flow.train(False)
        self.temperature = temperature
        kbt = (temperature*kelvin * MOLAR_GAS_CONSTANT_R)
        self.kbt = kbt.value_in_unit(kilojoules_per_mole)
        
    def force(self, x):
        n_particles = x.shape[-2]
        force = self.flow.force(x.view(-1, 3*n_particles)) * self.kbt
        return force.view(-1, n_particles, 3)
    
    def energy(self, x):
        n_particles = x.shape[-2]
        energy = self.flow.energy(x.view(-1, 3*n_particles)) * self.kbt
        return energy.flatten()
        

class Integrator(torch.nn.Module):
    def __init__(self, potential, time_step, masses):
        """time step in ps; masses in amu"""
        super().__init__()
        self.potential = potential
        self.time_step = time_step
        self.masses = masses
    
    def kinetic_energy(self, v):
        """in kJ/mol; amu * nm**2 / ps**2 = kJ/mol"""
        return 0.5*(self.masses[...,None]*v**2).sum(dim=-1).sum(dim=-1)
    
    def remove_center_of_mass_movement(self, v):
        com_momentum = (self.masses[..., None])*v.mean(dim=-2, keepdim=True)
        com_v = com_momentum / self.masses.sum()
        return v - com_v
                    
    def set_velocities_to_temperature(self, temperature, x):
        randoms = torch.randn_like(x)
        kbt = (MOLAR_GAS_CONSTANT_R * temperature * kelvin).value_in_unit(kilojoules_per_mole)
        return randoms * (kbt / self.masses[..., None]).sqrt()  
        
    
    
class VerletIntegrator(Integrator):
    def __init__(self, potential, time_step, masses):
        super().__init__(potential, time_step, masses)
    
    def forward(self, x, v):
        v = v + 0.5 * self.time_step * self.potential.force(x) / self.masses[..., None]
        x = x + self.time_step * v
        v = v + 0.5 * self.time_step * self.potential.force(x) / self.masses[..., None]
        #v = self.remove_center_of_mass_movement(v)
        return x, v
        
        
class LangevinIntegrator(Integrator):
    def __init__(self, potential, time_step, masses, temperature=300., friction=10.):
        """time step in ps; masses in amu"""
        super().__init__(potential, time_step, masses)
        self.temperature = temperature
        self.friction = friction
        kbt = (MOLAR_GAS_CONSTANT_R * temperature * kelvin).value_in_unit(kilojoules_per_mole)
        self.vcoeff = torch.sqrt(
            2.0 * self.friction / self.masses[...,None] 
            * kbt * self.time_step
        )
    
    def forward(self, x, v):
        v = v + 0.5 * self.time_step * self.potential.force(x) / self.masses[..., None]
        csi = torch.randn_like(v) * self.vcoeff
        v = v - self.friction * v * self.time_step + csi
        x = x + self.time_step * v
        v = v + 0.5 * self.time_step * self.potential.force(x) / self.masses[..., None]
        return x, v
    
        
class Simulation:
    def __init__(self, potential, integrator, x0, v0=None, temperature=None, report_interval=10):
        self.integrator = integrator
        self.potential = potential
        self.x = x0.clone()
        if v0 is None:
            assert temperature is not None
            self.v = integrator.set_velocities_to_temperature(temperature, x0)
        else:
            self.v = v0.clone()
        self.x.requires_grad = False
        self.v.requires_grad = False
        self.i = 0
        self.report_interval = report_interval
        self.energies = []
   
    def step(self, n_steps):
        with torch.no_grad():
            for i in range(n_steps):
                self.i += 1
                self.x, self.v = self.integrator(self.x, self.v)
                if self.i % self.report_interval == 0:
                    self._report(self.x, self.v) 
        
    def _report(self, x, v):
        kinetic_energy = self.integrator.kinetic_energy(v).cpu().numpy()
        potential_energy = self.potential.energy(x).cpu().numpy()
        self.energies.append([
            kinetic_energy, 
            potential_energy, 
            (kinetic_energy + potential_energy)
        ])
        
    
