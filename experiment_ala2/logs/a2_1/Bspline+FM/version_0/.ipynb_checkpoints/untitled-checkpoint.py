import torch
from matplotlib import pyplot as plt
import numpy as np
from bgforces.util import FlowPotentialWrapper
from bgforces.util import VerletIntegrator, Simulation, LangevinIntegrator
from bgforces.ala2 import Ala2Generator
from bgforces.ala2 import Ala2Data
from simtk.unit import amu

for TRANSFORMATION in ["RQspline", "Smooth", "Smooth+FM", "Bspline", "Bspline+FM"]:
    for j in range(1):
        print(TRANSFORMATION)
        MODEL = TRANSFORMATION+ str(j)
        CHECKPOINT = "logs/a2_1/{0}/version_{1}/checkpoints/epoch=9-step=70320.ckpt".format(TRANSFORMATION,j)
        loaded = Ala2Generator.load_from_checkpoint(
            CHECKPOINT
        )
        
        data = Ala2Data(batch_size=128, slice=1)
        data.prepare_data()
        data.setup()


        masses = []

        for atom in data.dataset.system.topology.atoms():
            masses.append(atom.element.mass.value_in_unit(amu))

        masses = torch.tensor(masses)

        x0=torch.tensor(data.dataset.xyz[:10])
        #x0.shape, masses.shape
        # flow_potential.force(x0)


        potential = FlowPotentialWrapper(loaded.model, data.dataset.temperature)


        npt_integrator = LangevinIntegrator(potential, 0.0001, masses=masses, temperature=300.)
        npt_simulation = Simulation(potential, npt_integrator, x0=torch.tensor(data.dataset.xyz[:10]), temperature=data.dataset.temperature)

        npt_simulation.step(50000)

        nve_integrator = VerletIntegrator(potential, 0.0001, masses=masses)
        nve_simulation = Simulation(
            potential, 
            nve_integrator,
            x0=npt_simulation.x,
            v0=npt_simulation.v
        )
        nve_simulation.step(50000)



        fig, ax = plt.subplots(figsize=(3,3))

        energies = np.array(nve_simulation.energies)
        dt = nve_simulation.report_interval * nve_simulation.integrator.time_step
        times = np.arange(len(energies))*dt
        ax.plot(times, energies[:,2,:])
        ax.set_xlim(0.0, 5.0)#times[-1]*1.0)
        ax.set_xlabel("t [ps]")
        ax.set_ylabel("Total Energy [kJ/mol]")
        
        fig.savefig(f"{MODEL.lower()}_simulation.pdf", dpi=360, bbox_inches="tight")
        fig.savefig(f"{MODEL.lower()}_simulation.pdf", dpi=360, bbox_inches="tight")