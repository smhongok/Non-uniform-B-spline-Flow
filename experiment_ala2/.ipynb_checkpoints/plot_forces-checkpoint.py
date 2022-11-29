#%load_ext autoreload
#%autoreload 2
import numpy as np
import torch
import time
from bgforces.ala2 import Ala2Generator
from matplotlib import pyplot as plt

TRANSFORMATIONS = ["RQspline", "Smooth", "Smooth+FM", "Bspline", "Bspline+FM"]
name = ["\n RQ-spline\n", "\n Smooth\n", "\n Smooth+FM\n", "Non-uniform\n B-spline\n (ours)", "Non-uniform\n B-spline+FM\n (ours)"]
fig, axes = plt.subplots(2,5,figsize=(8/3*5,4.5), sharex="col")
fig.tight_layout()
plt.rc('font', size=15)
#plt.yticks(fontsize=14)
#plt.xticks(fontsize=14)
for i in range(5):
    j=0
    TRANSFORMATION = TRANSFORMATIONS[i]
    MODEL = TRANSFORMATION+ str(j)
    CHECKPOINT = "logs/a2_1/{0}/version_{1}/checkpoints/epoch=9-step=70320.ckpt".format(TRANSFORMATION,j)
    loaded = Ala2Generator.load_from_checkpoint(
        CHECKPOINT
    )

    sum( np.prod(p.shape) for p in loaded.parameters())

    from bgforces.ala2 import Ala2Data
    data = Ala2Data(batch_size=128, slice=10)
    data.prepare_data()
    data.setup()
    
    data_smooth = np.load(f"data_samples_{MODEL.lower()}.npz")
    bg_smooth = np.load(f"bg_samples_{MODEL.lower()}.npz")

    

    data_mse = ((data_smooth["model_forces"] - data_smooth["target_forces"])**2).mean()
    np.sqrt(data_mse)

    bg_mse = ((bg_smooth["model_forces"] - bg_smooth["target_forces"])**2).mean()
    np.sqrt(bg_mse)

    def plot_energies(ax, d):
        md_energies = d["target_energies"].flatten()
        sample_energies = d["model_energies"].flatten()
        #md_energies -= md_energies.min()
        sample_energies -= (sample_energies.min() - md_energies.min())

        cut = max(np.percentile(sample_energies, 80), 20)

        # y-axis on the right
        ax2 = plt.twinx(ax)
        ax.get_yaxis().set_visible(False)

        ax2.hist(
            sample_energies, range=(-50, cut), bins=50, 
            density=False, label="Target", fc=(1.0, 0, 0, 0.6)
        )
        ax2.hist(
            md_energies, range=(-50, cut), bins=50, 
            density=False, label="Flow", fc=(0,0,0.8,0.2), lw=1, ec="black"
        )
        ax2.set_ylabel(f"Count   [#Samples / 10000]")
        ax2.legend()

    data_smooth["target_energies"].min() - data_smooth["model_energies"].min()
    bg_smooth["target_energies"].min() - bg_smooth["model_energies"].min()

    from bgforces.ala2.plot import plot_phi_psi
    from scipy.stats import gaussian_kde

    

    ax = axes[0,i]
    ax.scatter(data_smooth["model_forces"], data_smooth["target_forces"], s=0.1, color="blue")
    ax.set_xlim(-2800., 2800.)
    ax.set_ylim(-2800., 2800.)
    ax.set_title(name[i])
    if i==0:
        ax.set_ylabel("Density Estimation \n Forces [$k_B T$/nm]",fontsize=15)#, labelpad=-60)
        #ax.axes.yaxis.set_tick
    else:
        ax.axes.yaxis.set_ticklabels([])

    ax = axes[1,i]
    ax.scatter(bg_smooth["model_forces"], bg_smooth["target_forces"], s=0.1, color="red")
    ax.set_xlim(-2800., 2800.)
    ax.set_ylim(-2800., 2800.)
    ax.set_xlabel("Target Forces [$k_B T$/nm]",fontsize=15)
    if i==0:
        ax.set_ylabel("Sampling \n Forces [$k_B T$/nm]",fontsize=15)#, labelpad=-60)
    else:
        ax.axes.yaxis.set_ticklabels([])
#text1 = axes[0,0].text(1, 0, "DENSITY \n ESTIMATION", rotation=90, fontsize=15, va="center")
#text2 = axes[1,0].text(0, 1, "SAMPLING", rotation=90, fontsize=15, va="center")
fig.savefig("forces.png", dpi=360, bbox_inches="tight")
fig.savefig("forces.pdf", dpi=360, bbox_inches="tight")



