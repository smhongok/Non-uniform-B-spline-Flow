#%load_ext autoreload
#%autoreload 2
import numpy as np
import torch
import time
from bgforces.ala2 import Ala2Generator

TRANSFORMATIONS = ["RQspline", "Smooth", "Smooth+FM", "Bspline", "Bspline+FM"]
for i in range(5):
    NLLS =[]
    FMES =[]
    KLDS =[]
    S_times = []
    DE_times = []
    for j in range(1):
        TRANSFORMATION = TRANSFORMATIONS[i]
        print(TRANSFORMATION)
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

        tolist = lambda x: list(x.detach().numpy())

        from tqdm.notebook import tqdm

        class BG:
            samples = []
            model_energies = []
            target_energies = []
            model_forces = []
            target_forces = []
            start = time.time()  # 시작 시간 저장
            for i in tqdm(range(100)):
                _samples = loaded.model.sample(100)
                samples.append(tolist(_samples))
                model_energies.append(tolist(loaded.model.energy(_samples)[:,0]))
                target_energies.append(tolist(loaded.model._target.energy(_samples)[:,0]))
                model_forces.append(tolist(loaded.model.force(_samples)))
                target_forces.append(tolist(loaded.model._target.force(_samples)))
            sampling_time = (time.time() - start)/10
            print("sampling time per sample (ms):", sampling_time)
            S_times.append(sampling_time)

        np.savez(
            f"bg_samples_{MODEL.lower()}.npz", 
            samples=np.array(BG.samples),
            model_energies=np.array(BG.model_energies),
            target_energies=np.array(BG.target_energies),
            model_forces=np.array(BG.model_forces),
            target_forces=np.array(BG.target_forces)
        )

        class DATA:
            samples = []
            model_energies = []
            target_energies = []
            model_forces = []
            target_forces = []
            start = time.time()
            for i in tqdm(range(100)):
                _samples = data.val_xyz[100*i:100*(i+1)]
                samples.append(tolist(_samples))
                model_energies.append(tolist(loaded.model.energy(_samples)[:,0]))
                target_energies.append(tolist(loaded.model._target.energy(_samples)[:,0]))
                model_forces.append(tolist(loaded.model.force(_samples)))
                target_forces.append(tolist(loaded.model._target.force(_samples)))
            density_estimation_time = (time.time() - start)/10
            print("density estimation time per sample (ms):", density_estimation_time)
            DE_times.append(density_estimation_time)
        np.savez(
            f"data_samples_{MODEL.lower()}.npz", 
            samples=np.array(DATA.samples),
            model_energies=np.array(DATA.model_energies),
            target_energies=np.array(DATA.target_energies),
            model_forces=np.array(DATA.model_forces),
            target_forces=np.array(DATA.target_forces)
        )

        data_smooth = np.load(f"data_samples_{MODEL.lower()}.npz")
        bg_smooth = np.load(f"bg_samples_{MODEL.lower()}.npz")

        from matplotlib import pyplot as plt

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

        fig, axes = plt.subplots(2,3,figsize=(8,4.5), sharex="col")
        fig.tight_layout()

        plot_phi_psi(axes[0,0], torch.tensor(data_smooth["samples"]), data.dataset.system)
        text1 = axes[0,0].text(-6, 0, "TEST SET (MD)", rotation=90, fontsize=12, va="center")
        plot_phi_psi(axes[1,0], torch.tensor(bg_smooth["samples"]), data.dataset.system)
        text2 = axes[1,0].text(-6, 0, "BG SAMPLES", rotation=90, fontsize=12, va="center")

        ax = axes[0,1]
        ax.scatter(data_smooth["model_forces"], data_smooth["target_forces"], s=0.1, color="blue")
        ax.set_xlim(-2800., 2800.)
        ax.set_ylim(-2800., 2800.)
        ax.set_ylabel("Flow Forces [$k_B T$/nm]", labelpad=-60)

        ax = axes[1,1]
        ax.scatter(bg_smooth["model_forces"], bg_smooth["target_forces"], s=0.1, color="red")
        ax.set_xlim(-2800., 2800.)
        ax.set_ylim(-2800., 2800.)
        ax.set_xlabel("Target Forces [$k_B T$/nm]")
        ax.set_ylabel("Flow Forces [$k_B T$/nm]", labelpad=-60)

        plot_energies(axes[0,2], data_smooth)
        plot_energies(axes[1,2], bg_smooth)
        axes[1,2].set_xlabel("Energy   [$k_B T$]")

        fig.savefig(f"{MODEL.lower()}_bg.png", dpi=360, extra_bbox_agents=(text1, text2), bbox_inches="tight")
        fig.savefig(f"{MODEL.lower()}_bg.pdf", dpi=360, extra_bbox_agents=(text1, text2), bbox_inches="tight")

        nll = data_smooth["model_energies"].mean()
        fme = ((data_smooth["model_forces"] - data_smooth["target_forces"])**2).mean()/1e4
        kld = loaded._kld_for_samples(1000).item()
        print("NLL = " + str(nll))
        print("FME (1e4) = " + str(fme))
        print("KLD = " + str(kld))
        NLLS.append(nll)
        FMES.append(fme)
        KLDS.append(kld)
        print(np.mean(NLLS),np.std(NLLS))
        #print(S_times)
        #print(KLDS)
    print(NLLS)
    print(FMES)
    print(KLDS)
    print(S_times)
    print(DE_times)
    print(np.mean(NLLS),2*np.std(NLLS,ddof=1)/np.sqrt(10))
    print(np.mean(FMES),2*np.std(FMES,ddof=1)/np.sqrt(10))
    print(np.mean(KLDS),2*np.std(KLDS,ddof=1)/np.sqrt(10))
    print(np.mean(S_times),2*np.std(S_times,ddof=1)/np.sqrt(10))
    print(np.mean(DE_times),2*np.std(DE_times,ddof=1)/np.sqrt(10))


