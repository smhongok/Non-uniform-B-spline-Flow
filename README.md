# [AAAI 2023 Oral] Non-uniform B-spline Flows
The final version will be uploaded soon.
Our paper (proceeding) has not been published yet, but most of the codes in this repository are based on the implementation of Smooth normalizing flows (Köhler, Krämer, and Noé 2021, https://proceedings.neurips.cc/paper/2021/hash/167434fa6219316417cd4160c0c5e7d2-Abstract.html).
If you're interested in the Non-uniform B-spline flow implementation, you can find it in /bgflow/bgflow/nn/flow/transformer/bspline.py .

## Install
-  Download data (follow instructions in ./bgmol/bgmol/data/README.md).
- `conda env create -f condaenv.yml`
- `conda activate nubsf`
- `cd bgflow && python setup.py install && cd -`
- `cd bgmol  && python setup.py install && cd -`
- `cd bgforces  && python setup.py install && cd -`


## Experiments


### 2D Toy Examples
See notebooks in the `experiment_toy2d` directory.
You can get Figure 1,2 and 5 by running those notebooks.

### Alanine Dipeptide examples
Training is done via the `train.py` script in the `experiment_ala2` directory.
Call `python train.py --help` to see the available options.

Here is an example:

- `python train.py --transformer-type=bspline --activation-type=sin --max_epochs=10 --devices=gpus --gpus=1 --nll-weight=0.999 --force-weight=0.001` 

The plotting and analysis uses checkpoint files written by the train.py script.
In each notebook, the `CHECKPOINT` variable has to be set to the checkpoint path.

We provide one checkpoint for each scenario. (5 in total)
Check `experiment_toy2d/logs/a2_1` directory.

To reproduce our result, there should be 10 checkpoints for each scenario. (50 in total)

You can get Table 1,2, Figure 7,8,9 by running `plot_ala2_model.py`.
You can get Figure 3 by running `plot_forces.py`.
You can get Figure 4 running `simulate.py`.
