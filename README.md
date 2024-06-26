# [AAAI 2023 Oral] Non-uniform B-spline Flows
Official repository of a paper 'Non-uniform B-spline Flows'
by <a href="https://smhongok.github.io/">Seongmin Hong</a> and <a href="https://icl.snu.ac.kr/pi">Se Young Chun</a>.

Link: <a href="https://smhongok.github.io/nubsf.html">Project webpage</a>, <a href="https://arxiv.org/abs/2304.04555">arXiv</a>, <a href="https://ojs.aaai.org/index.php/AAAI/article/view/26441">Paper</a>

Most of the codes in this repository are based on the implementation of <a href="https://openreview.net/pdf?id=yxsak5ND2pA">Smooth normalizing flows</a> (Köhler, Krämer, and Noé, in NeurIPS 2021).
If you're interested in the Non-uniform B-spline flow implementation, you can find it in `/bgflow/bgflow/nn/flow/transformer/bspline.py`.

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
