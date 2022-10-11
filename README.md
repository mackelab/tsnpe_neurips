# Truncated proposals for scalable and hassle-free simulation-based inference
Code to reproduce results from `Truncated proposals for scalable and hassle-free simulation-based inference` by Michael Deistler, Pedro J. Goncalves*, Jakob H. Macke*, accepted at NeurIPS 2022.

### Structure of this repository
This repo contains four main folders. Each folder contains an individual `README.md` file which describes how the particular results can be reproduced. The folder `benchmark` contains code to run the benchmark tasks (Fig. 4). The folder `l5pc` contains code to produce the results for the pyloric network (Fig. 5) and for the L5PC (Fig. 6). The folder `sbi` is a modified version of the [`sbi` repository](https://github.com/mackelab/sbi) which implements the truncation of the proposal. Finally, the folder `paper` contains all code to assemble the figures shown in the paper from the produced results. It also contains the code to reproduce toy examples and illustrations (Fig. 1, Fig. 3)

### Installation

First, create the conda environment and activate it:
```
conda env create --file environment_vm.yml
conda activate tsnpe_neurips
```

Then, each subfolder is to be installed separately. Note that, in the second line, you will get an error saying that `sbibm 1.0.7 requires sbi==0.17.2, but you have sbi 0.18.0 which is incompatible`. You can savely ignore this error.
```
cd benchmark; pip install -e .; cd ..
cd sbi; pip install -e .; cd ..
cd l5pc; pip install -e .; cd ..
cd paper; pip install -e .; cd ..
```

Finally, we have to install the `pyloric` repo:
```
git clone git@github.com:mackelab/pyloric.git
cd pyloric
pip install -e .
```

### Citation
```
@article{deistler2022truncated,
  title = {Truncated proposals for scalable and hassle-free simulation-based inference},
  author = {Deistler, Michael and Goncalves, Pedro J and Macke, Jakob H},
  publisher = {arXiv},
  year = {2022},
}
```

### Contact
If you have questions, please reach out to `michael.deistler@uni-tuebingen.de`.
