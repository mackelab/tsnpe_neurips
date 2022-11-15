# Truncated proposals for scalable and hassle-free simulation-based inference
Code to reproduce results from `Truncated proposals for scalable and hassle-free simulation-based inference` by Michael Deistler, Pedro J. Goncalves*, Jakob H. Macke*, accepted at NeurIPS 2022.


### Structure of this repository
This repo contains four main folders. Each folder contains an individual `README.md` file which describes how the particular results can be reproduced:
- The folder `benchmark` contains code to run the benchmark tasks (Fig. 4)
- The folder `l5pc` contains code to produce the results for the pyloric network (Fig. 5) and for the L5PC (Fig. 6).
- The folder `sbi` is a modified version of the [`sbi` repository](https://github.com/mackelab/sbi) which implements the truncation of the proposal.
- The folder `paper` contains all code to assemble the figures shown in the paper from the produced results. It also contains the code to reproduce toy examples and illustrations (Fig. 1, Fig. 3)


### Completeness
This repository contains code that can be run to generate all results. In addition, it contains some files that are loaded in the notebooks in order to generate the figures (these files are also generated along the way when running the code). However, in order to maintain a reasonable file size of this repo, many intermediate results (which are not shown in the paper, e.g. trained neural networks after every round) are not stored in this repository. In addition, this repository does not include the full git history of the project. In order to recover these things, we point the reader to the [development repository of this project](https://github.com/mackelab/tsnpe_neurips_dev).


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

### Installation of pyloric network simulator
In order to reproduce the results in Fig. 5 (pyloric network), you have to install the [`pyloric` repo](https://github.com/mackelab/pyloric). Note: original results were generated with commit hash `1768cc4365d2ca24196562a4cf41a2b28cc6e647`.
```
git clone git@github.com:mackelab/pyloric.git
cd pyloric
pip install -e .
```

### Installation of L5PC simulator
Finally, in order to reproduce the results shown in Fig. 6, you have to set up the simulator for the layer 5 pyramidal cell (L5PC, Fig. 6). This simulator is written in [`Neuron`](https://www.neuron.yale.edu/neuron/). In order to compile the model, you have to:
```
cd l5pc/l5pc/model/x86_64
rm *.o; rm *.c; cd ..
nrnivmodl mechanisms
```

### Large file storage
This repo is significantly smaller than the [development repo](https://github.com/mackelab/tsnpe_neurips_dev). However, it still contains neural networks that are used to visualize results and a few datasets that are plotted. The overall filesize of this repo is 8.7 GB. Large files are stored with [Git LFS](https://git-lfs.github.com/).


### Citation
```
@inproceedings{
  deistler2022truncated,
  title={Truncated proposals for scalable and hassle-free simulation-based inference},
  author={Michael Deistler and Pedro J. Goncalves and Jakob H. Macke},
  booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=QW98XBAqNRa}
}
```


### Contact
If you have questions, please reach out to `michael.deistler@uni-tuebingen.de`.
