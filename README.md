
# Instructions

## Introduction

Is a implementation of
Krebs, Julian, Hervé Delingette, Boris Mailhé, Nicholas Ayache, and Tommaso Mansi. "Learning a probabilistic model for diffeomorphic registration." IEEE transactions on medical imaging 38, no. 9 (2019): 2165-2176.

Modified from the voxelmorph project found at https://github.com/voxelmorph/voxelmorph



## Setup
It might be useful to have each folder inside the `ext` folder on your python path.
assuming voxelmorph is setup at `/path/to/voxelmorph/`:

```
source activate seg
export PYTHONPATH=$PYTHONPATH:/home/ruihao/Learn-tissue-distribution/code/ext/neuron/:/home/ruihao/Learn-tissue-distribution/code/ext/pynd-lib/:/home/ruihao/Learn-tissue-distribution/code/ext/pytools-lib/

```

If you would like to train/test your own model, you will likely need to write some of the data loading code in 'datagenerator.py' for your own datasets and data formats. There are several hard-coded elements related to data preprocessing and format.


## Training
These instructions are for the MICCAI2018 variant using `train_miccai2018.py`.  
If you'd like to run the CVPR version (no diffeomorphism or uncertainty measures, and using CC/MSE as a loss function) use `train.py`

1. Change the top parameters in `train_miccai2018.py` to the location of your image files.
2. Run `train_miccai2018.py` with options described in the main function at the bottom of the file. Example:  
```
cd /home/workspace1/deformation_generator/circle_toy/vae_voxelmorph/test_translation/
nohup python3 train_miccai2018.py --gpu 0 --model_dir output0318/ > train255.out
Tail -f train255.out

```


## Testing (measuring Dice scores)
To test our VAE generator, we want to plot the latent space and use uniform and random sampling to generate some example outputs.
Reference: https://github.com/chaitanya100100/VAE-for-Image-Generation/tree/master/src

```
python3 test_cvae.py --gpu 0 --model_file /home/workspace1/deformation_generator/circle_toy/vae_voxelmorph/test_translation/output0318/1499.h5

```
