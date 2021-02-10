# Time lapse lumen segmentation

Scripts to segment lumen in lightsheet stacks.

## Installation

It is recommended to create a new python (tested with ```python=3.6```) environment, e.g. with ```venv``` or ```conda```, and activate this environment before installing this code.

Using conda, you can do this with: 

```
conda create -n time-lapse python=3.6
conda activate time-lapse
```
in the ```Anaconda prompt```.

**NOTE** make sure that you have **git lfs** installed when cloning from github. Otherwise, the models' weights are not cloned properly and you would have to download them manually from github. If the ```.h5``` files in the ```models/``` filetree are just a few KB in size, then it's an indicator that there's a problem with your ```git lfs```.


Then install all dependencies:

```
cd time-lapse/
pip install .
```

You might have to authenticate with your github account in order to install all necessary dependencies during this step.

## Run the segmentation

You can apply a segmentation model from ```models/``` to a set of new images:

```
run_lumen_segmentation /path/to/some/*.tif  --output /path/to/output/folder/ --model models/organoid/v0/model_best.h5
```

Make sure that your environment is active before calling the script.