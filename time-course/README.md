# Time-course organoid segmentation

Scripts to segment organoids in MIPs.

## Installation

It is recommended to create a new python (tested with ```python=3.6```) environment, e.g. with ```venv``` or ```conda```, and activate this environment before installing this code.

Using conda, you can do this with: 

```
conda create -n time-course python=3.6
conda activate time-course
```
in the ```Anaconda prompt```.

Then install all dependencies:

```
cd time-course/
pip install .
```

If you have a cuda-compatible GPU available, you might want to install it with GPU support:
```
pip install .[gpu]
```

In order for this to work, you will need a working installation of CUDA. If you are using conda, you can get it with

```
conda install cudatoolkit=10.1
```


## Running tests

In order to run the tests, you'll need:

```
pip install pytest
```

and then you can run all tests like this:

```
pytest tests/
```

Again, this assumes you have already activated the corresponding environment and you are in the this folder.

## Run the segmentation

First, activate the environment, e.g. with ```conda activate time-course``` if you installed it with conda.

```
luigi --module instance_orgs.tasks.segmentation RunSegmentationModelTask --input-folder /path/to/TIF_OVR_MIP/ --file-pattern '*C01.tif' --output-folder path/to/output-folder/ --magnification 40 --local-scheduler
```

This will segment all images that match the ```--file-pattern``` in the folder specified by ```--input-folder```. The ```--magnification``` factor is used to adjust the preprocessing, e.g. 20 for a 20x acquisition or 40 for a 40x acquisition. ```--local-scheduler``` is a luigi specific flag that will avoid having to start a central scheduler in order to run the task.
