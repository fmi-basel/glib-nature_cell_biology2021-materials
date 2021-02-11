'''test lumen segmentation script.
'''
import os
import glob

from skimage.io import imread
import pytest
import numpy as np
import luigi
import tensorflow as tf

from instance_orgs.tasks.segmentation import DEFAULT_MODEL
from instance_orgs.tasks.segmentation import RunSegmentationModelTask


TEST_DATA = os.path.join(os.path.dirname(__file__), 'data', 'mini_40x.tif')
TEST_DATA_SHAPE = (2, 3000, 3000)



                      
def test_data_is_available():
    assert os.path.exists(TEST_DATA)
    stack = imread(TEST_DATA)
    assert stack.shape == TEST_DATA_SHAPE

@pytest.mark.parametrize('model_folder', [DEFAULT_MODEL['model_folder']])
def test_model_is_available(model_folder):
    assert tf.saved_model.contains_saved_model(model_folder)


    
def test_segmentation_task(tmpdir):
    '''segmentation workflow on the test data.
    '''
    output_folder = str(tmpdir)
    result = luigi.build([
        RunSegmentationModelTask(output_folder=output_folder,
                                 input_folder=os.path.dirname(TEST_DATA),
                                 magnification=40,
                                 file_pattern='*.tif')
    ],
                         local_scheduler=True,
                         detailed_summary=True)

    if result.status not in [
            luigi.execution_summary.LuigiStatusCode.SUCCESS,
            luigi.execution_summary.LuigiStatusCode.SUCCESS_WITH_RETRY
    ]:
        raise RuntimeError(
            'Luigi failed to run the workflow! Exit code: {}'.format(result))


    assert os.path.exists(os.path.join(output_folder, 'segmentation.log'))
    assert len(glob.glob(os.path.join(output_folder, '*'))) == 2
    
    segmentation_path = os.path.join(output_folder, os.path.basename(TEST_DATA))
    assert os.path.exists(segmentation_path)

    segmentation = imread(segmentation_path)
    assert segmentation.shape == TEST_DATA_SHAPE[1:]

    assert segmentation.min() == 0
    assert segmentation.max() == 4

