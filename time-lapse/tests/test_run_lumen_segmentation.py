'''test lumen segmentation script.
'''
import os
import glob

from skimage.io import imread
import pytest
import numpy as np

SCRIPT = 'run_lumen_segmentation'

TEST_DATA = os.path.join(os.path.dirname(__file__), 'data', 'lifeact.tif')
TEST_DATA_SHAPE = (76, 466, 456)

MODELS = [os.path.join(os.path.dirname(__file__), '..', 'models', name, 'v0', 'model_best.h5')
          for name in ['organoid', 'enterocyst']]


                      
def test_data_is_available():
    assert os.path.exists(TEST_DATA)
    stack = imread(TEST_DATA)
    assert stack.shape == TEST_DATA_SHAPE


@pytest.mark.parametrize('model_path', MODELS)
def test_model_is_available(model_path):
    assert os.path.exists(model_path)

def test_run_lumen_segmentation_is_installed(script_runner):
    ret = script_runner.run(SCRIPT, '--help')
    assert ret.success

@pytest.mark.parametrize('model_path', MODELS)
@pytest.mark.script_launch_mode('subprocess')
def test_run_lumen_segmentation(script_runner, tmpdir, model_path):
    ret = script_runner.run(SCRIPT, TEST_DATA, '--output', str(tmpdir), '--model', model_path)
    assert ret.success

    for output_path in (tmpdir / subfolder / os.path.basename(TEST_DATA)
                        for subfolder in ['boundary_pred', 'lumen_pred']):
        assert output_path.exists()

        pred = imread(str(output_path))
        assert pred.shape == TEST_DATA_SHAPE

        assert 0 <= pred.min()
        assert pred.max() <= 255
        assert pred.dtype == 'uint8'
        
        rel_volume = (pred >= 127).sum() / np.prod(pred.shape)
        assert 0.02 <= rel_volume <= 0.1
