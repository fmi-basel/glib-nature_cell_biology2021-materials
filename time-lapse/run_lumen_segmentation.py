import os
import logging
import argparse

import numpy as np
from tqdm import tqdm

from skimage.io import imread
from skimage.io import imsave
from skimage.measure import block_reduce
from skimage.transform import resize

from dlutils.models import load_model
from dlutils.prediction import predict_complete
from dlutils.models.utils import get_input_channels

# logger format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] (%(name)s) [%(levelname)s]: %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S')
logger = logging.getLogger(__name__)


def parse():
    '''parse command line arguments.

    '''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', help='input image(s)', nargs='+')
    parser.add_argument('--output', help='output directory', required=True)
    parser.add_argument('--model', help='model to load', required=True)
    return parser.parse_args()


def normalize(img):
    '''
    '''
    lower, upper = np.percentile(img.flat, (1.0, 99.0))
    return (img - lower) / (upper - lower)


def get_outpath(input_path, out_dir):
    '''construct output path as out_dir/filename

    '''
    if not os.path.exists(out_dir):
        logger.info(
            'Out directory {} doesnt exist, creating it.'.format(out_dir))
        os.makedirs(out_dir)
    basename = os.path.basename(input_path)
    return os.path.join(out_dir, basename)


def _swap_outputs(predictions):
    '''
    '''
    keys = list(predictions.keys())
    assert len(keys) == 2
    first, second = keys
    predictions[first], predictions[second] = predictions[second], predictions[
        first]
    return predictions


def predict_volume(model, img, block_size=1, border=50, batch_size=4):
    '''apply model slice-wise to image volume.

    Parameters
    ----------
    model : keras.Model
        Prediction model to be applied.
    img : ndarray
        Image to be processed. Axes are expected to be in order (Z, X, Y)
    downsampling : int
        Blocksize for block_reduce in case a downsampling is performed.

    '''
    original_shape = img.shape

    img = block_reduce(img, (1, block_size, block_size), func=np.max)
    logger.debug('Image shape original: {}, and downsampled: {}'.format(
        original_shape, img.shape))

    img = normalize(img)

    def generator(image_volume):
        '''generates input slices of the required size.

        '''
        n_slices = get_input_channels(model)
        slice_offset = n_slices // 2

        # make z-axis the last axis to treat it as channels.
        image_volume = np.rollaxis(image_volume, 0, image_volume.ndim)

        for idx in range(image_volume.shape[-1]):
            if n_slices == 1:
                indices = idx
            else:
                indices = range(idx - slice_offset, idx + slice_offset + 1)
                indices = [
                    max(0, min(image_volume.shape[-1] - 1, x)) for x in indices
                ]

            yield image_volume[..., indices]

    output = {str(key): [] for key in model.output_names}
    for img_slice in generator(img):

        pred = predict_complete(
            model, img_slice, border=border, batch_size=batch_size)
        for key in output.keys():
            output[key].append(pred[key])

    for key in output.keys():
        output[key] = np.asarray(output[key])

    logger.debug('Predictions: ' + ''.join(
        ['{}: {}'.format(key, val.shape) for key, val in output.items()]))

    # optional resizing/upsampling
    for key, val in output.items():
        output[key] = resize(val.squeeze(), original_shape, mode='reflect')
    logger.debug('Predictions resized: ' + ''.join(
        ['{}: {}'.format(key, val.shape) for key, val in output.items()]))

    return output


def process(inputs, out_dir, model_path):
    '''run model on all input images.

    '''
    if any(
            os.path.dirname(os.path.abspath(path)) == os.path.abspath(out_dir)
            for path in inputs):
        raise ValueError('Output directory must be different from input!')

    logger.info('Loading model from {} ...'.format(model_path))
    model = load_model(model_path)
    logger.info('successful!')

    for input_path in tqdm(inputs, desc='Processing volumes', ncols=80):
        img = imread(input_path)

        prediction = predict_volume(
            model, img, block_size=2)

        # Workaround for lumen models
        prediction = _swap_outputs(prediction)

        for key, val in list(prediction.items()):
            out_path = get_outpath(input_path, os.path.join(out_dir, key))
            imsave(out_path, (val * 255).astype(np.uint8))


def main():
    try:
        args = parse()
        process(args.inputs, args.output, args.model)
    except Exception as err:
        logger.error(str(err), exc_info=True)
        return 1
    return 0


if __name__ == '__main__':
    main()
