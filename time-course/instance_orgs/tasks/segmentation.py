import os

import luigi
from luigi.util import requires

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from skimage.measure import block_reduce

from .collector import TiffImageTarget, ImageCollectorTask
from ..logging import get_logger
from ..postprocessing import embeddings_to_segmentation
from ..postprocessing import fill_holes_in_objects


# Default model path and parameters
MODEL_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'models'))

DEFAULT_MODEL = {
    'model_folder': os.path.join(MODEL_FOLDER, 'dapi', 'v0'),
    'base_factor': 8,
    'base_magnification': 20,
    'bandwidth': 0.05,
    'nonmax_size': 15,
    'nonmax_sigma': 1,
    'nonmax_min_votes': 9,
}


def preprocess(image, downsampling_factor):
    '''downsamples and rescales to [-1, 1].
    '''
    if not image.ndim == 2:
        raise RuntimeError(f'Unexpected image shape: {image.shape}')
    image = block_reduce(image, (downsampling_factor, downsampling_factor),
                         np.mean)
    vmin, vmax = np.percentile(image.flat, (0, 100.0))
    image = 2.0 * (image - vmin) / (vmax - vmin) - 1.0
    return image[None, ..., None]


@requires(ImageCollectorTask)
class RunSegmentationModelTask(luigi.Task):
    '''applies a given segmentation model.
    '''

    output_folder = luigi.Parameter()
    '''folder into which the segmentations will be written.
    '''

    model_folder = luigi.Parameter(default=DEFAULT_MODEL['model_folder'])
    '''folder containing trained model.
    '''

    magnification = luigi.IntParameter()
    '''magnification used on the scope.
    '''
    base_factor = luigi.IntParameter(default=DEFAULT_MODEL['base_factor'])
    base_magnification = luigi.IntParameter(
        default=DEFAULT_MODEL['base_magnification'])

    # parameters for embeddings_to_segmentation
    nonmax_size = luigi.IntParameter(default=DEFAULT_MODEL['nonmax_size'])
    nonmax_sigma = luigi.FloatParameter(default=DEFAULT_MODEL['nonmax_sigma'])
    nonmax_min_votes = luigi.IntParameter(
        default=DEFAULT_MODEL['nonmax_min_votes'])
    bandwidth = luigi.FloatParameter(default=DEFAULT_MODEL['bandwidth'])
    fill_holes = luigi.BoolParameter(default=True)

    @property
    def _downsampling_factor(self):
        '''calculates an adjusted downsampling factor based on the given magnification.
        '''
        factor = self.magnification / self.base_magnification * self.base_factor
        return int(round(factor))

    def _load_predictor(self):
        '''returns a function that composes preprocessing and actual model.
        '''
        self._model = tf.keras.models.load_model(self.model_folder)

        def _predictor(image):
            prep = preprocess(image, self._downsampling_factor)
            return dict(
                zip(self._model.output_names,
                    self._model.predict(prep)[0]))

        return _predictor

    def run(self):
        '''
        '''
        logger = get_logger(
            self.__class__.__name__,
            os.path.join(self.output_folder, 'segmentation.log'))
        logger.info(str(self))

        try:
            predictor_fn = self._load_predictor()
        except Exception as err:
            logger.error(
                f'Could not load model from {source.path}. Error: {err}')
            raise

        def _process(source, target):
            '''process a single input-output pair.
            '''
            if target.exists():
                logger.info('Segmentation at %s already exists. Skipping ...',
                            target.path)
                return 1

            logger.debug(f'Loading and processing {source.path} ...')

            try:
                image = source.load()
                if image.ndim == 3 and image.shape[
                        0] == 2:  # drop acquisition mask
                    image = image[0]
            except Exception as err:
                logger.error(
                    f'Could not load image from {source.path}. Error: {err}')
                return 1

            try:
                prediction = predictor_fn(image)
            except Exception as err:
                logger.error(
                    f'Could not predict on image {source.path}. Error: {err}')
                return 1

            try:
                segmentation, probs = embeddings_to_segmentation(
                    prediction['fg'].squeeze() > 0.5,
                    prediction['embedding'].squeeze(),
                    bandwidth=self.bandwidth,
                    nonmax_size=self.nonmax_size,
                    sigma=self.nonmax_sigma,
                    min_votes=self.nonmax_min_votes)
                if self.fill_holes:
                    segmentation = fill_holes_in_objects(segmentation)

                segmentation = resize(segmentation,
                                      image.shape,
                                      preserve_range=True,
                                      order=0,
                                      mode='reflect',
                                      anti_aliasing=True).astype('uint16')
            except Exception as err:
                logger.error(
                    f'Postprocessing failed for image from {source.path}. Error: {err}'
                )
                return 1

            if not np.any(segmentation):
                logger.warn('Encountered an empty segmentation!')

            try:
                target.save(segmentation, compress=9)
            except Exception as err:
                logger.error(
                    f'Could not save segmentation to {target.path}. Error: {err}'
                )
                return 1
            return 0

        # do the work
        pairs = self._all_input_output_pairs()
        num_errors = sum(
            _process(source, target)
            for source, target in tqdm(pairs, desc='Processing', ncols=80))

        if num_errors >= 1:
            raise RuntimeError(
                f'{self.__class__.__name__} terminated with {num_errors} errors! See the log for more details'
            )

        logger.info('Done.')

    def _all_input_output_pairs(self):
        '''generates pairs of inputs and outputs
        '''

        def _get_fname(img_path):
            return os.path.splitext(os.path.basename(img_path))[0] + '.tif'

        def _make_target(source):
            return TiffImageTarget(
                os.path.join(self.output_folder, _get_fname(source.path)))

        return [(source, _make_target(source)) for source in self.input()]

    def output(self):
        '''defines output of the task.
        '''
        if not self.input():
            raise ValueError('No input images provided!')

        return [target for _, target in self._all_input_output_pairs()]
