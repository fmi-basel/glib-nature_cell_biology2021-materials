'''image file collector task and targets for tifffile-readable images.
'''
import glob
import os

import luigi
from tifffile import imread, imsave


class TiffImageTarget(luigi.LocalTarget):
    '''provides load and save utilities for local tiff image file targets.
    '''

    def load(self):
        '''
        '''
        return imread(self.path)

    def save(self, vals, **kwargs):
        '''
        '''
        with self.temporary_path() as path:
            imsave(path, vals, **kwargs)


class FileCollectorTask(luigi.ExternalTask):
    '''collects targets from the file system.
    The targets are sorted alphabetically with respect to their filename.
    '''
    input_folder = luigi.Parameter()
    '''input folder to collect from.
    '''

    file_pattern = luigi.Parameter()
    '''filename pattern matching files to be included.
    Allowed wildcards are *, ?, [seq] and [!seq] (see fnmatch)
    '''

    def make_target(self, path):
        '''creates the target for a given path. This method is intended
        to be specialized in inheriting classes.
        '''
        return luigi.LocalTarget(path)

    def output(self):
        '''returns a LocalTarget for each file in the input_folder matching
        the given file_pattern.
        '''
        if not os.path.exists(self.input_folder):
            raise FileNotFoundError('Input folder {} does not exist!'.format(
                self.input_folder))

        # gather all files.
        paths = sorted(
            glob.glob(os.path.join(self.input_folder, self.file_pattern)))
        if not paths:
            raise RuntimeError('No files matching {} found in {}!'.format(
                self.file_pattern, self.input_folder))

        targets = [self.make_target(path) for path in paths]
        return targets


class ImageCollectorTask(FileCollectorTask):
    '''collects image targets from the file system.
    The targets are sorted alphabetically with respect to their filename.
    Currently, only tifffile compatible images are supported.
    '''

    def make_target(self, path):
        '''
        '''
        # TODO Adding support for more image targets.
        ext = os.path.splitext(path)[1]
        if ext.lower() in ['.tif', '.tiff', '.stk']:
            return TiffImageTarget(path)
        raise RuntimeError(
            'The extension {} is currently not associated with any ImageTarget'
            .format(ext))
