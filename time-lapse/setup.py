from setuptools import setup, find_packages


TF_VERSION='>=1.10,<1.11'

setup(name='glib-ncb2021-timelapse',
      version='0.0.0',
      packages=find_packages(exclude=['tests', 'notebooks', 'models']),
      license='MIT',
      description=('lumen segmentation scripts.'),
      install_requires=[
          'h5py==2.10',
          'scikit-image==0.16.2',
          'numpy==1.14.5',
          f'tensorflow{TF_VERSION}',
          'scipy==1.5.0',   # explicitly define these indirect dependencies here
          'matplotlib<3.0', # in order to fix issues with pip's resolver.
          'dl-utils @ git+https://github.com/fmi-basel/dl-utils.git@pre-tf2',
          'tqdm',
      ],
      extras_require={'gpu': [f'tensorflow-gpu{TF_VERSION}']},
      entry_points={
          'console_scripts': [
              'run_lumen_segmentation = run_lumen_segmentation:main',
          ]
      })
