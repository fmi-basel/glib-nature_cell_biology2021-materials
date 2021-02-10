from setuptools import setup, find_packages


TF_VERSION='>=1.10,<1.11'

setup(name='glib-ncb2021-timelapse',
      version='0.0.0',
      packages=find_packages(exclude=['tests', 'notebooks', 'models']),
      license='MIT',
      description=('lumen segmentation scripts.'),
      install_requires=[
          f'tensorflow{TF_VERSION}',
          'dl-utils @ git+https://github.com/fmi-basel/dl-utils.git@pre-tf2',
          'scikit-image>=0.16,<0.17',
          'numpy',
          'tqdm',
      ],
      extras_require={'gpu': [f'tensorflow-gpu{TF_VERSION}']},
      entry_points={
          'console_scripts': [
              'run_lumen_segmentation = run_lumen_segmentation:main',
          ]
      })
