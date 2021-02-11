from setuptools import setup, find_packages

TF_VERSION = '>=2.3,<2.4'

setup(
    name='glib-ncb2021-timecourse',
    version='0.0.0',
    packages=find_packages(exclude=['tests', 'notebooks', 'models']),
    license='MIT',
    description=('overview segmentation scripts.'),
    install_requires=[
        f'tensorflow{TF_VERSION}',
        'tifffile==2020.6.3',
        'imagecodecs==2020.5.30',
        'luigi>=2.8.11,<3.0',
        'scikit-learn==0.22.1',
        'scikit-image==0.16.2',
        'scipy==1.4.1',
        'numpy==1.18.1',
        'numba==0.49.0',
        'tqdm==4.46.0',
    ],
    include_package_data=True,
    extras_require={'gpu': [f'tensorflow-gpu{TF_VERSION}']},
)
