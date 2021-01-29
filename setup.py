from setuptools import setup, find_packages

setup(
    name='python-template',
    version='0.0.0',
    author='John Doe',
    author_email='john.doe@fmi.ch',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    license='MIT',
    description=(
        'A short description about the package.'
    ),
    long_description=open('README.md').read(),
    install_requires=[
        # list of dependencies, e.g.
        #"numpy >= 1.17.4",
        #"pandas >= 0.25.3",
    ],
    setup_requires=['pytest-runner',],
    tests_require=['pytest',],
)
