from setuptools import setup, find_packages

try:
    import sys
    from os.path import realpath, dirname
    ESTIMATOR_DIR = dirname(realpath(__file__))
    sys.path.insert(1, '{}/hdestimator'.format(ESTIMATOR_DIR))
    from _version import __version__
except:
    __version__ = "unknown"


setup(
    name='hdestimator',
    version=__version__,
    install_requires=[
        'h5py',
        'pyyaml',
        'numpy',
        'scipy',
        'mpmath',
        'matplotlib',
        'numba',
    ],
    packages=find_packages(),
    include_package_data=True,
    py_modules=['hdestimator'],

    url='',
    license='BSD 3-Clause',
    author='',
    author_email='',
    description='The history dependence estimator tool',
)
