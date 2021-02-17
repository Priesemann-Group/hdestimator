from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(
    name='hdestimator',
    version='0.9',
    install_requires=[
        'h5py',
        'pyyaml',
        'numpy',
        'scipy',
        'mpmath',
        'matplotlib',
        'cython',
    ],
    packages=find_packages(),
    include_package_data=True,
    py_modules=['hdestimator'],

    url='',
    license='MIT',
    author='',
    author_email='',
    description='The history dependence estimator tool',
    ext_modules=cythonize(["src/hde_fast_embedding.pyx"],
                          # "hde_fast_utils.pyx"],
                          annotate=False),
    include_dirs=[numpy.get_include()]
)
