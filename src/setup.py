from setuptools import setup
from Cython.Build import cythonize
import numpy

# to compile, run
# python3 setup.py build_ext --inplace

setup(name="Speedy Module",
      ext_modules=cythonize(["hde_fast_embedding.pyx"],
                             # "hde_fast_utils.pyx"],
                            annotate=False),
      include_dirs=[numpy.get_include()])
