from setuptools import setup
from Cython.Build import cythonize
import numpy

# to compile, run
# python3 setup.py build_ext --inplace

setup(name="Fase Embedding",
      ext_modules=cythonize(["fast_embedding.pyx"],
                            annotate=False),
      include_dirs=[numpy.get_include()])
