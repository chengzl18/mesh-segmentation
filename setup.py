from setuptools import setup
import numpy
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("dijkstra.pyx"),
    include_dirs=[numpy.get_include()],
)
