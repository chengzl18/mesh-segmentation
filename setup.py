"""from setuptools import setup
from distutils.extension import Extension
# from Cython.Build import cythonize"""
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name='c_utils app',
    ext_modules=[
        Extension('c_utils',
                  sources=['c_utils.pyx'],
                  # https://learn.microsoft.com/en-us/cpp/build/reference/o-options-optimize-code?view=msvc-170
                  extra_compile_args=['/O2'],  # /O2 for MSVC; -O3 for gcc and clang
                  language='c++',
                  # compiler_directives={'language_level': "3"}
                  )
    ],
    cmdclass={'build_ext': build_ext},
    # https://stackoverflow.com/questions/14657375/cython-fatal-error-numpy-arrayobject-h-no-such-file-or-directory
    include_dirs=[numpy.get_include()],
)
"""
setup(
    ext_modules=[
        Extension('c_utils',
                  sources=['c_utils.pyx'],
                  extra_compile_args=['-O3'],
                  language='c++')
    ],
    extra_compile_args=["-O3"],
    include_dirs=[numpy.get_include()]
)"""
