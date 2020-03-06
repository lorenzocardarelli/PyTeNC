from Cython.Distutils import build_ext
from Cython.Build import cythonize
from distutils.extension import Extension
from distutils.core import setup
import numpy

ext_modules=[
            Extension(
                      'slicing',
                      ['slicing.pyx'],
                      include_dirs=[numpy.get_include()]
                      )
             ]

setup(
      ext_modules = cythonize("slicing.pyx"),
      include_dirs=[numpy.get_include()],
      name='thread_demo',
      cmdclass={'build_ext': build_ext},
#      ext_modules=ext_modules
      )


