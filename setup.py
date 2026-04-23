# setup_simple.py
from setuptools import setup, Extension
import numpy as np


pso_module = Extension(
    'pso',
    sources=[
        'src/pso_core.cpp',
        'src/pso_bindings.cpp',
    ],
    include_dirs=[
        np.get_include(),
        './',         #
        'pybind11/include', # your pybind11 include
        'eigen-3.4.0'  # your eigen3 include
    ],
    language='c++',
    extra_compile_args=[
        '-std=c++11',
        '-O3',
        '-march=native',  # 
        '-fopenmp',       # 
    ],
    extra_link_args=['-fopenmp'],
)

setup(
    name='pso-optimizer',
    version='0.1.0',
    ext_modules=[pso_module],
    install_requires=['numpy>=1.19.0'],
)
