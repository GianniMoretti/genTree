import os
import numpy as np

from os.path import join
from setuptools import setup, Extension
from Cython.Build import cythonize

# Comando: python setup.py build_ext --inplace
# Editable mode: pip install -e .

directory_path = os.path.dirname(os.path.abspath(__file__))

ext_data = {
    'genTree.tree': {
        'sources': [join(directory_path, 'genTree', 'tree.pyx')],
        'include': [np.get_include()],
    },
}

extensions = []

for name, data in ext_data.items():
    sources = data['sources']
    include = data.get('include', [])

    obj = Extension(
        name,
        sources=sources,
        include_dirs=include
    )

    extensions.append(obj)

setup(
    name='genTree',
    author='Gianni Moretti',
    ext_modules=cythonize(extensions, language_level="3"),
    packages=['genTree'],
    zip_safe=False
)
