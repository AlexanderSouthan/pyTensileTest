from setuptools import setup, find_packages

setup(
    name='pyTensileTest',
    version='0.0.1',
    packages=find_packages(where='src'),
    install_requires=['numpy', 'matplotlib', 'pandas', 'pip', 'little_helpers', 'pyPreprocessing', 'pyDataFitting']
)
