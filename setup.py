#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='ai-studio',
    version='0.0.3',
    description='',
    author='AIP Labs',
    author_email='studio@aiplabs.io',
    url='https://www.aiplabs.io/',
    packages=find_packages(exclude=['tests', '.cache', '.venv', '.git', 'dist']),
    scripts=[],
    install_requires=[
        'keras-model-specs>=2.3.0',
        'Keras==2.3.0',
        'visual-qa',
        'pandas',
        'matplotlib',
        'jupyter',
        'treelib==1.5.5',
        'jsonschema',
        'scipy',
        'sklearn',
        'plotly',
        'pyyaml',
        'freezegun==0.2.2',
        'GPUtil',
        'h5py==2.10.0',
        'gcloud',
        'httplib2==0.15.0',
        'anytree'
    ]
)
