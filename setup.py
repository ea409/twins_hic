from setuptools import setup 

setup(name="HiSiNet", 
version='0.0', 
author="Ediem Al-jibury",
author_email="ealjibur@ic.ac.uk",
description="HiSiNet: a tool for Hi-C analysis",
packages=['HiSiNet'],
install_requires=['cooler==0.8.10', 'frozendict==1.2','scipy>=1.5.2', 'torch==1.6.0','numpy>=1.18.0', 'Cython==0.29.21', 
                 'hic-straw==0.0.8' ], 
setup_requires=['cooler==0.8.10', 'frozendict==1.2','scipy>=1.5.2', 'torch==1.6.0','numpy>=1.18.0', 'Cython==0.29.21', 
                 'hic-straw==0.0.8'],
python_requires='>=3.6'
)
