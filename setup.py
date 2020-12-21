from setuptools import setup 

setup(name="HiSiNet", 
version='0.0', 
author="Ediem Al-jibury",
author_email="ealjibur@ic.ac.uk",
description="HiSiNet: a tool for Hi-C analysis",
packages=['skimage','cooler', 'frozendict','scipy', 'torch', 'collections', 'numpy','hic-straw', 'pickle'], 
url="https://gitlab.doc.ic.ac.uk/ealjibur/CNN",
python_requires='>=3.6',
)