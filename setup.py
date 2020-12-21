from setuptools import setup 

setup(name="HiSiNet", 
version='0.0', 
author="Ediem Al-jibury",
author_email="ealjibur@ic.ac.uk",
description="HiSiNet: a tool for Hi-C analysis",
install_requires=['cooler', 'frozendict','scipy==1.5.2', 'torch','numpy==1.18.0','hic-straw'], 
url="https://gitlab.doc.ic.ac.uk/ealjibur/CNN",
python_requires='>=3.6'
)