"""
# File Name: setup.py
# Description:
"""
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='ressac',
      version='1.1.2',
      packages=find_packages(),
      description='Resnet based single-cell ATAC-seq clustering',
      long_description='',


      author='mxy',
      author_email='xyMa00@126.com',
      url='https://github.com/xyMa00/bio_z7.git',
      scripts=['Ressac.py'],
      install_requires=requirements,
      python_requires='>3.7.0',
      license='MIT',

      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
     ],
     )
