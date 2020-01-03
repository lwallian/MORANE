
#########################################
#########################################
#########################################
Description :
#########################################

The python code performs the data assimilation using precomputed ROM coefficients and ROM noise statistics.
The main script is super_main_from_existing_ROM.py
Most parameters have to be specified in the begining of main_from_existing_ROM.py

#########################################
#########################################
#########################################
Installation :
#########################################

1) (If needed,) 
Install anaconda, including spyder, python 3.*, etc

#########################################

2) Create the environment mecflu from environment.yml  as follow :
Go the code folder, and from the conda terminal, enter:

conda env create -f environment.yml
conda activate mecflu
conda update --all
conda install -c anaconda hdf5
spyder

## These commands will install the correct version of python packages.
More information about virual environement 
https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/

## Alternative commands :

conda env create -f environment.yml
conda activate mecflu
conda update spyder
conda install -c conda-forge qtpy
 conda install qt=5 pyqt=5 qtpy=1.1.2 --force
conda install PySide
conda remove spyder
conda install -c anaconda spyder
conda install -c anaconda pyqt
pip uninstall PyQt5
pip uninstall spyder
conda install -c anaconda spyder
pip uninstall QtPy
conda install --force-reinstall QtPy qt
conda install --force-reinstall pyqt qt
pip uninstall PyQt5
conda install --force-reinstall pyqt qt
conda install -c anaconda hdf5
spyder

#########################################

3) Always work from in this environnement.

Example with spyder:

Open "Spyder(mecflu)"

Or,
from a conda terminal:

conda activate mecflu
spyder
