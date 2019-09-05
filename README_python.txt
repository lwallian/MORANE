

Create virual environement 
https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/

_______________________________________


Les commandes utilisées pour avoir la bonne version de conda/python/spyder
___________________________________________

In the code folder :

conda env create -f environment.yml
conda activate mecflu
conda update --all
conda install -c anaconda hdf5
spyder



Or alternatively:

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
