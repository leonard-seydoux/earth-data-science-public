# Machine Learning and its limitations in seismology

Welcome to this short course on the use of machine learning in seismology. The course is designed to give you a basic understanding of machine learning and how it can be used in seismology. It is not intended to be a comprehensive course on machine learning, but rather a short introduction to the topic.

## Contents

- [1. Installation instructions](#1-installation-instructions)
  - [1.1 Regular Python packages](#11-regular-python-packages)
  - [1.2 PyTorch](#12-pytorch)
  - [1.3 Seisbench](#13-seisbench)
  - [1.4 Running the notebooks](#14-running-the-notebooks)
- [2. Datasets download](#2-datasets-download)
  - [2.1 MNIST: handwritten digits](#21-mnist-handwritten-digits)
  - [2.3 Other seismic data](#23-other-seismic-data)

The present document is used for the hands-on part of the course. We will use Jupyter notebooks to run the code and to explore the data. Prior to the course, you should have installed the Anaconda Python distribution on your computer, with the following packages.

## 1. Installation instructions

### 1.1 Regular Python packages

The best way to work with Python is to use a package manager such as `conda` or `pip`. We will use `conda` in this course. The good idea with `conda` is that it allows to create isolated environments, which is very useful when working with different projects. Because Python packages are often updated, it is also a good idea to create a new environment for each project. This way, you can be sure that the packages you are using are the ones you want to use, and that they are compatible with each other.

First, you should create a clean Python environment and name it after the course. This can be done by opening a terminal and typing:

    conda create --name spin-ml

This will create an empty environment called `spin-ml`. You can activate this environment by typing:

    conda activate spin-ml

This will change your Python path and make sure that all the packages you install will be installed in this environment. By default, we will be using the channel `conda-forge` to install the packages. This can be done by typing:

    conda config --add channels conda-forge

The latest version of Python (3.11 in March 2023) may not be completely compatible with all the packages we will use. Therefore, we will use Python 3.9. This can be done by typing:

    conda install python=3.9

Now we can install all (but one of) the required packages. We will use primarily Numpy, Scipy, Pandas, Scikit-Learn and Matplotlib. We will also use Seaborn and PyTorch. This can be done by typing:

    conda install numpy scipy pandas scikit-learn matplotlib seaborn obspy tqdm

### 1.2 PyTorch

This Python package is one of the most popular deep learning libraries. It can both run on CPU or GPU. Because we are going ot work with our laptots here, we will use the CPU version. Several possibilities exist to install PyTorch, depending on your operating system. If the following commant fails, please refer to the [PyTorch documentation](https://pytorch.org/get-started/locally/) to install the package on your computer.

    conda install pytorch torchvision -c pytorch

### 1.3 Seisbench

We will also use the Seisbench package, which is a Python package for the analysis of seismic data. This package is not yet available on the `conda-forge` channel. Therefore, we will install it with pip, as recommended in the [Seisbench documentation](https://seisbench.readthedocs.io/en/stable/pages/installation_and_usage.html).

Note that it's safer to install it in a dedicated conda environment, as it may require to install other packages. In particular, the torch version may differ from the one you have installed in the previous step. Therefore, we will install it in a dedicated environment. This can be done by typing:

    conda create --name seisbench
    conda activate seisbench
    conda install python=3.9
    pip install torch==1.12.1 seisbench

### 1.4 Running the notebooks

All the practicals are in the form of Jupyter notebooks, which require further packages to be installed. In all environements, you can install the Jupyter package by typing:

    conda install jupyter

You now have several options to run the notebooks. One strategy is to use a dedicated software such as VSCode, which has a built-in Jupyter notebook interface. Another option is to use Jupyter Lab, which will run a Jupyter notebook server in your browser.

- __Using Jupyter Lab__: If you would like to use Jupyter Lab, you can install it by typing `conda install jupyterlab`. Then, follow the instructions described in the [Jupyter Lab documentation](https://jupyter.org/install) to start the server on your computer.

- __Using VSCode__: If you use VSCode, you can directly open the folder in which the Jupyter notebooks are located, and open it with the explorer. Note that running notebooks with VSCode require to install several extensions in VSCode. Please follow the instructions described in the [VSCode documentation](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

- __Using other software or converting to Python scripts__: You can also consider using other softwares to run the notebooks. For example, you can use [Google Colab](https://colab.research.google.com/) to run the notebooks in the cloud. You can also use [Binder](https://mybinder.org/) to run the notebooks in the cloud. Please refer to the documentation of these softwares to learn how to use them. The notebooks can also be converted to regular Python scripts to be run in a terminal. This can be done by typing `jupyter nbconvert --to script notebook_name.ipynb`.

## 2. Datasets download

### 2.1 MNIST: handwritten digits

> We decided to ship the MNIST dataset is shipped within this repository. It is located in the `data` folder.

### 2.2 Seismic data from the SeisBench package

We will also use the SeisBench package to download some seismic data. This package is already installed in the `seisbench` environment. After activating this environment, you can download the data by typing:

``` python
import seisbench
from seisbench.data import Iquique

# Download the data
print(seisbench.cache_root)
Iquique()
```

This will download the data in the `seisbench.cache_root` folder (usually in your home directory). You can change the path by defining the `SEISBENCH_DATA` environment variable. This dataset is about 4GB. Make sure you have the space available on your computer. We can also give the dataset with a USB stick if you prefer. If you go for this option,please contact me to get a USB stick. The data will be located in the `iquique` folder. You can copy this repository inside:

    $SEISBENCH_CACHE_ROOT/datasets/

where you have to replace `$SEISBENCH_CACHE_ROOT` by the path to the `seisbench` cache root that you can get from

```python
import seisbench
print(seisbench.cache_root)
```

### 2.3 Other seismic data

We will also understand and test several earthquake analysis deep learning models. Any seismic data you may want to bring (or already have) is welcome. If you wish to work on a dedicated seismic timeseries, we will download some from IRIS with ObsPy. This can be done by typing:

```python
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

client = Client("IRIS")

# Download the data
st = client.get_waveforms(
    network="IU",
    station="ANMO",
    location="00",
    channel="BHZ",
    starttime=UTCDateTime("2019-01-01"),
    endtime=UTCDateTime("2019-01-02"),
    attach_response=True,
)
```
