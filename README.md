# Earth data science

![](https://img.shields.io/badge/Python-3.10+-blue)
[![](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![](https://img.shields.io/badge/Year-2024/2025-yellow)
![](https://img.shields.io/badge/Status-Completed-green)
![](https://img.shields.io/badge/Institution-IPGP-red)


<img width="250px" src="lectures/images/xkcd.png" align="left" style="padding: 5px 20px 0px 0px;"/>

This repository contains the materical for the class _Earth data science_ delivered at the [Institut de Physique du Globe de Paris](https://www.ipgp.fr/) for master students. The course is an introduction to scientific computing and the use of Python for solving geophysical problems. The course is mostly based on practical sessions where students will learn how to use Python to solve problems related to the Earth sciences with statistical and machine learning methods. The course and notebooks rely on the Python [scikit-learn](https://scikit-learn.org/stable/) library, [pandas](https://pandas.pydata.org/), [pytorch](https://pytorch.org/), and the [deep learning](https://www.deeplearningbook.org/) book by Ian Goodfellow, Yoshua Bengio and Aaron Courville. This course is a legacy of the course of the same name by [Antoine Lucas](http://dralucas.geophysx.org/). The lectures are taught by [Léonard Seydoux](https://sites.google.com/view/leonard-seydoux/accueil) and the practicals by [Antoine Lucas](http://dralucas.geophysx.org/), [Alexandre Fournier](https://www.ipgp.fr/~fournier/), [Éléonore Stutzmann](https://www.ipgp.fr/~stutz/), and [Léonard Seydoux](https://sites.google.com/view/leonard-seydoux/accueil). 

The goal of this course is to introduce students to the basics of scientific computing and to the use of Python for solving geophysical problems. The course mostly consists in practical sessions where students will learn how to use Python to solve problems related to the Earth sciences mith statistical and machine learning methods. The course and notebooks rely on the Python [scikit-learn](https://scikit-learn.org/stable/) library, [pandas](https://pandas.pydata.org/), [pytorch](https://pytorch.org/), and the [deep learning](https://www.deeplearningbook.org/) book by Ian Goodfellow, Yoshua Bengio and Aaron Courville.

The course contains 8 hour of lecture followed by 20 hours of practical sessions made with Jupyter notebooks. The lecture notes are available in the `lectures` folder and the practicals in the `labs` folder. You can find an introductory README file in each folder.

<div style="clear: both;"></div>

## Lectures

The lectures will fit within two sessions of four hours each. The following list of lectures is proposed in the different subfolders of the `lectures` folder.

1. __Introduction to machine learning__. This section introduces the use cases of machine learning in the Earth sciences and the basic concepts of supervised and unsupervised learning.
2. __Definitions__. This section introduces the basic definitions of machine learning, including the various notations and the different types of learning.
3. __Supervised machine learning: regression__. This section introduces the concept of regression and the different metrics used to evaluate the performance of a regression model.
4. __Supervised machine learning: classification__. This section introduces the concept of classification and the different metrics used to evaluate the performance of a classification model
5. __Deep learning: the multilayer perceptron__. This section introduces the concept of deep learning and the multilayer perceptron
6. __Deep learning: convolutional neural networks__. This section introduces the concept of convolutional neural networks
7. __Applications__. This section introduces the different applications of machine learning in the Earth sciences
8. __Unsupervised learning__. This section introduces the concept of unsupervised learning, relying a lot on the previously seen concepts

## Notebooks    

### Structure

The following list of labs is proposed in the different subfolders of the `labs` folder.

0. [Self-evaluation](labs/0-self-evaluation/self-evaluation.ipynb) (1 hour). This lab is a self-evaluation of your Python skills. It is required to enroll in the course. A small solution will be delivered at the beginning of the lab session. 
1. [River sensor calibration](labs/1-calibration/) (4 hours). This lab allow to perform a first simple machine learning task: the calibration of a river sensor with supervised learning, where the goal is to predict the suspended sediment concentration from the turbidity of the water.
3. [Lidar data classification](labs/2-lidar-classification) (~8 hours). In this lab, we will classify lidar cloud points into different classes using supervised machine learning tools. Since this is a more complex task, we will take more time to complete it.
4. [Deep learning classification of MNIST](labs/3-deep-learning) (~4 hours). In this lab, we will use deep learning to classify the MNIST dataset with a fully connected neural network, and then with a convolutional neural network.


The solution to the different labs will be proposed progressively during the course in the corresponding folders. Note that the solutions provided are not necessarily the best ones. The main idea of these sessions is for you to be overly curious and to try to find the solutions that best fit your needs, and your understanding of the problem. Some of you may complete the tasks at a faster pace than others, and we encourage you to help your peers during the labs, and also to explore further aspects of the problems that are not covered in the labs.

### Running the labs on the Youpisco virtual machine

The easiest way to run most notebooks of this course is to connect to Youpisco, a virtual machine specifically designed for the course. You can connect to Youpisco at the following address __only from the local IPGP network__: https://youpisco.ipgp.fr. __Note that the first time you connect, you can create an account with your choice of username and password.__ Opening a page on Youpisco will open a Jupyter notebook interface as the screenshot shown below.

![Youpisco welcome screen](lectures/images/youpisco-welcome.png)

Once at that stage, you can open a terminal by clicking on the `Terminal` button on the launcher, navigate to your favorite directory (typically, on your home directory) and clone this repository by running the following command:

```bash
git clone https://github.com/leonard-seydoux/earth-data-science-public.git
```

This command should download the repository in a folder called `earth-data-science-public`, visible on the left panel of the Jupyter interface. You can then navigate to the `earth-data-science-public` folder and open the notebooks in the `labs` folder therin. Once open, you can start executing the cells by ensuring that the kernel is set to `Python 3 (Earth Data Science)`.

### Running the labs on your own machine

If you prefer, you can also run the notebooks on your own machine. To do so, you must install the required packages. The easiest way to do so is to use the `conda` package manager, and to create a new environment with the required packages listed in the file [requirements.txt](requirements.txt). To do so, you can run the following commands in a terminal:

```bash
conda env create -f requirements.txt
conda activate eds
```

__If you prefer this solution, please make sure you have a good knowledge of Python and the package manager `conda`.__



