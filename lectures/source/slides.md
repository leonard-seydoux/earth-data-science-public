---
theme: presentation
marp: true
math: katex
paginate: false
---

<!-- _class: titlepage-->

# Earth data science

Master lecture at the [institut de physique du globe de Paris](https://www.ipgp.fr) inspired by the [scikit-learn](https://scikit-learn.org/stable/) documentation and the [deep learning](https://www.deeplearningbook.org/) book.

`Léonard Seydoux` `Antoine Lucas` `Éléonore Stutzmann` 
`Alexandre Fournier` `Geneviève Moguilny` 

[![height:30](https://img.shields.io/badge/github-leonard--seydoux/earth--data--science--public-white?logo=github)](https://github.com/leonard-seydoux/earth-data-science-public) 

---

<!-- _class: titlepage -->

# Contents

1. [Introduction](#1-introduction)
2. [Definitions](#2-definitions)
3. [Regression](#3-regression)
4. [Classification](#4-classification)
5. [Unsupervised learning](#5-unsupervised-learning)
6. [Deep learning](#6-deep-learning)
7. [Convolutional neural networks](#7-convolutional-neural-networks)

---

# 1. Introduction

![bg left 60%](contents/figures/reinforcement_learning.png)

---

## Goals of the class

- __Identify__ data-related problems

- __Define__ data-related problems

- __Build__ machine-learning solutions

- __Read & digest__ scientific papers

![bg right 80%](contents/images/papers/bergen2019machine.png)

<!-- _footer: Bergen et al. (2019)  -->

---

## Goals of the class

- __Identify__ data-related problems

- __Define__ data-related problems

- __Build__ machine-learning solutions

- __Read & digest__ scientific papers

- __Keep up__ with the ongoing pace!

![bg right 80%](https://www.science.org/cms/10.1126/science.abm4470/asset/ca252939-7ddb-4fe2-af0b-aa147130f29c/assets/images/large/science.abm4470-f1.jpg)

<!-- _footer: Mousavi et al. (2022) -->

---

## References

This course makes large use of the [scikit-learn](https://scikit-learn.org/stable/) library because it is

- Open source
- Didactic
- Built on top of NumPy and SciPy
- A greybox library

![bg right 40%](https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/250px-Scikit_learn_logo_small.svg.png)

<!-- _footer:  [www.scikit-learn.org](https://scikit-learn.org/stable/) -->

---

##  References

The [deep learning](https://www.deeplearningbook.org/) book by Goodfellow, Bengio, and Courville is also a major reference that covers:

- **Historical** aspects
- **From scratch** to deep learning
- **Examples** and exercises
- **Freely** accessible online

![bg right 65%](https://m.media-amazon.com/images/I/A1GbblX7rOL._AC_UF1000,1000_QL80_.jpg)

<!-- _footer:  [www.deeplearningbook.org](https://www.deeplearningbook.org/) -->

---

## How fast can you describe the following images?
![](contents/images/papers/karpathy2015deep-nocap.png)
<!-- _footer: Karpathy & Fei-Fei (2015) -->

---

## How accurate are those descriptions?
![](contents/images/papers/karpathy2015deep.png)
<!-- _footer: Karpathy & Fei-Fei (2015) -->

---

## Data to knowledge pipeline

1. Identify objects
2. Recognize object categories
3. Relate objects
4. Sort links by priority
5. Generate text out of it

![bg right 75%](contents/images/papers/karpathy2015motivation.png)
<!-- _footer: Karpathy & Fei-Fei (2015) -->

---

## Data to knowledge pipeline

We can decompose complex tasks into simpler subtasks that machines can learn to solve. This is the essence of __machine learning__.

![bg right 80%](contents/images/deep-learning-book/figure-1-2.png)
<!-- _footer: Goodfellow et al. (2016) -->

---

## Can you spot<br>the seismogram?

![ bg right 70%](contents/images/papers/valentine2012spot.png)

__Message__: not all times series of 500 samples are seismograms. Seismograms are just a small subset.

<!-- _footer: Valentine & Trampert (2012)-->

---

## Example task: seismic event detection and classification 

Most humans can pinpoint events. <br><br>

![bg right 70%](contents/images/papers/moran2008helens-nolabels.png)

<!-- _footer: modified after Moran et al. (2008) -->

---

## Example task: seismic event detection and classification 


Most humans can pinpoint events. 
Experts can __classify__ them.


![bg right 70%](contents/images/papers/moran2008helens.png)

<!-- _footer: modified after Moran et al. (2008) -->

---

## Example task: dive into previously unseed data

![](contents/images/papers/clinton2021marsquake.png)

Expert-detected marsquake within continuous insight data. 

<!-- _footer: Clinton et al. (2021) -->

---

## Machine learning tasks

- Time-consuming tasks
- Hard-to-describe tasks
- Exploration of new data

![bg right 80%](https://www.science.org/cms/10.1126/science.aau0323/asset/0e7cf386-2fd1-4683-8114-1bb875dbc580/assets/graphic/363_aau0323_f3.jpeg)

<!-- _footer: Bergen et al. (2019) -->

---

# 2. Definitions

![bg left 80%](contents/figures/exp_task_perf_venn.png)

---

## General definition

An algorithm learns from **experience** with respect to a **task** and **performance**, if its performance at solving the task improves with experience.

![bg right 80%](contents/figures/exp_task_perf_venn.png)


---

## The data, the model, and the loss

<div class="box">

🙊

__the data__ 

A set of samples $\mathbf{x}_i$ and labels $\mathbf{y}_i$ to learn from:

$$\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^N$$

</div>
<div class="box">

🙉 

__the model__ 

A parametric function $f_\theta$ that maps data $\mathbf{x}$ to  $\hat{\mathbf{y}}$ 

$$f_\theta : \mathbf{x} \mapsto \hat{\mathbf{y}}$$

</div>
<div class="box">

🙈 

__the loss__

A measurement of the  model performance

$$\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$$

</div>
<div align=center  data-marpit-fragment="0"

__Learning__ is equivalent to find the optimal parameters $\theta^*$ that minimize the loss function $\mathcal{L}$, as in

$$\theta^* = \underset{\theta}{\arg\!\min}\, \mathcal{L}\Big(f_\theta(\mathbf{x}), \mathbf{y}\Big)$$

</div>

---

## Vocabulary and symbols

<div>

An image is a sample $\bf x$ with 
$$\mathbf{x} \in \mathbb{X} = \mathbb{R}^{H \times W \times C}$$
$H$ is the height, $W$ the width, and $C$ the channels. The labels are a category $y$ with
$$y \in \mathbb{Y} = \{0, 1, \ldots, K\}$$
with $K$ the number of categories. 
Note that $y$ is scalar in this case.

</div>
<div>

| Symbol | Name |
|:-|:-|
|$\left\{ \mathbf{x}_i \in \mathbb{X} \right\}_{i =  1\ldots N}$| Collection of __samples__|
|$\left\{ \mathbf{y}_i \in \mathbb{Y} \right\}_{i =  1\ldots N}$| Collection of __labels__|
|$\mathbf{x}=(x_1, \ldots, x_F)$| Set of sample __features__|
|$\mathbf{y}=(y_1, \ldots, y_T)$| Set of label __targets__|
|$N$| Dataset size|
|$F$| Feature space dimensions|
|$T$| Target space dimension|
|$\mathbb{X}$| Data space|
|$\mathbb{Y}$| Label space|

</div>

---

## Types of machine learning problems


__Supervised__
![width:330](contents/figures/classification.png) 
__Predict__ $\mathbf{y}$ from $\mathbf{x}$ 
(regression, classification)

__Unsupervised__
![width:330](contents/figures/clustering.png) 
Learn data __distribution__
$p(\mathbf{x})$ (clustering, reduction)

__Reinforcement__
![width:330](contents/figures/reinforcement_learning.png) 
Learn a policy to maximize 
a __reward__ (gaming, robotics)

---

## Supervised learning tasks

__Regression__
$x$ and $y$ are continuous
![width:350px](contents/figures/regression.png)

__Classification__
$x$ is continuous and $y$ is discrete 
![width:350px](contents/figures/classification.png)

---


# 3. Regression

![bg left 60%](contents/figures/regression.png)

--- 

## Regression

__Dataset:__ the set of $N$ samples $x_i$ and corresponding labels $y_i$ such as 

$$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$$

__Problem:__ optimize the parameters $\theta$ of  $f_\theta$ to predict $y$ from $x$. Find the optimal parameters $\theta^*$ that minimize $\mathcal{L}$, such as

$$\theta^* = \underset{\theta}{\arg\!\min }\mathcal{L}\Big(f_\theta(x), y\Big).$$

![bg right 55%](contents/figures/linear_regression_true.png)

---

## Linear regression

__Linear model:__ coefficients $\theta = (a, b) \in \mathbb{R}^2$ that map $x$ to $y$ with 

$$f_\theta : x \mapsto y= ax + b.$$

__Loss function:__ for instance mean squared error: 
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \left( f_\theta(x_i) - y_i \right)^2.$$

How do we find $\theta^*$ to minimize $\mathcal{L}$?

![bg right 55%](contents/figures/linear_regression_true.png)

---

## Naive attempt: grid search

__Implementation:__ find $\theta^*$ among regularly spaced tested values of $\theta$. 

__Pros:__ easy to implement, exhaustive search, uncertainty estimation.

__Cons:__ unscalable: if 0.1s / evaluation, then 2 parameters with 100 values each takes 1/4 hour. __For 5 parameters it takes more than 30 years!__

![bg right 55%](contents/figures/linear_regression_grid_search.gif)

---

## Random search

__Implementation:__ sample random values of $\theta$, keep the best one.

__Pros:__ easy to implement, scalable, uncertainty estimation, can include prior knowledge.

__Cons:__ not exhaustive, can be slow to converge.

![bg right 55%](contents/figures/linear_regression_random_search.gif)

---

## Gradient descent

__Implementation:__ estimate the gradient of $\mathcal{L}$ w.r.t. the parameters $\theta$, update the parameters towards gradient descent.

__Pros:__ converges faster than random search.

__Cons:__ gets stuck in local minima, slow to converge, needs differentiability. 

![bg right 55%](contents/figures/linear_regression_gradient_descent.gif)

---

## Gradient descent

__Implementation:__

1. Initiate model $\,\theta = (a_0, b_0)$
1. Compute gradient $\,\nabla \mathcal{L}(\theta)$
1. Update $\,\theta \leftarrow \theta - \eta \nabla \mathcal{L}(\theta)$
1. Repeat until convergence

__Hyperparameters:__ the __learning rate__ $\eta$ defines the update step.

</div>

![bg right 55%](contents/figures/linear_regression_gradient_descent.gif)



---

## How to deal with learning rate?

__Too slow__
![width:330](contents/figures/learning_rate_slow.png)

__Too fast__    
![width:330](contents/figures/learning_rate_fast.png)

__Just right__
![width:330](contents/figures/learning_rate_right.png)

Looking for the best learning rate is called  __hyperparameters tuning__.

---

## Curve fitting issues

__Underfitting__
![width:330](contents/figures/fit_under.png)
 
__Overfitting__
![width:330](contents/figures/fit_over.png)

__Just right__
![width:330](contents/figures/fit_right.png)

__How to prevent overfitting?__

---

## Splitting the dataset

__Underfitting__
![width:330](contents/figures/fit_under_test.png)

__Overfitting__
![width:330](contents/figures/fit_over_test.png)

__Just right__
![width:330](contents/figures/fit_right_test.png)

Splitting the dataset into a __training__ and a __testing__ set, monitor both performances

---

## Why so many regression algorithms?


Because of combination of models, losses, and regularizations. 

The [scikit-learn.org](https://scikit-learn.org/stable/) website provides a unified interface in a `greybox style`. The model selection is made by experience or __trial and error__.

![bg right 40% grayscale brightness:1.2](https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/250px-Scikit_learn_logo_small.svg.png)

<!-- _footer:  [www.scikit-learn.org](https://scikit-learn.org/stable/) -->

---

## Guidelines for exploring relevant models

![width:900](https://scikit-learn.org/stable/_downloads/b82bf6cd7438a351f19fac60fbc0d927/ml_map.svg)

<!-- _footer:  [www.scikit-learn.org](https://scikit-learn.org/stable/) -->

--- 

## Notebook 1

![bg right](https://wp1.aeris-data.fr/wp-content-aeris/uploads/sites/56/2021/03/QCK_1-scaled.jpg) 

Learning suspended sediment concentration from river turbidity <br>
![](contents/figures/notebook_1.png) 

<!-- _footer: Jupyter $\\times$ Obsera -->

---

# 4. Classification

![bg left 60%](contents/figures/classification.png)

---

## Supervised learning tasks

__Regression__
![width:350px](contents/figures/regression.png)
$x$ and $y$<br>are continuous

__Classification__
![width:350px](contents/figures/classification.png)
$x$ is continuous<br>and $y$ is discrete 

---

<!-- _footer: www.scikit-learn.org -->

## Classification algorithms

![](contents/figures/classification_comparison.png)
Here again, we have many possibilities.

---

## Classification

<div>

__Experience__: labels $y \in \{0, 1\}$ for two features $\mathbf{x} \in \mathbb{R}^2$.

__Task__: predict $\hat{y}$ of each sample $\mathbf{x}$.

__Performance__: how should we measure the performance of a classifier?

</div>

![bg right 70%](contents/figures/svm_data.png)

---


## Support vector machines



Support vector machines search the hyperplane of normal vector $\mathbf{w}$ and bias $b$ that split the classes.

> Note: in 2D, a hyperplane is a line.

The support vectors are the samples that are closest to the other class.

![bg right 70%](contents/figures/svm_support_vectors.png)

---

## Support vector machines



The decision function $f(\mathbf{x})$ dependson  the sign of the linear combination of the normal vector and the sample:

$$f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$$

The quantity to minimize is the __Hinge loss__:

$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N \max\left(0, 1 - f(\mathbf{x}_i) y_i\right)$$


![bg right 70%](contents/figures/svm.png)

---

## Support vector machines

The decision $f(\mathbf{x})$ is the sign the normal vector scalar the sample:

$$f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$$

We me minimize is the __hinge loss__:

$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N \max\left(0, 1 - f(\mathbf{x}_i) y_i\right)$$

__What about non linear problems?__

![bg right 70%](contents/figures/svm.png)

--- 

## Kernel trick for non-linear problems


The kernel trick allows to map the data to a __higher dimensional__ space
made from the input features where the problem is __linearly separable__. <br>
The __Radial Basis Functions__ (RBF) is an infinite kernel 

$$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\sigma^2}\right)$$

![bg right 50%](contents/figures/kernel_trick.png)

---

## Support vector classifier<br>with multiple classes

__SVC__ is a generalization of the SVM that digests more than two classes. 

Again, the decision function is linear in the kernel space. We can project it back to the data space to inspect it.

__How to select the best model?__


![bg right 80%](contents/figures/iris.png)

---

## Confusion matrix 

Performance of a classifier can be derived from the confusion matrix:
- __Accuracy__: fraction of correct predictions $= \frac{TP + TN}{TP + TN + FP + FN}$

- __Precision__: fraction of true positives among predicted positives $= \frac{TP}{TP + FP}$
- __Recall__: fraction of true positives among actual positives $= \frac{TP}{TP + FN}$

![bg right 80%](contents/figures/iris_confusion.png)

<!-- _footer: SVC applied to Iris dataset -->

---

## Confusion matrix

Performance of a classifier can be derived from the confusion matrix:
- __Accuracy__: fraction of correct predictions $= \frac{TP + TN}{TP + TN + FP + FN}$

- __Precision__: fraction of true positives among predicted positives $= \frac{TP}{TP + FP}$
- __Recall__: fraction of true positives among actual positives $= \frac{TP}{TP + FN}$

![bg right 70%](contents/figures/confusion_matrix.png)

---

## Decision trees

__Decision trees__ learn to predict $y$ based on feature values of $\mathbf{x}$ by recursively splitting the data into subsets based on feature thresholds.

![bg right 90%](contents/figures/decision_tree.png)

<!-- _footer: from [scikit-learn.org](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) -->

---

## Random forests

__Random forests__ are ensembles of decision trees that vote for $y$. All trees are trained on random subsets of the data and features to reduce overfitting.

![bg right 85%](contents/figures/random_forest.png)

<!-- _footer: from [scikit-learn.org](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html) -->

---

## Representation matters

<!-- ![bg right 90%](images/deep-learning-book/figure-1-1.png) -->
![bg right 45%](contents/figures/representation_matters.png)
There is no need for a complex model if you have a good __representation__ of the data.

---

## Choosing features

Hand-designed? 
Learned features?

![bg right 90%](contents/figures/choosing_features.png)

<!-- _footer: Goodfellow et al. (2016) -->

---

<!-- _footer: still from Valentine & Trampert (2012) -->

## Representation matters

<div>

We can see waveforms $\mathbf{x}\in\mathbb{R}^N$ as points of a $N$-dimensional space 

<img src="images/waveforms/waveform_0.png" width=700/>

Yet, seismic waveform do not occupy this space fully, likely very sparse.

### Dimension > Information

</div>

![bg right 80%](images/papers/valentine2012spot.png)

---

## Manifolds


Random sampling of a face.<br>__Likelihood to end up with a face?__<br>
![width:120](contents/data/deep-learning-figure-x-x-1.jpg)<br>
Likewise waveforms images lie on a __manifold__ of much lower dimension.

![bg right 80%](contents/figures/representation_matters_face.gif)

<!-- _footer: modified from Goodfellow et al. (2016) -->

---

## Seismo-volcanic signal classification

__Supervised learning__ experiences a set of examples containing features $\mathbf{x}_i \in \mathbb{X}$ associated with labels $\mathbf{y} \in \mathbb{Y}$ to be predicted from the features (here, classification). <br>

![bg right 90%](images/examples/malfante_2018.png)

<!-- _footer: Malfante et al. (2018) -->

---

## Seismo-volcanic signal classification

In this case, $\mathbf{x} \in \mathbb{R}^{3 \times N}$ 
and $\mathbf{y} \in [0, \ldots, 5]$. 

Which __representation__ of $\mathbf{x}$ works best?

![bg right 90%](images/examples/malfante_2018.png)

<!-- _footer: Malfante et al. (2018) -->

---

## Handcrafted features

![](images/examples/features.png)

<!-- _footer: Jasperson et al. (2022) -->

---

## Performance
Accuracy of the predictions measures the model's performance (= confusion matrix) <br><br> <img src="images/examples/malfante_accuracy.png" width=800/><br>
What is the guarantee that the features we choose are the best ones?

<!-- _footer: Malafante et al. (2018) -->


---

<!-- _footer: Jupyter $\\times$ Espinosa-Curilem et al. (2025) -->

## Notebook 2

Classification of seismo-volcanic events<br>
![](contents/figures/notebook_2.png)

![bg right grayscale](https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Chillan_y_Nevados_de_Chillan_1.jpg/2560px-Chillan_y_Nevados_de_Chillan_1.jpg)


---

<!-- _footer: Jupyter $\\times$  Brodu and Lague (2012) -->

## Notebook 3

Lidar cloud point classification
![](contents/figures/notebook_3.png)

![bg right grayscale](https://upload.wikimedia.org/wikipedia/commons/d/de/Forêt_fontainebleau_pins.jpg)

---

# 5. Unsupervised learning

![bg left 60%](contents/figures/clustering.png)

---

## Types of machine learning problems


__Supervised__
![width:330](contents/figures/classification.png)
__Predict__ $\mathbf{y}$ from $\mathbf{x}$ 
(regression, classification)

__Unsupervised__
![width:330](contents/figures/clustering.png)
Learn data __distribution__ 
$p(\mathbf{x})$ (clustering, reduction)

__Reinforcement__
![width:330](contents/figures/reinforcement_learning.png)
Learn a policy to maximize 
a __reward__ (gaming, robotics)

---

## Contents of this class make use of the scikit-learn library

![width:850px](https://scikit-learn.org/stable/_downloads/b82bf6cd7438a351f19fac60fbc0d927/ml_map.svg)

<!-- _footer:  [www.scikit-learn.org](https://scikit-learn.org/stable/) -->

---

## Learning the structure of the data without labels


__Clustering__
![width:340](contents/figures/clustering.png)
<br> Group similar data points  
based on some similarity criterion

__Dimensionality reduction__
![width:340](contents/figures/dimensionality_reduction.png)
<br> Find a low-dimensional 
representation of the data 


--- 

## Clustering: class-membership identification without labels

![width:900](contents/figures/clustering_comparison.png)

Again, many possibilities.

<!-- _footer: from [scikit-learn.org](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html) -->

---

## Definitions of clustering

![bg right 70%](contents/figures/k_means_initial.png)

- _Top-bottom_: partition the heterogeneous data into __homogeneous__ subsets

- _Bottom-up_: group the data samples based on some criterion of __similarity__

__We need to provide a definition of similarity or homogeneity__


---

## _k_-means clustering

![bg right 70%](contents/figures/k_means_final.png)


$k$-means is a clustering algorithm that partitions the data into $k$ clusters by minimizing the __inertia__:

$$
\arg\min_{\mathbf{C}} \sum_{i=1}^{k} \sum_{\mathbf{x}_j \in C_i} \|\mathbf{x}_j - \mu_i\|^2
$$

where $\mu_i$ is the centroid of cluster $C_i$.

--- 

## _k_-means clustering

In practice, we need to provide the number of clusters $k$, and the algorithm will find the best partition: 

1. __Initialize__ centroids $\mu_i$ randomly
2. __Assign__ samples to centroids
3. __Update__ centroids
4. __Repeat__ until convergence.

![bg right 70%](contents/figures/k_means.gif)

<!-- _footer: Wikipedia -->

---

## Spectral clustering

Spectral clustering uses the eigenvectors of the affinity matrix to find the clusters. 

1. Compute the __affinity matrix__
3. Get the first $m$ __eigenvectors__.
4. __Cluster__ the data using $k$-means.

![bg right 70%](contents/figures/spectral_clustering.png)


---

## Learning the structure of the data without labels


__Clustering__
![width:340 opacity:0.5](contents/figures/clustering.png)
<br> Group similar data points  
based on some similarity criterion

__Dimensionality reduction__
![width:340](contents/figures/dimensionality_reduction.png)
<br> Find a low-dimensional 
representation of the data 



---

## Eigenvectors of a matrix

The eigenvectors of a matrix are obtained by solving the eigenvalue problem:

$$
\mathbf{A}\mathbf{x} = d \mathbf{x}
$$

where $\mathbf{A}$ is a square matrix, $\mathbf{x}$ is the eigenvector, and $d$ is the eigenvalue.

![bg right 70%](contents/figures/eigenvectors.png)

---

## Singular value decomposition

The singular value decomposition (SVD) of a matrix $\mathbf{A}$ is defined as:
$$
\mathbf{A} = \mathbf{U}\mathbf{D}\mathbf{V}^T
$$
where $\mathbf{U}$ and $\mathbf{V}$ are orthogonal matrices, and $\mathbf{\Sigma}$ is a diagonal matrix.
![bg right 90%](contents/figures/svd.png)


---

## Principal component analysis


Principal component analysis (PCA) is a dimensionality reduction technique that uses eigendecomposition:

1. __Compute__ data covariance matrix
2. __Compute__ eigenvectors
3. __Project__ samples onto eigenvectors


![bg right 70%](contents/figures/dimensionality_reduction.png)


<!-- _footer: Wikipedia -->

---

## Independent<br>component analysis

Independent component analysis (ICA) is a dimensionality reduction technique that maximizes the independence of the components.

![bg right 90%](https://scikit-learn.org/stable/_images/sphx_glr_plot_ica_vs_pca_001.png)

---

## Principal vs. independent<br>component analysis

For blind source separation, ICA is preferred over PCA.

![bg right 90%](https://scikit-learn.org/stable/_images/sphx_glr_plot_ica_blind_source_separation_001.png)

---

## Kernel PCA, of course

__Kernel PCA__ is a non-linear dimensionality reduction technique that uses the kernel trick to project the data onto a higher-dimensional space

![bg right 90%](contents/figures/kernel_pca.png)

---

# 6. Deep learning

![bg left 70%](contents/images/deep-learning-book/figure-1-2.png)

---

## Artificial neuron unit

A __neuron__, or unit, takes a set of inputs $\bf x$ and outputs an activation value $h$, as
$$
h = \varphi\left(\sum_{i=1}^{N} w_i x_i + b \right)
$$
with $w_i$ the weights, $b$ the bias, $\varphi$ is the activation function, and $N$ is the number of inputs.

![bg right 80%](contents/figures/neural_net.png)

---

## The sigmoid unit

A __neuron__, or unit, transforms a set of inputs $\bf x$ into an output $h$, as

$$
h = \varphi\left(\sum_{i=1}^{N} w_i x_i + b \right)
$$

Common activation functions include the __sigmoid__ function, defined as

$$
\varphi(z) = \frac{1}{1 + e^{-z}}
$$

![bg right 75%](contents/figures/sigmoid.png)

---

## The rectified linear unit

A __neuron__, or unit, transforms a set of inputs $\bf x$ into an output $h$, as

$$
h = \varphi\left(\sum_{i=1}^{N} w_i x_i + b \right)
$$

Common activation functions include the __rectified linear unit__ (ReLU), defined as

$$
\varphi(z) = \max(0, z)
$$

![bg right 75%](contents/figures/relu.png)

---

## The multilayer perceptron

A __multilayer perceptron__ is a neural network with multiple hidden layers:
$$
\begin{align*}
h_i^{(1)} &= \varphi^{(1)}\left(\sum_j w_{ij}^{(1)}x_j + b_i^{(1)}\right)\\
h_i^{(2)} &= \varphi^{(2)}\left(\sum_j w_{ij}^{(2)}h_j^{(1)} + b_i^{(2)}\right)\\
y_i &= \varphi^{(3)}\left(\sum_j w_{ij}^{(3)}h_j^{(2)} + b_i^{(3)}\right)
\end{align*}
$$

![bg right 80%](contents/figures/neural_net.png)

---

## The multilayer perceptron

<div>

A __multilayer perceptron__ is a neural network with multiple hidden layers. Generally speaking (omitting biases):
$$
y = \varphi^{(\ell)}\left(\mathbf{W}^{(\ell)}\varphi^{(\ell - 1)}\left(\mathbf{W}^{(\ell - 1)} \ldots \varphi^{(1)}\left(\mathbf{W}^{(1)}\mathbf{x}\right) \ldots \right)\right)
$$
</div>

![bg right 80%](contents/figures/neural_net.png)    

---

## Solving the XOR problem


Multi-layer perceptrons that solves the XOR problem with binary activations:<br>
![width:400](contents/figures/neural_net_solve_xor.png)


![bg right 75%](contents/figures/xor.png)


<!-- _footer: Section 6.1 of Goodfellow et al. (2016) -->

---

## Gradient descent 


We note $f_\theta(x): x \mapsto y$ the model, where $\theta$ are the parameters of the model (including biases and weights).

1. __Learning__ is the process of finding the parameters $\theta^*$ that minimize the loss $\mathcal{L}$.
2. The __backpropagation__ computes the loss function gradient with respect to $\theta$.
3. The __gradient descent__ updates $\theta$ in the direction of the steepest descent.

![bg right 56%](contents/figures/linear_regression_gradient_descent.gif)

---

## Gradient computation with backpropagation

1. __Initialization__: the weights are initialized randomly, the biases to zero
2. __Feed forward__: the input is propagated through the network to compute the output
3. __Loss__: the loss is computed between the output and the target
4. __Back propagation__: computation of the gradient from the loss to the input
5. __Gradient descent__: update the parameters in the direction of the steepest descent

---

## Gradient-based optimization

Once the gradient is computed, the parameters are updated using the __gradient descent__ algorithm:

$$
\begin{align*}\\
\theta &\leftarrow \theta - \eta \frac{\partial \mathcal L}{\partial \theta}
\end{align*}
$$

where is $\eta$ the __learning rate__ that controls the size of the update.

![bg right 56%](contents/figures/linear_regression_gradient_descent.gif)

---

## Gradient descent issues

- __Local minima__: getting stuck in a local minimum.

- __Sattling points__: behaves as a local minimum but is not.

- __Plateau__: flat loss function, vanishing gradient, slow convergence.

![bg right 56%](contents/figures/linear_regression_gradient_descent.gif)

---


## Gradient descent common issues with plateau



![width:400](contents/figures/relu.png) ![width:400](contents/figures/sigmoid.png)
__Plateau__ are flat regions of the loss function where the gradient is zero. This can happen with activation functions such as the sigmoid function with saturation. It can also happen with the ReLU function for inputs with negative values.

---


## Gradient-descent tricks to avoid issue

- __Learning rate__: set up, and maybe adapt it. 
- __Momentum__: use the gradient of the previous iteration to update the parameters.
- __Normalization__: normalize the inputs of each layer.
- __Stochastic gradient descent__: use a mini-batch of samples to compute the gradient.
- __Dropout__: randomly drop some neurons during training.

---

## Learning rate

The __learning rate__ controls the size of the update of the parameters:

$$
\theta \leftarrow \theta - \eta \cfrac{\partial \mathcal L}{\partial \theta}
$$

We must look the best learning rate via __hyperparameters tuning__. We can also __adapt__ the learning rate.


![bg right 90%](contents/figures/learning_rate_slow.png)
![bg right 90%](contents/figures/learning_rate_fast.png)

---

## Gradient descent momentum

<div>

The __momentum__ is a technique to accelerate the gradient descent by adding a fraction of the gradient of the previous iteration:
$$
\begin{align*}
p &\leftarrow \alpha p - \eta \frac{\partial \mathcal L}{\partial \theta}\\
\theta &\leftarrow \theta + p
\end{align*}
$$
where $\alpha$ is the a damping parameter, and $v_i$ is the __velocity__. Lower values of $\alpha$ give more weight to the current gradient, higher values give more weight to the previous gradients.

</div>

![bg right 90%](contents/figures/gradient_descent_momentum_lr1_mom0.6.png)
![bg right 90%](contents/figures/gradient_descent_momentum_lr0.5_mom1.png)



<!-- _footer: From Zhang et al. (2021) -->

---

## Data normalization

<div

To avoid getting in the saturation of sigmoidal activation functions, it is important to normalize the data. This can be done by __normalizing the input and the features__:

$$
\hat x_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}
$$

where $\mu_i$ is the mean of the input, $\sigma_i$ is the standard deviation of the input, and $\epsilon$ is a small constant to avoid division by zero. You can also apply the normalization after the activation function.

</div>

---

## Training

<div>

The __training curves__ are a good way to monitor the training of a model.

- Slow: increase the learning rate.
- Growing: decrease the learning rate.
- Cross-validation: within 0.0001 to 0.1

</div>

![bg right 80%](contents/figures/training_curve.png)

---

## Stochastic gradient descent

<div>

The gradient of the loss function with respect to the parameters $\theta$ is computed using the __full-batch gradient descent__ equal to:

$$
\frac{\partial \mathcal L}{\partial \theta} = \frac{1}{N} \sum_{i=1}^N \frac{\partial \mathcal L^{(i)}}{\partial \theta}\\
$$

The __stochastic gradient descent__ is a technique to compute the loss gradient from every sample in the dataset at each iteration.

</div>

![bg right 60%](contents/figures/stochastic_gradient_descent.png)

---

## Splitting the dataset

__Underfitting__
![width:330](contents/figures/fit_under_test.png)

__Overfitting__
![width:330](contents/figures/fit_over_test.png)

__Just right__
![width:330](contents/figures/fit_right_test.png)

Splitting the dataset into a __training__ and a __testing__ set, monitor both performances

---


## The right complexity

The __model complexity__ is roughly the number of parameters of the model. The __model generalization error__ is the error on the test set.

![bg right 90%](images/models/complexity.png)

---

## Regularization

__Regularization__ is a technique to control overfitting by adding a penalty term $\mathcal{R}$ to the loss function. The __regularization parameter__ $\lambda$ controls the strength of the regularization.

$$
\mathcal{L}_\mathrm{reg} = \mathcal{L} + \lambda \mathcal{R} = \mathcal{L} + \lambda \|\mathbf{\theta}\|^2_2
$$

![bg right 90%](images/models/wd.png)

<!-- _footer: From Goodfellow et al. (2016) -->

---

<!-- _backgroundColor: white -->

## A fully connected network for solving the MNIST classification

<!-- _footer: LeCun et al. (1998) -->

Handwritten digits set of grayscale images $x \in \mathbb{R}^{28 \times 28}$ and classes $y \in \{0, \dots, 9\}$.<br>
![width:700](images/datasets/mnist.png)
__Goal__: predict the number encoded in the pixels.


---

<!-- _footer: adamharley.com (Harley, 2015) -->

## A fully connected network for solving the MNIST classification

https://adamharley.com/nn_vis/mlp/3d.html

---

## Example in seismology: fully-connected autoencoder

<!-- _footer: from Valentine and Trampert (2012) -->

<div>

Training of a fully-connected autoencoder on real seismic data. 

This is an __unsupervised__ learning task: the input and output are the same.

![width:400](images/examples/valentine_2.png)

</div>

![bg right 89%](images/examples/valentine_ae.png)

---

## Example in seismology: fully-connected autoencoder

<!-- _footer: from Valentine and Trampert (2012) -->

- We __learned__ a low-dimensional representation for the seismic data.

- These are the __latent variables__ of the autoencoder.

![bg right 90%](images/examples/valentine_1.png)

---

## Example in seismology: fully-connected autoencoder

<!-- _footer: from Valentine and Trampert (2012) -->

<div>

Example applications:
- quality assessment
- compression

</div>

![bg right 90%](images/examples/valentine_3.png)

---

![bg left 70%](images/models/invariance.jpg)

# 7. Convolutional neural networks


---

## Limitations of fully connected networks

- Translation
- Rotation
- Scaling
- Shearing
- Illumination
- Occlusion

![bg right 80%](images/models/invariance.jpg)

---

## Example: the handwritten digits

<!-- _backgroundColor: white -->

<!-- _footer: LeCun et _al._ (1998) -->

Handwritten digits set of grayscale images $x \in \mathbb{R}^{28 \times 28}$ and classes $y \in \{0, \dots, 9\}$.<br> 
<img src="images/datasets/mnist.png" style="width: 70%;">

---

## Limitations of fully connected networks


<div>

An image may be of $200 \times 200$ pixels $\times 3$ color channels. With a __fully connected network__ with $1000$ hidden units, we would have $N = 200 \times 200 \times 3 \times 1000 = 120$M parameters. <br><br>__This clearly does not scale to large images.__

</div>

![bg right 80%](images/models/densely.png)

---

<!-- _backgroundColor: white -->

## Convolutional neural networks

<div>

__Convolutional layers__ are a type of layer that are used in convolutional neural networks. They are composed of a set of learnable filters.

![width:900](images/models/convlay.png)

Each hidden unit look a local content from the input image, althought the weights are shared across the entire image.

</div>

---

## Convolutional neural networks

<div>

Discrete image convolution:

$$ (A * B)_{ij} = \sum_n \sum_m A_{nm}B_{i-n, j-m} $$

where $A$ is a input image, and $B$ is a convolutional kernel (weights) to learn.

> Convolutional layers extract local features from the input image ≠ fully connected layers that extract global features.

</div>

![bg right 50%](images/models/no_padding_no_strides.gif)

<!-- _footer: From Vincent Dumoulin, Francesco Visin (2016) -->

---

<!-- _backgroundColor: white -->

## Convolution operation

<div>

$$(A * B)_{ij} = \sum_n \sum_m A_{nm}B_{i-n, j-m}$$
![](images/datasets/zebra_filtered.jpeg)

</div>

---

<!-- _backgroundColor: white -->

## Convolution operation

<div>

$$(A * B)_{ij} = \sum_n \sum_m A_{nm}B_{i-n, j-m}$$
![](images/datasets/zebra_edges.jpeg)

</div>

---

<!-- _backgroundColor: white -->

## Convolution unit

![](images/models/zebra_conv.png)

---

<!-- _backgroundColor: white -->

## Convolutional neural network: example with VGG16


Now, we can understand this winning architecture for image classification.<br>
![width:850](images/models/vgg16.png)
Note the last three layers are __fully connected__.
When extracting low-dimensional data from images, this is often needed.

</div>

---

<!-- _backgroundColor: white -->

## Convolutional neural network: example with VGG16

<!-- _footer: from Zeiler and Fergus (2013) -->

<div>

Here are the __filters from the first layer__ of VGG16 after training on 100k+ images. These filters collect various shapes, scales, colors, etc.

![](images/models/vgg16.png)

</div>

![width:500](images/models/vgg_layer_1.png)

---

<!-- _footer: adamharley.com (Harley, 2015) -->

## A convolutional network for solving the MNIST classification

### https://adamharley.com/nn_vis/cnn/3d.html


---

## Deep-learning applications in seismology


- Signal detection, pattern recognition
- Classification
- Source localization from sparse or evolving datasets
- Denoising and compression

![bg right 80%](images/references/seismic_signal_class.png)

---

## ConvNetQuake



__Features__: 3-comp. waveform $x \in \mathbb{R}^{N \times 3}$
__Target__: prob. of event in cell $1$ to $6$
__Loss__: cross-entropy with regularization $\mathcal{L} = - \sum_c q_c \log p_c + \lambda \| \mathbf{w}\|^2_2$ 

![width:350](images/models/perol_2.png)


![bg right 80%](images/models/perol_1.png)


<!-- _footer: From Perol et al. (2016) -->

---

## PhaseNet

__Features__: 3-component seismic signal $x \in \mathbb{R}^{3000 \times 3}$. 

__Targets__: probabilities $p_i(x)$ of $P$, $S$, and $N$oise over time $= y \in \mathbb{R}^{3000 \times 3}$

![bg right 80%](images/models/beroza_example.png)


<!-- _footer: From Zhu et al. (2016) -->

---

<!-- _backgroundColor: white -->

## Seismic phase picking with PhaseNet

__Features__: 3-component seismic signal $x \in \mathbb{R}^{3000 \times 3}$
__Targets__: probabilities $p_i(x)$ of $P$, $S$, and $N$oise over time $= y \in \mathbb{R}^{3000 \times 3}$
__Loss__: cross-entropy $\mathcal{L} = -\sum_i\sum_x p(x)\log(q(x))$<br>
![width:800](images/models/unet_phasnet.jpg)

</div>

<!-- _footer: From Zhu et al. (2016) -->

---

<!-- _backgroundColor: white -->

## Seismic phase picking with PhaseNet

__Features__: 3-component seismic signal $x \in \mathbb{R}^{3000 \times 3}$
__Predictions__: likelihood $q_i(x)$ of $P$, $S$, and $N$oise over time<br>
![](images/models/beroza_2.png)

<!-- _footer: From Zhu et al. (2016) -->

---

## Seismic phase picking with PhaseNet
<!-- _backgroundColor: white -->

__Features__: 3-component seismic signal $x \in \mathbb{R}^{3000 \times 3}$
__Predictions__: likelihood $q_i(x)$ of $P$, $S$, and $N$oise over time<br>
![](images/models/beroza_3.png)

<!-- _footer: From Zhu et al. (2016) -->

---

## Transfer learning 

<div>

__Transfer learning__ is the use of a pre-trained model $f_\alpha = f_{\theta^*}$ on a new task as a initial point for training a new model $f_\alpha \rightarrow f_{\alpha^*}$.

__Fine-tuning__ is the partial re-training of a pre-trained model on a new task, while keeping the weights of the pre-trained layers fixed.

</div>

![bg right 90%](images/examples/scedc_mapplot.png)

---

## Deep-learning libraries in Julia and Python

<div style="flex-basis: 25%;">

### Warning

Libraries are constantly evolving, and the documentation is often incomplete.

</div>
<div style="flex-basis: 50%; columns:2;">
<img src="images/examples/logo_sklearn.png" width=200/>
<img src="images/examples/logo_tf2.png" width=300/>
<img src="images/examples/logo_keras.png" width=250/>
<img src="images/examples/logo_pytorch.png" width=250/>
<img src="images/examples/logo_julia.png" width=150/>
<img src="images/examples/logo_seisbench.svg" width=400/>

</div>

---

## TensorFlow Playground


### https://playground.tensorflow.org

---

## Interesting online projects

### https://quickdraw.withgoogle.com/data


---

## Deep unsupervised learning

Deep neural networks can be used for unsupervised learning:
__Autoencoders__: learn a low-dimensional representation of the data.
__Generative adversarial networks__: learn a generative model of the data.

![width:900px](https://fr.mathworks.com/discovery/autoencoder/_jcr_content/mainParsys/image.adapt.480.medium.svg/1665035671723.svg)

---

## Autoencoders

Autoencoders are neural networks that learn a low-dimensional representation of the data. They are composed of an __encoder__ and a __decoder__.

The input $\mathbf{x}$ is encoded into a latent representation $\mathbf{z}$, and decoded into $\mathbf{x}'$.

The loss function is the difference between the input and the reconstruction: 

$$
\mathcal{L} = \|\mathbf{x} - \mathbf{x}'\|^2
$$


![bg right 80%](images/examples/valentine_ae.png)

---

## Convolutional autoencoders

Convolutional autoencoders use convolutional layers instead of fully connected layers.
They are used for image denoising, compression, and quality assessment.

![width:1000px](https://miro.medium.com/v2/resize:fit:1400/1*gzJAJDLDavH_W7Zv2M2J7w.png)

---

<!-- _class: titlepage-->

# _The end!_


---

<!-- _color: white -->
<!-- _backgroundColor: #333 -->
<!-- _footer: Jupyter $\\times$ PhaseNet -->

## Notebook 4 (teaser)

### Pick _P_ and _S_ waves within continuous seismograms

![bg right 80%](images/papers/zhu2018phasenet.png)