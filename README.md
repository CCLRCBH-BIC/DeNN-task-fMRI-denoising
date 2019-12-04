DeNN-task-fMRI-denoising
================================
### A deep neural network algorithm to denoise task fMRI data
### Introduction
This project aims to clean task fMRI data based on the task design matrix. With the assumption that neuronal activity is limited in gray matter, we maximized the correlation difference between gray matter time series and non-gray matter time series with task design matrix. Please note that any task stimuli not modelled in the design matrix is treated as "noise", thus it is necessary to re-denoise the data if a different design matrix is specified. We have tested the toolbox with simulated fMRI data, regular fMRI data (TR >= 2 seconds) and fast fMRI data (TR = 0.72 second) and overall achieved better performance than traditional denoising techniques. If you have any questions, please send an email to Zhengshi yang (yangz@ccf.org).


### Citations
If you find this toolbox is useful in your project, plaese cite our work.
*  **Zhengshi Yang**, Xiaowei Zhuang, Karthik Sreenivasan, Virendra Mishra, Tim Curran, and Dietmar Cordes (2019). A robust deep neural network for denoising task-based fMRI data: An application to working memory and episodic memory. Medical Image Analysis.

### Requited libraries
- [Matlab](https://www.https://www.mathworks.com/products/matlab.html): Matlab is used to extract time series from NIFTI file and save .mat file for denoising in Python.
- [Python](https://www.python.org/downloads/): Python 3 by default
- [Keras](https://www.https://keras.io/): Keras with Theano as backend is used in this project.
- [Theano](https://www.http://deeplearning.net/software/theano/):
- [numpy](http://www.numpy.org/):
- [scipy](https://www.https://www.scipy.org/):

### How to run the toolbox
*  **HCP_GLManalysisv2.m**: This file can be revised to the data directory you are working on to extract time series from fMRI NIFTI files and other necessary information required for denoising the data. Function HCP_tgtlureDesignmatrix.m is used in this file to get event-related design matrix from the eprime output for Human Connectome Project working memory task fMRI data. The function HCP_blockDesignmatrix is used to extract block design matrix from the eprime output. Canonical hemodynamic response function is used. The output .mat file is used for denoising fMRI data.
*  **HCP_sub.py**: This file uses the output .mat file from HCP_GLManalysisv2.m and run our denoising neural network to clean task fMRI data. Revise the data path to your own directory before running this script.
