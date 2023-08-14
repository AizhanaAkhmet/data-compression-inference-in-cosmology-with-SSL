# Compression of Cosmological Data and Inference with Self-Supervised Machine Learning
This repository contains the code used for the paper "Compression of Cosmological Data and Inference with Self-Supervised
Machine Learning" submitted to the ICML 2023 Workshop on Machine Learning for Astrophysics.

## Software dependencies and Datasets
This code uses `numpy`, `scipy`, `scikit-learn`, and `pytorch` packages. 

The data for mock lognormal fields is generated with the `powerbox` package (<https://powerbox.readthedocs.io/>). 
The total matter density fields are from the IllustrisTNG and SIMBA suites of the CAMELS simulations (<https://camels.readthedocs.io/>).

## Code
The code is organized according to the three datasets used in the study: mock lognormal fields (`ln_fields`), total matter density fields from CAMELS simulations (`camels_fields`), and toy power spectra with various baryonic effects (`baryonic_effects_toy_Pk`). Each folder contains Jupyter notebooks to generate, plot, and analyze the datasets. 
The `trained_models` folders include trained neural network models used to quote the results and the analysis in the paper.
