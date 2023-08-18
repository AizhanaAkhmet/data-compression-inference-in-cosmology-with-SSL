# Data Compression and Inference in Cosmology with Self-Supervised Machine Learning
This repository contains the code used for the paper "Compression of Cosmological Data and Inference with Self-Supervised
Machine Learning" submitted to the ICML 2023 Workshop on Machine Learning for Astrophysics and "Data Compression and Inference in Cosmology with Self-Supervised Machine Learning" (in preparation).

## Software dependencies and datasets
This code uses `numpy`, `scipy`, `matplotlib`, `scikit-learn`, and `pytorch` packages. 

The data for mock lognormal fields is generated with the `powerbox`[^1] and `pyccl`[^2] packages. Simulation-based inference on SSL summaries is conducted with the `sbi`[^3] package.

The total matter density fields are from the IllustrisTNG and SIMBA suites of the CAMELS simulations[^4].

## Code description
The code is organized according to the three datasets used in the study: mock lognormal fields (`ln_fields`), total matter density fields from CAMELS simulations (`camels_fields`), and toy power spectra with various baryonic effects (`baryonic_effects_toy_Pk`). Each folder contains Jupyter notebooks to generate, plot, and analyze the datasets. 
The `trained_models` folders include trained neural network models used to quote the results and the analysis in the paper.

The `utils_modules` folder contains help functions used throughout the notebooks, including self-supervised loss function, custom neural network architectures for the encoder and projector neural networks, and custom dataset classes for SSL. 

[^1]: <https://powerbox.readthedocs.io/>
[^2]: <https://ccl.readthedocs.io/>
[^3]: <https://www.mackelab.org/sbi/>
[^4]: <https://camels.readthedocs.io/>
