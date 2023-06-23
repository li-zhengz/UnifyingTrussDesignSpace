# UnifyingTrussDesignSpace
This repository contains the generative modeling framework and code for constructing a unified design space for the use of inverse design with truss lattices as described in 'Unifying the design space of truss metamaterials by generative modeling'. 

The corresponding dataset used in this work can be found under [ETH Zurich research colletion](https://doi.org/10.3929/ethz-b-000618078).

## Requirements
- Python (tested on version 3.10.4)
- Python packages:
	- Pytorch (tested on CUDA version 11.0)
	- Pandas
	- NumPy
	- SciPy
	- [prefetch_generator](https://pypi.org/project/prefetch_generator/)

## File descriptions
- `__fwdTrain__`:
  - `main.py`: trains the presented generative modeling framework and the property predictor using the dataset given above.
  - `validation.py`: loads the pre-trained models and predicts the reconstructed truss structures and their corresponding effective stiffness (given in Voigt notation) for a test dataset. Additionally, it randomly samples 1000 points from a multivariate Gaussian distribution and calculates the validity score (the percentage of random samples that can be translated into a valid truss structure).
- `__invOpt__`:
  - `invOpt.py`: inverse design of structures with desired properties using gradient-based optimization.
- `__models__`:
  - `parameters.py`: hyperparameters used in training (neural network model shape, training epochs, learning rate, loss weight, etc.).
  - `model.py`: defines the structure of the VAE model `vaeModel` and the property predictor `cModel`.
  - `utils.py`: data postprocessing.
- `__results__`: saves the trained model, including the VAE model and the property predictor.

## Quick start
To start with the code, simply clone this repository via
```
git clone https://github.com/li-zhengz/UnifyingTrussDesignSpace.git
```
and run the corresponding script (estimated runtime of around 1 minute).
### Generative modeling and forward property prediction
After cloning this repo and running the `validation.py` script, you can obtain the reconstructed adjacency matrix, node positions, predicted effective stiffness, and the ground truth values for a holdout test dataset of size 2000 (estimated runtime of around 2 minutes). 

To run the code on your own data, the truss structures input must be provided in the form of the adjacency matrix `adj` and node offsets `x` as described in the publication. By loading the pre-trained neural network models, you can get the encoded latent representations by 

```python
encoded, mu, std = model.encoder(adj, x)
```
The jointly-trained property predictor predicts the effective stiffness matrix by 
```python
c_pred = c_model(c_input)
```
Note that the output `c_pred` is given in Voigt notation and only contains the following 9 independent components of the stiffness tensor $(C_{1111}, C_{1122}, C_{1133}, C_{2222}, C_{2233}, C_{3333}, C_{2323}, C_{3131}, C_{1212})$.
The variational autoencoder (VAE) reconstructs the truss structure by
```python
adj_decoded, x_decoded = model.decoder(encoded)
```
### Inverse design
The continuous latent representation of truss lattices allows for the use of gradient-based optimization to search for structures with desired properties. An example optimizing for maximum effective Young's moduli $E_{33}$ can be found in `invOpt.py`.
You can use the code to optimize for other properties by defining the target property in `opt_target`, e.g., 
```python
# target property names: ['E11', 'E22', 'E33', 'G23', 'G31', 'G12', 'v21', 'v31', 'v32', 'v12', 'v13', 'v23']
opt_target = ['E33'] 
```
The obtained inverse designs of truss structures are saved in the form of their adjacency matrix and node positions. 
