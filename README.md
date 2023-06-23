# UnifyingTrussDesignSpace
This repository contains the generative modeling framework and code for constructing a unified design space for the use of inverse design with truss lattices as described in 'Unifying the design space of truss metamaterials by generative modeling'. The corresponding dataset can be found under [ETH Zurich research colletion](https://www.research-collection.ethz.ch/handle/20.500.11850/618078).

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
- `__models__`
		- `parameters.py`: hyperparameters used in training (neural network model shape, training epochs, learning rate, loss weight, etc.).
		- `model.py`: defines the structure of the VAE model `vaeModel` and the property predictor `cModel`.
		- `utils.py`: data postprocessing.
- `__results__`: saves the trained model, including the VAE model and the property predictor.
