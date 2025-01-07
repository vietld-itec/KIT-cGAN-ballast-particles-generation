README

# KIT-cGAN: Ballast Particles Generation using cGAN

## Description

This repository contains a Python implementation of a Conditional Generative Adversarial Network (cGAN) for generating ballast particles. The provided code allows you to configure the cGAN model, train it, and save the generated outputs.

## Requirements

Hardware: GPU is recommended for faster training.

Programming Language: Python 3.x

## Dependencies:
```
numpy

torch

yaml

csv

torch.utils.data
```

## Install dependencies using:
```
pip install numpy torch pyyaml
```

## Files

```main.py```: Contains the main function for training the cGAN model.

```config.yaml```: YAML file for specifying hyperparameters (e.g., g_hid, d_hid, lr_range).

```csv/csvdat360plus.csv```: Example dataset file used for training.

```main.py```: Placeholder for custom implementations (e.g., CustomArgs, CustomDataset, run_model, save_model).

## Configuration

Modify ```config.yaml``` to define the following:

```g_hid: [128, 256, 512]```     # Generator hidden layer sizes
```d_hid: [128, 256, 512]```     # Discriminator hidden layer sizes
```lr_range: [0.001, 0.0005, 0.0001]```  # Learning rate options

## Usage

### Prepare the Dataset

Place your dataset file in the ```csv/```  folder. Ensure it matches the required format.

### Set Up the Configuration

Edit ```config.yaml``` to specify the desired hyperparameters.

### Run the Script
Execute the script with the following command:

```python main.py <config_path> <g_index> <d_index> <lr_index> [<nsamples>]```

```<config_path>```: Path to the config.yaml file.

```<g_index>```: Index of the generator hidden layer size.

```<d_index>```: Index of the discriminator hidden layer size.

```<lr_index>```: Index of the learning rate.

```[<nsamples>]``` (optional): Number of samples (default: 360).

Example:

```python main.py config.yaml 3 6 0```

Output

Generated models and logs will be saved in a folder named ```selected-vcgan-<nsamples>```.

A CSV file named ```combination.csv``` will record the training details.

Customization

Replace ```main.py``` with the actual implementations of:

```CustomArgs```: Defines arguments for the model.

```CustomDataset```: Custom dataset loader.

```run_model```: Executes the cGAN training.

```save_model```: Saves the trained models.

## Example Config YAML

```g_hid: [128, 256, 512]```
```d_hid: [128, 256, 512]```
```lr_range: [0.001, 0.0005, 0.0001]```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or issues, please contact:

Viet Le Dinh: vietld@kumoh.ac.kr

Phone: (+82)010 2920 1514
