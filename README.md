# CycleGAN-Pytorch

A simple and efficient implementation of CycleGAN in PyTorch. This repository is a derivative of my work as a Junior AI Consultant.

## Getting Started

To get started, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/saitejeswar1/CycleGAN-Pytorch.git
   cd CycleGAN-Pytorch
   ```
 2. Create a virtual environment using the provided Conda environment file:
```bash
conda env create -f environment.yml
conda activate <env_name>  # Replace <env_name> with the name of the environment
```
## Features
- A clean and simple implementation of CycleGAN.
- Built entirely using PyTorch.
- Easy to extend and modify for research and experimentation.

## About CycleGAN
CycleGAN is a type of Generative Adversarial Network (GAN) that specializes in image-to-image translation tasks without requiring paired datasets. Some popular use cases include:
- Transforming photos into artworks.
- Style transfer between different visual domains.
- Enhancing image quality in various fields such as medical imaging or satellite imagery.

## Usage
### Training
To train the CycleGAN model, ensure your dataset is structured properly and run the training script:
```bash
python train.py --dataset <dataset_name>
```
### Testing
To test the trained model, use the testing script:
```bash
python test.py --model_path <path_to_model> --input_dir <input_images_dir>
```
### Configuration
You can customize the training and testing parameters via the configuration file or by passing arguments to the scripts.

## Acknowledgements
- PyTorch: https://pytorch.org/
- Original CycleGAN Paper: https://arxiv.org/abs/1703.10593
