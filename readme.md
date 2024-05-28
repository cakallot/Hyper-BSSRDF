# Hyper-BSSRDF Project

## Project Overview

The Hyper-BSSRDF project aims to develop a neural network system that can predict the parameters of a subsurface scattering (BSSRDF) model using a hypernetwork. The project involves two main components:

1. **Hypernetwork**: Predicts the weights for the BSSRDF network based on material parameters.
2. **BSSRDF Network**: Takes coordinates as input and predicts RGB values representing the subsurface scattering effects.

## Requirements

To set up and run this project, you need the following packages:

- torch==1.8.1+cu102
- torchvision==0.9.0+cu102
- torchaudio==0.8.0
- torchmeta==1.8.0
- tqdm
- matplotlib
- scikit-learn

Install these packages using the following command:

```
pip install torch==1.8.0+cu102 torchvision==0.9.0+cu102 torchaudio==0.8.0 torchmeta tqdm matplotlib scikit-learn --extra-index-url https://download.pytorch.org/whl/cu102
```

## Dataset Preparation

The dataset should be a PyTorch `TensorDataset` with the following structure:

- **Inputs**: Tensor of shape `[num_samples, 9]` where the first 4 columns are material parameters and the remaining 5 are other relevant features.
- **Outputs**: Tensor of shape `[num_samples, 3]` representing the RGB values.

### Example Dataset Structure

```
import torch

data = {
    'input_tensor': torch.randn(2000000, 9),  # Replace with actual data
    'output_tensor': torch.randn(2000000, 3)  # Replace with actual data
}

torch.save(data, 'new_materials_dataset.pth')
```

## Project Structure

The project includes the following main files:

- **dataset.py**: Handles dataset loading and preprocessing.
- **model.py**: Contains model definitions for the Hypernetwork and BSSRDF network.
- **train.py**: Training script for the models.
- **evaluate.py**: Evaluation script for the models.

## How to Run the Project

### Step 1: Prepare the Dataset

Ensure the dataset is saved in the correct format as described above and stored in `new_materials_dataset.pth`.

### Step 2: Train the Models

Run the training script to train the overfitting BSSRDF network.

```
python train_bssrdf.py
```

### Step 3: Evaluate the Models

After training, run the evaluation script to assess the model's performance.

```
python test.py
```

## Training Process

### First Phase: Training Both Networks

1. **Initialize Models**: Randomly initialize the Hypernetwork and BSSRDF network.
2. **Data Sampling**: Sample a material from the dataset and its associated data.
3. **Forward Propagation**: Perform forward propagation through both networks.
4. **Loss Calculation**: Compute the loss and update the weights of both networks.

### Second Phase: Fine-Tuning

1. **Data Sampling**: Sample a material and split the data into two subsets.
2. **Fine-Tune Hypernetwork**: Update the Hypernetwork using the first subset.
3. **Fine-Tune BSSRDF Network**: Update the BSSRDF network using the second subset.

## Evaluation

Evaluate the model's performance using metrics such as PSNR, MSE, and MAE. The evaluation script will load the trained models and compute these metrics for the test dataset.