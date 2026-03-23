# Data Pipeline Report: PyTorch Dataset Loader

This report details the implementation of `dataset_loader.py`, which is the bridge connecting your physical image files (the 15k benign images and ~9.3k malimg images) to the PyTorch CNN training loop.

## 1. Goal
Load images from the `archive/malimg_dataset` folder structure, apply the standard pre-processing (converting to PyTorch Tensors), and strictly enforce a **Binary Classification Scheme**:
*   `0` = Benign (from the `benign` folder)
*   `1` = Malware (from the 25 other family folders)

## 2. Core Mechanism: `ImageFolder` & Target Transforms
PyTorch provides a highly optimized class called `torchvision.datasets.ImageFolder`. 
When you point `ImageFolder` at a directory (e.g., `train/`), it automatically looks at the subdirectories and assigns an integer index to each class based on **alphabetical order**.

### The Alphabetical Issue
Since "benign" starts with the letter 'b', it sits alphabetically between the malware families (e.g., after `Agent.FYI` and before `C2LOP.P`). 
By default, `ImageFolder` would assign a random number like `6` to `benign` and numbers `0-5, 7-25` to malware.

### The Binary Mapping Solution
To enforce your `0` and `1` requirement, the script dynamically identifies the exact assigned index of the `benign` folder using `dataset.classes.index('benign')`.

We then pass a `target_transform` function to the dataset. Right before PyTorch passes an image to the CNN, this transform intercepts the true label and overwrites it:
```python
# If the original folder was 'benign' -> return 0
# If the original folder was anything else -> return 1
```
This guarantees absolute mathematical certainty that your CNN only ever sees `0` for benign and `1` for malware.

## 3. Countering Dataset Imbalance (Class Weights)
As discussed previously, you have:
*   ~12,000 Benign training images
*   ~7,400 Malware training images
*(There is an imbalance leaning towards Benign).*

If we train the CNN natively, it will predict "Benign" slightly more often just to get a higher baseline accuracy, without actually learning the features.

### The `calculate_class_weights` Function
The script includes a mathematical function that counts the exact number of malware and benign samples loaded inside the training set. It calculates a penalty weight for each class based on the formula:

> `Weight = Total_Samples / (Number_of_Classes * Class_Count)`

*   Because Malware has fewer samples, its weight will be **higher** (e.g., `1.3`).
*   Because Benign has more samples, its weight will be **lower** (e.g., `0.8`).

When you write the PyTorch Training Loop (Stage 2), you will pass these weights to the `nn.CrossEntropyLoss` function. This tells the network: *"If you get a Malware image wrong, the penalty is 1.3x more severe than if you get a Benign image wrong."* This forces the network to treat both classes equally, completely neutralizing the imbalance bias.

## 4. How to Use `dataset_loader.py`
The script exposes a main function: `get_data_loaders(data_dir, batch_size=32)`.

In your future CNN training script (`train.py`), you will simply write:
```python
from dataset_loader import get_data_loaders

train_loader, val_loader, test_loader, class_weights = get_data_loaders(
    data_dir=r"C:\Users\...\archive\malimg_dataset",
    batch_size=32
)
```
This seamlessly streams randomized, augmented, and binary-labeled batches of 32 images directly into your neural network architecture.
