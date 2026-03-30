import torch
import torch.nn as nn
import torchvision.models as models

class CustomCNN(nn.Module):
    """
    Tier 1 Baseline Model: Custom Lightweight CNN
    Designed for fast iteration and clear gradient analysis during adversarial training.
    Architecture: 4 Conv Blocks -> Flatten -> Dense -> Dropout -> Dense
    """
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Block 1: 1 -> 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # 256 -> 128
        )
        
        # Block 2: 32 -> 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # 128 -> 64
        )
        
        # Block 3: 64 -> 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # 64 -> 32
        )
        
        # Block 4: 128 -> 256 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2) # 32 -> 16
        )
        
        # Classifier
        # Flattened size: 256 channels * 16 * 16 spatial dimensions = 65,536
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Prevent overfitting on texture
            nn.Linear(512, 1) # Output 1 logit for Binary Classification
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

def get_resnet18_grayscale():
    """
    Tier 2 Final Evaluator: Modified ResNet-18
    Modifies the standard ResNet18 input layer to accept 1-channel grayscale malware images
    instead of the default 3-channel RGB.
    """
    # Load ResNet18 without pre-trained weights since malware textures !== ImageNet
    model = models.resnet18(weights=None)
    
    # 1. Modify the first convolutional layer
    # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1, # Change 3 to 1
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias
    )
    
    # 2. Modify the fully connected layer for binary classification output
    # Add Dropout before the final dense layer for regularization
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 1)
    )
    
    return model

def get_efficientnet_b0_grayscale():
    """
    EfficientNet-B0 adapted for single-channel grayscale malware images.
    
    EfficientNet-B0 was designed via Neural Architecture Search for optimal
    accuracy/parameter trade-off. With ~5.3M parameters vs ResNet-18's 11.2M,
    it provides a controlled architectural comparison:
    - Different design philosophy (compound scaling vs residual blocks)
    - Half the parameter count
    - Depthwise separable convolutions instead of standard conv layers
    
    If both architectures show similar logit separation and evasion rates,
    this confirms the source bias hypothesis is dataset-level, not model-level.
    
    Grayscale adaptation: average the 3 RGB input channel weights into 1.
    """
    import torchvision.models as tv_models
    
    # Load without pretrained weights — same as ResNet-18 for fair comparison
    model = tv_models.efficientnet_b0(weights=None)
    
    # Adapt first conv layer from 3-channel to 1-channel
    # EfficientNet-B0 first layer: features[0][0] is a Conv2d(3, 32, ...)
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )
    
    # Replace the classifier head for binary output
    # EfficientNet-B0 classifier: Sequential(Dropout, Linear(1280, 1000))
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1)
    )
    
    return model


class MaleX3C2D(nn.Module):
    """
    3-Convolution 2-Dense (3C2D) model - Mohammed et al. (2021), MaleX paper.

    Architecture as described in paper:
        Conv(32, 3x3) -> MaxPool(2)
        Conv(64, 3x3) -> MaxPool(2)
        Conv(128, 3x3) -> MaxPool(2)
        Dense(512, Dropout=0.5)
        Dense(256, Dropout=0.5)
        Output (sigmoid - here we use a single logit for BCEWithLogitsLoss)

    BUG FIX vs previous version:
        Original error: After 3x MaxPool(2) on 256x256 input, spatial size is 32x32.
        Flattening 128 x 32 x 32 = 131,072 features into FC(131072->512) created
        67.3M parameters in a single layer alone.

        Fix: AdaptiveAvgPool2d((4, 4)) reduces spatial to 4x4 regardless of input size.
        128 x 4 x 4 = 2,048 features -> FC(2048->512) -> ~1M params, total ~2-3M.

    Input:  [B, 1, 256, 256] grayscale tensor (normalized to [-1, 1])
    Output: [B, 1] single logit for BCEWithLogitsLoss
    """

    def __init__(self):
        super(MaleX3C2D, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1: 1 -> 32 channels, 256 -> 128
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 2: 32 -> 64 channels, 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 3: 64 -> 128 channels, 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Global spatial reduction: 32x32 -> 4x4 (input-size independent)
            # Without this, FC(131072->512) = 67M params (the previous bug).
            # With this,    FC(2048->512)   = ~1M params (correct).
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # 128 channels x 4 x 4 = 2,048 input features to classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),  # single logit for BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_resnet18_pretrained_grayscale():
    """
    ResNet-18 fine-tuned from ImageNet pretrained weights for grayscale malware images.
    Following Mohammed et al. (2021): load pretrained weights, adapt first conv layer
    for 1-channel input, replace final FC for binary classification.

    Key difference from get_resnet18_grayscale(): uses pretrained initialization.
    The first conv layer weights are averaged across the 3 RGB channels to initialize
    the single grayscale channel, preserving the pretrained feature structure.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Adapt first conv layer: average the 3-channel weights into 1 channel.
    # This preserves the learned low-level feature structure instead of
    # reinitializing from scratch.
    old_conv1 = model.conv1
    new_conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=False,
    )
    # Initialize new 1-channel weights by averaging the 3 RGB channel weights.
    # Shape: [64, 3, 7, 7] -> mean over dim=1 -> [64, 1, 7, 7]
    new_conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)
    model.conv1 = new_conv1

    # Replace the classifier head for binary output.
    # Higher dropout (0.5) for stronger regularization during fine-tuning.
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 1),
    )

    return model


if __name__ == "__main__":
    # Test instantiations and passing dummy tensors
    print("Testing Model Instantiations...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Dummy image batch: [Batch_Size, Channels, Height, Width]
    dummy_input = torch.randn(4, 1, 256, 256).to(device)
    
    # 1. Test Custom CNN
    custom_cnn = CustomCNN().to(device)
    out_custom = custom_cnn(dummy_input)
    print(f"CustomCNN initialized. Output shape: {out_custom.shape} (Expected: 4, 1)")
    
    # 2. Test ResNet18
    resnet18_gray = get_resnet18_grayscale().to(device)
    out_resnet = resnet18_gray(dummy_input)
    print(f"ResNet18 initialized. Output shape: {out_resnet.shape} (Expected: 4, 1)")
    
    print("\n✅ Both models successfully configured for 256x256 grayscale inputs.")
