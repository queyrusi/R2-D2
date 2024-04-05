import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import platform
import warnings
import os
import time

cwd = os.getcwd()

# Check if MPS or GPU is available
has_gpu = torch.cuda.is_available()
has_mps = getattr(torch,'has_mps',False)

device = "mps" if getattr(torch,'has_mps',False) \
    else "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")

# Define the transformation
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((299, 299)),  # Resize to match Inception v3 input size
    torchvision.transforms.ToTensor(),  # Convert image to tensor
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Load the pre-trained Inception v3 model
# Mute the warning
warnings.filterwarnings("ignore", category=UserWarning)

model = torchvision.models.inception_v3(pretrained=True)

# Replace the last layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 1),
    torch.nn.Sigmoid()
)

model.aux_logits = False  # Ensure that auxiliary logits are not used for loss calculation

# Check if the saved model file exists
model_file = './model/inception_v3.pth'
if os.path.exists(model_file):
    # Load the saved model state
    model.load_state_dict(torch.load(model_file))
else:
    print("Saved model file does not exist.")
model.eval()

# Move the model to the target device
model = model.to(device)

# Load the D1_images dataset
dataset = torchvision.datasets.ImageFolder('./images/D2_images', transform=transform)
# Run inference on the D1_images dataset
times_file = open('times/d2_inference_times.txt', 'w')
for i, (inputs, labels) in enumerate(dataset):
    inputs = inputs.unsqueeze(0).to(device)
    start_time = time.time()
    outputs = model(inputs)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000
    times_file.write(f"Inference time for image {i}: {inference_time} milliseconds\n")
    predicted = torch.round(outputs)
    print(f'Image {i}: Class {predicted.item()}')
times_file.close()
