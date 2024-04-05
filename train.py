"""
This script trains an Inception v3 model to classify images as either 'goodware' or
'malware'.

It first checks if MPS or GPU is available and sets the target device accordingly. 
It then defines the transformations to apply to the images, which include resizing 
the images to 299x299 (as required by Inception v3) and converting them to tensors.

The script is expected to continue with loading the datasets, creating a DataLoader,
loading the pre-trained Inception v3 model, replacing the last layer to match the
number of classes, moving the model to the target device, defining the loss 
function and optimizer, and then training the model.

Usage:
    python train.py

Requirements:
    - PyTorch
    - torchvision
    - platform

"""
import os 

import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
import platform
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
import warnings

# Mute the warning
warnings.filterwarnings("ignore", category=UserWarning)

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

# Load the dataset
dataset = torchvision.datasets.ImageFolder('images/D0_images', transform=transform)

# Load the test dataset
test_dataset = torchvision.datasets.ImageFolder('images/D0_images', transform=transform)

# Split the dataset into training and evaluation sets
train_size = int(0.95 * len(dataset))
eval_size = len(dataset) - train_size

train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

# Create DataLoaders for training and evaluation
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False, drop_last=True)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

# Load the pre-trained Inception v3 model
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

# Move the model to the target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Initialize TensorBoard writer
writer = SummaryWriter()

try:
    for epoch in range(15):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_dataloader, 0), desc="passing through train"):
            if i==0:
                break
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())  # Squeeze the output to remove extra dimension

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

        if False:
            # Evaluate the model on the evaluation set
            eval_loss = 0.0
            correct = 0
            total = 0
            TP, FP, TN, FN = 0, 0, 0, 0
            with torch.no_grad():
                for data in tqdm(eval_dataloader, desc="running through eval"):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels.float())
                    eval_loss += loss.item()
                    predicted = torch.round(outputs).cpu().T  # Convert probabilities to binary predictions
                    total += labels.size(0)
                    correct += (predicted == labels.cpu()).sum().item()
                    
                    # Calculate TP, FP, TN, FN
                    TP += ((predicted == 1) & (labels == 1)).sum().item()
                    FP += ((predicted == 1) & (labels == 0)).sum().item()
                    TN += ((predicted == 0) & (labels == 0)).sum().item()
                    FN += ((predicted == 0) & (labels == 1)).sum().item()
            
            # Write TP, FP, TN, FN and current epoch to train.log
            with open('eval.log', 'a') as f:
                f.write(f'Epoch: {epoch + 1}, TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}\n')

            print('Epoch %d: Evaluation loss: %.3f, Accuracy: %.2f %%' %
                (epoch + 1, eval_loss / len(eval_dataloader), 100 * correct / total))
            writer.add_scalar('Evaluation Loss', eval_loss / len(eval_dataloader), epoch)

        # Evaluate the model on the test set
        eval_loss = 0.0
        correct = 0
        total = 0
        TP, FP, TN, FN = 0, 0, 0, 0
        with torch.no_grad():
            for i_batch, data in enumerate(tqdm(test_dataloader, desc="running through test")):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                eval_loss += loss.item()
                predicted = torch.round(outputs).cpu().T  # Convert probabilities to binary predictions
                total += labels.size(0)
                correct += (predicted == labels.cpu()).sum().item()
                
                # Calculate TP, FP, TN, FN
                TP += ((predicted == 1) & (labels == 1)).sum().item()
                FP += ((predicted == 1) & (labels == 0)).sum().item()
                TN += ((predicted == 0) & (labels == 0)).sum().item()
                FN += ((predicted == 0) & (labels == 1)).sum().item()
                
                # Write TP, FP, TN, FN and current epoch to test.log every ten batches
                if (i_batch + 1) % 10 == 0:
                    with open('logs/test_D0.log', 'a') as f:
                        f.write(f'Epoch: {epoch + 1}, Batch: {i_batch + 1}, TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}\n')
        
        print('Epoch %d: Evaluation loss: %.3f, Accuracy: %.2f %%' %
              (epoch + 1, eval_loss / len(eval_dataloader), 100 * correct / total))
        writer.add_scalar('Evaluation Loss', eval_loss / len(eval_dataloader), epoch)

        # Create the ./model directory if it doesn't exist
        if not os.path.exists('./model'):
            os.makedirs('./model')

        # Save the model
        torch.save(model.state_dict(), './model/inception_v3.pth')

    print('Finished Training')

except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    # Save the model
    torch.save(model.state_dict(), './model/inception_v3.pth')
