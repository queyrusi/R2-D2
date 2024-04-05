#!/usr/bin/env python
"""
This script generates heatmaps using GradCAM for a given set of images.
"""

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import os
import platform
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import warnings
import cv2
import argparse
from tqdm import tqdm
warnings.filterwarnings("ignore")

def get_device():
    # Check if MPS or GPU is available
    has_gpu = torch.cuda.is_available()
    has_mps = getattr(torch, 'has_mps', False)

    device = "mps" if getattr(torch, 'has_mps', False) \
        else "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Python Platform: {platform.platform()}")
    print(f"PyTorch Version: {torch.__version__}")
    print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
    print("MPS is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
    print(f"Target device is {device}")
    return device

device = get_device()

def load_model():
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
    if not os.path.exists(model_file):
        model_file = '../model/inception_v3.pth'
    if os.path.exists(model_file):
        # Load the saved model state
        model.load_state_dict(torch.load(model_file))
    else:
        raise FileNotFoundError("Saved model file does not exist.")
    model.eval()

    return model


def preprocess_image(image_path):
    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    preprocessed_image = preprocess(image)
    input_tensor = preprocessed_image.unsqueeze(0)

    return input_tensor


def generate_heatmap(model, input_tensor, image_path):
    target_layers = [layer for layer in [model.Mixed_5b, model.Mixed_5c, model.Mixed_5d, model.Mixed_6a, model.Mixed_6b, model.Mixed_6c,
                    model.Mixed_6d, model.Mixed_6e, model.Mixed_7a, model.Mixed_7b, model.Mixed_7c, model.avgpool]]

    cam = AblationCAM(model=model, target_layers=target_layers)

    # targets = [ClassifierOutputTarget(1)]

    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]

    # Get name of file
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    # Save the heatmap as a numpy array 
    # np.save(f'XAI/heatmaps/{name}.npy', grayscale_cam)

    grayscale_cam_tensor = torch.tensor(grayscale_cam)
    gsc_cam_scaled = (grayscale_cam_tensor - grayscale_cam_tensor.min()) / (grayscale_cam_tensor.max() - grayscale_cam_tensor.min()) * 255

    # Convert grayscale_cam to an image and save it as a PNG
    heatmap_image = Image.fromarray(np.uint8(gsc_cam_scaled), mode='L')
    dataset = os.path.basename(os.path.dirname(image_dir)).split('_')[0]
    if not os.path.exists(f'heatmaps/{dataset}'):
        os.makedirs(f'heatmaps/{dataset}')
    heatmap_image.save(f'heatmaps/{dataset}/{name}.png')

    heatmap = cv2.applyColorMap(np.uint8(gsc_cam_scaled), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = Image.fromarray(heatmap).convert('RGBA')

    return overlay


def blend_images(image_path, overlay):
    image_tensor = transforms.ToTensor()(image_path)
    resized_image = torchvision.transforms.functional.resize(image_tensor, (224, 224))
    plot_compatible_image = np.transpose(resized_image, (1, 2, 0))
    blend_compatible_PIL_image = image_path.resize((224, 224)).convert('RGBA')

    overlayed_image = Image.blend(blend_compatible_PIL_image, overlay, alpha=0.5)

    return plot_compatible_image, overlayed_image


def visualize_results(image, overlayed_image):
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title('R2-D2 Byte image')
    axs[1].imshow(overlayed_image)
    axs[1].axis('off')
    axs[1].set_title('AblationCAM overlay')
    plt.subplots_adjust(wspace=0.15)
    plt.savefig('output.pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    model = load_model()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, help="Path to the directory containing the images")
    args = parser.parse_args()
    image_dir = args.image_dir
    for image_file in tqdm(os.listdir(image_dir), desc="Generating heatmaps", unit='image'):
        if image_file.endswith(".png"):
            image_path = os.path.join(image_dir, image_file)
            input_tensor = preprocess_image(image_path)
            overlay = generate_heatmap(model, input_tensor, image_path)
