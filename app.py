import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
import os
import json
from torchvision import models
import torch.nn as nn

# --- 1. Model and Label Setup ---
# This script is a self-contained example that will create a dummy model
# and dummy example images to ensure the Gradio interface launches.

# Define the number of output classes and class labels
NUM_CLASSES = 200  # use the real number of classes
# Assuming 'class_names' is defined in a previous cell and contains the list of class names
CLASS_LABELS = class_names

# Load a pre-trained ResNet18 model
model = models.resnet18() # Changed from vgg16 to resnet18

# Replace the classifier layer
num_ftrs = model.fc.in_features # Changed from classifier[6].in_features to fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES) # Changed from classifier[6] to fc

# Load trained weights
model_save_path = "resnet18_bird_prototype.pth" # Changed to the correct model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded successfully from '{model_save_path}'.")
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_save_path}'. Please ensure the model has been trained and saved.")
    # Exit or handle the error appropriately if the model file is not found
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    # Exit or handle the error appropriately if model loading fails
    exit()


# Define the image transformations.
preprocess = transforms.Compose([
    transforms.Resize((128, 128)), # Changed from 224x224 to 128x128
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. Prediction Function ---
def classify_bird(image: Image.Image):
    """
    Classifies a bird image using the loaded PyTorch model.
    """
    if image is None:
        return {}

    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    confidences = {CLASS_LABELS[i]: float(probabilities[i]) for i in range(len(CLASS_LABELS))}
    sorted_confidences = sorted(confidences.items(), key=lambda item: item[1], reverse=True)
    return dict(sorted_confidences)

# --- 3. Gradio Interface Setup ---
# Create a dummy 'images' directory and placeholder image files for Gradio examples
images_dir = 'images'
os.makedirs(images_dir, exist_ok=True)
dummy_image_paths = []
for filename in ["blue_jay.jpg", "bald_eagle.jpg", "pigeon.jpg"]:
    path = os.path.join(images_dir, filename)
    # Create a dummy image with the correct size
    Image.new('RGB', (128, 128), color = 'white').save(path) # Changed size
    dummy_image_paths.append(path)
print(f"Dummy example images created in '{images_dir}' directory.")

# Define the input and output components for the UI.
image_input = gr.Image(type="pil", label="Upload an image of a bird")
label_output = gr.Label(num_top_classes=3, label="Prediction")

# Create the Gradio interface with dynamic examples
gr.Interface(
    fn=classify_bird,
    inputs=image_input,
    outputs=label_output,
    title="Bird Species Recognizer",
    description="Upload an image to recognize the bird species. This is a demo; a real model file is required for accurate predictions.",
    examples=dummy_image_paths
).launch()
