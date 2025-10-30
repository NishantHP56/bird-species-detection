import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn as nn

# Load class names (update with your dataset class list)
path = "path_where_kagglehub_downloaded"  # optional if needed
train_dir = os.path.join(path, 'Train')
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
NUM_CLASSES = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("resnet18_bird_prototype.pth", map_location=device))
model.to(device)
model.eval()

# preprocess
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_bird(image):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {class_names[i]: float(probabilities[i]) for i in range(NUM_CLASSES)}
    return dict(sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:3])

gr.Interface(
    fn=classify_bird,
    inputs=gr.Image(type="pil", label="Upload Bird Image"),
    outputs=gr.Label(num_top_classes=3, label="Top Predictions"),
    title="Bird Species Detector",
    description="Upload a bird image to identify its species using a ResNet18 model trained on 220 bird categories."
).launch()
