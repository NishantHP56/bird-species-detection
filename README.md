# 🦜 Bird Species Detection with ResNet18

This project uses deep learning (PyTorch) to identify bird species from images.  
It trains a ResNet18 CNN model on a Kaggle dataset containing over **220 bird species**, and provides an interactive **Gradio web interface** for real-time predictions.

---

## 🚀 Features
- Trains a **ResNet18** model on bird image data
- Automatically downloads dataset via KaggleHub
- Achieves high validation accuracy with data augmentation
- Saves and reloads model checkpoints
- Includes a **Gradio web UI** for easy image upload and species recognition

---

## 🧩 Tech Stack
- **Python 3.9+**
- **PyTorch** (model training)
- **Torchvision** (datasets & transforms)
- **Gradio** (web interface)
- **KaggleHub** (automatic dataset download)
- **TQDM** (progress visualization)

---

## 📁 Project Structure
bird-species-detection/
│
├── src/
│   ├── train_birds.py         # training + validation loop (from your code)
│   ├── utils.py               # helper functions (dataset, transforms, labels)
│   └── app.py                 # Gradio app (inference)
│
├── models/
│   ├── resnet18_bird_prototype.pth
│
├── data/
│   ├── Train/
│   └── Test/
│
├── images/                    # demo example images
├── requirements.txt
├── README.md
└── .gitignore


---

## 🧠 Model Overview
- **Architecture:** ResNet18  
- **Loss:** CrossEntropyLoss  
- **Optimizer:** Adam (lr = 0.001)  
- **Input size:** 128×128  
- **Epochs:** 27  
- **Dataset:** [Bird Species Classification (220 categories)](https://www.kaggle.com/datasets/kedarsai/bird-species-classification-220-categories)

---

## ⚙️ How to Run

### 1️⃣ Clone the Repository
```bash


Install Dependencies
pip install -r requirements.txt

Train the Model
python train_birds.py

Run the Gradio App
python app.py

