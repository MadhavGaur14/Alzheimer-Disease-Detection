from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os

# Initialize the app
app = Flask(__name__)

# Load ResNet50 model (you can change to ResNet101 if needed)
class ResNetModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Load ResNet50 with pretrained weights
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Adjust final layer for 4 classes

    def forward(self, x):
        return self.resnet(x)

model = ResNetModel()  # Create an instance of your ResNet model

# Load model weights (ensure the path is correct)
model.load_state_dict(torch.load("best_resnet_model.pth", map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define a transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to the input size of your model
    transforms.ToTensor(),            # Convert the image to a tensor
])

# Class names mapping
class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

@app.route('/')
def home():
    return render_template('x.htm')  # Serve the HTML file

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file temporarily
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Open the image file
    image = Image.open(filepath).convert('RGB')  # Convert to RGB for ResNet
    image = transform(image)  # Apply the transforms
    image = image.unsqueeze(0)  # Add a batch dimension

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    # Remove the uploaded file after processing
    os.remove(filepath)

    # Return the prediction result as class name
    prediction_class = class_names[predicted.item()]  # Map index to class name
    return jsonify({"prediction": prediction_class})

if __name__ == '__main__':
    # Create an uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    app.run(port=5001, debug=True)  # Run on port 5001
