import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request
import csv
import io
from flask import Flask, render_template, request, make_response, redirect, url_for
# ... (keep your torch and PIL imports here)

app = Flask(__name__)

# --- 1. CONFIGURATION & DATABASE ---
# Ensure these match the alphabetical order of your training folders exactly
MINERALS = [
    'aba_panu_aba_panu', 
    'abukumalit', 
    'actinolite-asbestos', 
    'adamin', 
    'aegirine_augite', 
    'afghanite', 
    'aftitalit', 
    'afwillite', 
    'agalmatolite', 
    'agardite-ce', 
    'smoky_quartz'
]

# Load the mineral database (ensure minerals_db.json is in the same folder)
DATABASE_PATH = 'minerals_db.json'
if os.path.exists(DATABASE_PATH):
    with open(DATABASE_PATH, 'r') as f:
        MINERAL_DATA = json.load(f)
else:
    MINERAL_DATA = {}
    print(f"⚠️ Warning: {DATABASE_PATH} not found!")

# --- 2. MODEL LOADING ---
def get_model(model_filename):
    # Initialize the ResNet18 architecture
    model = models.resnet18()
    # Modify the final layer to match your 11 mineral classes
    model.fc = nn.Linear(model.fc.in_features, len(MINERALS))
    
    try:
        # Load your trained weights (.pth file)
        model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
        model.eval()
        print(f"✅ Model '{model_filename}' loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"❌ Error: {model_filename} not found in project directory.")
        return None

# Load the model once when the app starts
MODEL_FILE = 'mineral_resnet18_v2.pth'
model = get_model(MODEL_FILE)

# --- 3. IMAGE PREPROCESSING ---
# These transforms ensure the uploaded image matches the training data format
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. FLASK ROUTES ---

@app.route('/')
def home():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and model prediction."""
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    if not model:
        return render_template('index.html', error="Model not loaded. Check terminal.")

    try:
        # 1. Open and transform the image
        img = Image.open(file).convert('RGB')
        img_t = transform(img)
        
        # Add batch dimension
        batch_t = torch.unsqueeze(img_t, 0)

        # --- SAFETY TWEAK ---
        model.eval() # Explicitly tell model NOT to use training memory
        # --------------------

        # 3. Perform the Prediction
        with torch.no_grad():
            output = model(batch_t)
            
            # Apply Softmax to get probabilities
            probabilities = torch.softmax(output, dim=1) # Using torch.softmax directly is cleaner
            conf, index = torch.max(probabilities, 1)
            
            # Get the predicted mineral key and confidence score
            predicted_key = MINERALS[index.item()]
            confidence_val = round(conf.item() * 100, 2)

        # --- MEMORY CLEANUP ---
        del img, img_t, batch_t, output # Manually clear the heavy image data from RAM
        # ----------------------

        # 4. Fetch the detailed data from the JSON database
        info = MINERAL_DATA.get(predicted_key, {
            "title": predicted_key.replace('_', ' ').title(),
            "structure": "Data Unavailable",
            "hardness": "N/A",
            "gravity": "N/A",
            "refractive_index": "N/A",
            "molar_mass": "N/A"
        })

        return render_template('index.html', result=info, confidence=confidence_val)
    
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return render_template('index.html', error="An error occurred while processing the image.")

# --- ROUTE: BROWSE DATABASE ---
@app.route('/database')
def browse_database():
    """Displays all 11 minerals in a searchable table format."""
    # We pass the entire JSON database to the new template
    return render_template('database.html', minerals=MINERAL_DATA)

# --- ROUTE: EXPORT REPORT ---
@app.route('/export/<mineral_key>')
def export_report(mineral_key):
    """Generates a downloadable CSV report for a specific mineral."""
    info = MINERAL_DATA.get(mineral_key)
    
    if not info:
        return "Mineral data not found", 404

    # 1. Create a string buffer to hold CSV data
    si = io.StringIO()
    cw = csv.writer(si)
    
    # 2. Write the rows (Header followed by Data)
    cw.writerow(['PROPERTY', 'VALUE'])
    cw.writerow(['Mineral Name', info.get('title')])
    cw.writerow(['Crystal Structure', info.get('structure')])
    cw.writerow(['Mohs Hardness', info.get('hardness')])
    cw.writerow(['Specific Gravity', info.get('gravity')])
    cw.writerow(['Refractive Index', info.get('refractive_index')])
    cw.writerow(['Molar Mass', info.get('molar_mass')])
    
    # 3. Create the response object
    output = make_response(si.getvalue())
    
    # 4. Set headers so the browser treats it as a download
    file_name = f"{mineral_key}_report.csv"
    output.headers["Content-Disposition"] = f"attachment; filename={file_name}"
    output.headers["Content-Type"] = "text/csv"
    
    return output

# --- ROUTE: ANALYTICS (SIDEBAR PLACEHOLDER) ---
@app.route('/analytics')
def analytics():
    """A placeholder route for the analytics sidebar button."""
    return "Analytics Page - Under Construction"

# --- 5. START THE APP ---
if __name__ == '__main__':
    # debug=True allows the app to refresh automatically when you save changes
    app.run(debug=True, port=5000)