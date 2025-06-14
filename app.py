import os
import logging
import tensorflow as tf

# Suppress TensorFlow GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from functools import lru_cache
from tensorflow.keras.applications.efficientnet import preprocess_input
import time
import markdown2

# Load environment variables
load_dotenv('config.env')

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Safety settings for Gemini
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

try:
    # Initialize Gemini API with supported model name
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        safety_settings=safety_settings
    )
    print("Successfully initialized Gemini model")
except Exception as e:
    print(f"Error initializing Gemini: {str(e)}")
    raise

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'  # Use /tmp for Vercel
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size: 16MB
app.secret_key = os.urandom(24)

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Model
try:
    logger.info("Attempting to load the model...")
    model_path = 'plant_disease_efficientnetb0'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    # Load model with custom objects to handle optimizer configuration
    model = tf.keras.models.load_model(model_path, compile=False)
    logger.info("Model loaded successfully")

    # Ensure model expects (None, 224, 224, 3) input and outputs (None, 15)
    logger.info(f"Model Input Shape: {model.input_shape}")
    logger.info(f"Model Output Shape: {model.output_shape}")

    # Compile the model with the correct optimizer configuration
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    logger.info("Model compiled successfully")

except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    print("="*80)
    print("ERROR: Could not load the model file.")
    print("Please ensure you have:")
    print(f"1. The model file '{model_path}' in the project directory")
    print("2. The correct model architecture matching the class names")
    print("="*80)
    raise

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ✅ ONLY 15 CLASSES — MATCHES TRAINING ORDER
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Cache for disease info
@lru_cache(maxsize=100)
def get_cached_disease_info(disease_name):
    return get_disease_info(disease_name)

def get_disease_info(disease_name):
    try:
        # Handle both formats: 'Tomato_Early_blight' and 'Potato___Early_blight'
        if 'healthy' in disease_name.lower():
            # Split based on either '___' or '_'
            separator = '___' if '___' in disease_name else '_'
            plant_type = disease_name.split(separator)[0].replace('_', ' ')
            prompt = f"""As a plant expert, provide detailed information about healthy {plant_type} plants.
            Use these sections:
            1. Normal Appearance and Features
            2. Optimal Growing Conditions
            3. Best Practices for Maintenance
            4. Signs of Good Health
            5. Recommended Care Routine
            
            For each section, provide 3-5 specific bullet points. Focus only on {plant_type} plants."""
        else:
            # Try splitting by '___' first
            if '___' in disease_name:
                parts = disease_name.split('___')
                plant_type = parts[0].replace('_', ' ')
                disease = parts[1].replace('_', ' ')
            else:
                # Fallback: split by single '_' (for Tomato_Bacterial_spot)
                parts = disease_name.split('_')
                plant_type = parts[0]
                disease = ' '.join(parts[1:])  # Take rest as disease name

            prompt = f"""As a plant disease expert, provide detailed information about {disease} in {plant_type} plants.
            Use these sections:
            1. Causes and Triggers
            2. Symptoms and Visual Indicators
            3. Prevention Methods
            4. Treatment Options
            5. Best Practices for Cultivation
            
            For each section, provide 3-5 specific bullet points. Focus specifically on {disease} in {plant_type} plants."""
        
        logger.info(f"Prompt sent to Gemini for '{disease_name}': {prompt}")
        
        response = gemini_model.generate_content(
            contents=prompt,
            generation_config={
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 1024,
            }
        )
        
        logger.info(f"Raw Gemini API response for '{disease_name}': {response.text}")

        if response and response.text:
            generated_text = response.text.strip()
            if generated_text:
                return generated_text

        logger.warning(f"No valid response from Gemini for {disease_name}, using fallback info")
        return get_fallback_info(disease_name)

    except Exception as e:
        logger.error(f"Error getting disease info for {disease_name}: {str(e)}")
        return get_fallback_info(disease_name)

def get_fallback_info(disease_name):
    """Fallback content when Gemini fails"""
    if 'healthy' in disease_name:
        plant_type = disease_name.split('___')[0]
        return f"{plant_type} appears healthy. No disease detected."
    return "Gemini API unavailable. Showing placeholder info."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_stream):
    """Preprocess uploaded image using correct pipeline for EfficientNetB0"""
    try:
        img = Image.open(image_stream).convert('RGB')  # Always use RGB
        img = img.resize((224, 224))  # Resize to match training
        img_array = np.array(img)  # Convert to NumPy array

       
        img_array = preprocess_input(img_array)  # This line ensures compatibility
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        return img_array

    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise

@app.route('/')
def home():
    logger.info("Home page accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        img_array = preprocess_image(file.stream)

        # Make prediction
        predictions = model.predict(img_array)
        logger.info(f"Raw prediction vector: {predictions[0]}")  # Log full prediction vector

        top_index = np.argmax(predictions[0])
        confidence = float(predictions[0][top_index])

        logger.info(f"Top class index: {top_index}")
        logger.info(f"Predicted Disease: {CLASS_NAMES[top_index]}")
        logger.info(f"Confidence: {confidence:.4f}")

        # Only include predictions above 10% confidence
        top_predictions = [{
            'disease': CLASS_NAMES[top_index],
            'confidence': confidence
        }]

        # Store top prediction in session
        disease_name = CLASS_NAMES[top_index]
        disease_info = get_cached_disease_info(disease_name)
        session['disease_info'] = disease_info
        session['disease_name'] = disease_name

        return jsonify({
            'predictions': top_predictions,
            'disease_info': disease_info
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/disease-details/<disease_name>')
def disease_details(disease_name):
    disease_info = session.get('disease_info')
    if not disease_info:
        disease_info = get_cached_disease_info(disease_name)
        session['disease_info'] = disease_info
    # Convert markdown to HTML
    disease_info_html = markdown2.markdown(disease_info)
    return render_template('disease_details.html',
                           disease_name=disease_name,
                           disease_info=disease_info_html)

@app.route('/test-gemini')
def test_gemini():
    try:
        test_prompt = "What are 3 main causes of Late Blight in Potatoes?"
        response = gemini_model.generate_content(test_prompt)
        logger.info(f"Gemini test response: {response.text}")
        return jsonify({"status": "success", "response": response.text})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# For Vercel deployment
app = app

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True)