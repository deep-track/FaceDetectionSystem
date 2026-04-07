import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import transforms
from scripts.train_image_model import load_trained_model

app = FastAPI(title="DeepTrack Prediction API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set to CPU for standard web hosting, or CUDA if deployed on a GPU server
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "data/best_swin.pth" # place the .pth file in the same folder

# load the model once when the API starts
try:
    model = load_trained_model(MODEL_PATH, DEVICE)
except Exception as e:
    print(f"Warning: Model not found at {MODEL_PATH}. API will fail until weights are added.")
    model = None

# Exact transforms used during Kaggle evaluation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# pred endpoint
@app.post("/v1/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model weights not loaded on server.")
        
    try:
        # Read the image uploaded by the user
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess and add batch dimension (Shape becomes [1, 3, 224, 224])
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            # Note: The model returns a tuple (output, features), so we unpack it
            output, _ = model(input_tensor)
            
            # Convert raw logits to percentages
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, prediction_idx = torch.max(probabilities, 0)
            
        # Map index to class names (Ensure this matches your Kaggle 0/1 mapping!)
        class_names = ["Real", "Fake"] 
        predicted_class = class_names[prediction_idx.item()]
        
        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence_percentage": round(confidence.item() * 100, 2),
            "raw_scores": {
                "Real": round(probabilities[0].item() * 100, 2),
                "Fake": round(probabilities[1].item() * 100, 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    
# small test ui

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DeepTrack Test UI</title>
        <style>
            body {
                font-family: Arial;
                background: #f4f6f8;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .box {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                width: 400px;
                text-align: center;
            }
            button {
                padding: 10px;
                width: 100%;
                margin-top: 10px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background: #0056b3;
            }
            img {
                max-width: 100%;
                margin-top: 15px;
                border-radius: 5px;
            }
            .result {
                margin-top: 15px;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <div class="box">
            <h2>DeepTrack Image Prediction</h2>

            <input type="file" id="imageInput" accept="image/*">
            <button onclick="uploadImage()">Predict</button>

            <img id="preview" />
            <div class="result" id="result"></div>
        </div>

        <script>
            async function uploadImage() {
                const fileInput = document.getElementById("imageInput");
                const resultDiv = document.getElementById("result");
                const preview = document.getElementById("preview");

                resultDiv.innerHTML = "";

                if (!fileInput.files.length) {
                    alert("Please select an image first.");
                    return;
                }

                const file = fileInput.files[0];
                preview.src = URL.createObjectURL(file);

                const formData = new FormData();
                formData.append("file", file);

                resultDiv.innerHTML = "Processing...";

                try {
                    const response = await fetch("/v1/predict", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        resultDiv.innerHTML = "Error: " + data.detail;
                        return;
                    }

                    resultDiv.innerHTML = `
                        <strong>Prediction:</strong> ${data.prediction}<br>
                        <strong>Confidence:</strong> ${data.confidence_percentage}%<br><br>
                        <strong>Real:</strong> ${data.raw_scores.Real}%<br>
                        <strong>Fake:</strong> ${data.raw_scores.Fake}%
                    `;

                } catch (err) {
                    resultDiv.innerHTML = "Error connecting to API.";
                }
            }
        </script>
    </body>
    </html>
    """