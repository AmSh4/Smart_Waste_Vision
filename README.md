# SmartWasteVision

**AI-powered Waste Sorting & Recycling Optimization System**

**Short description**: AI-driven system to classify waste (plastic, organic, metal) using computer vision and provide actionable recycling insights. Includes dataset generation, training pipeline, API, Streamlit dashboard, and a small DB for logs.

## Why this project?
- Real-world environmental impact: automated sorting helps recycling centers and smart bins.
- End-to-end: dataset → training → model → API → dashboard → DB logs.
- Search-friendly: includes keywords like waste sorting, recycling, computer vision, YOLO alternative, CNN, IoT-ready.

## Project Structure
```
SmartWasteVision/
|__ data/sample_images/        # small synthetic dataset (generated)
|__ src/
     |__ api/                     # FastAPI app to serve predictions
     |__ model/                   # training script and model helper
     |__ utils/                   # utility scripts
     |__ db/                      # sqlite database
|__ dashboard/                 # Streamlit demo app
|__ notebooks/                 # optional analysis notebooks
|__ scripts/                   # dataset generation
|__ README.md
|__ requirements.txt
```

## Quickstart (runs locally)
1. Create virtual env and install requirements:
   ```bash
   python -m venv venv && source venv/bin/activate   
   ```
2. Generate dataset:
   ```bash
   python scripts/generate_dataset.py
   ```
3. Train the tiny CNN (demo):
   ```bash
   python src/model/train.py
   ```
   This saves `src/model/waste_cnn.pth`.
4. Initialize DB:
   ```bash
   python src/db/init_db.py
   ```
5. Run API:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```
6. Run dashboard:
   ```bash
   streamlit run dashboard/streamlit_app.py
   ```

## Files of interest
- `scripts/generate_dataset.py` — creates synthetic 64x64 images for 3 classes
- `src/model/train.py` — training loop using PyTorch; lightweight for demo
- `src/model/model.py` — model loader & inference helper
- `src/api/main.py` — FastAPI prediction endpoint `/predict`
- `dashboard/streamlit_app.py` — interactive demo
- `src/db/init_db.py` — initializes SQLite DB for logging predictions

## Notes 
- Replace the tiny CNN with a state-of-the-art detector (YOLOv8/Detectron2) and add transfer learning for production-grade accuracy.
- Connect to cloud storage and use real bin camera streams, add MQTT for IoT integration.
- MLOps ideas: model registry, CI for training, scheduled retraining on new labeled data.

