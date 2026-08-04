# MindCare AI

An AI-powered mental health classification and recommendation system built with
Scikit-Learn, PyTorch, and Flask.

## Project Structure

```
MEDAI/
├── backend/
│   ├── __init__.py              # Package initialiser
│   ├── config.py                # Central configuration singleton
│   ├── logger.py                # Centralised logging setup
│   ├── utils.py                 # Shared utility helpers
│   ├── data_loader.py           # CSV loading and validation
│   ├── preprocessing.py         # Data cleaning and scaling
│   ├── feature_engineering.py   # Feature transformation pipeline
│   ├── trainer.py               # Scikit-Learn multi-model training
│   ├── pytorch_model.py         # PyTorch neural network architecture
│   ├── pytorch_trainer.py       # PyTorch training loop
│   ├── evaluator.py             # Evaluation metrics and plots
│   ├── evaluation.py            # Public re-export alias
│   ├── model_comparator.py      # Model comparison and selection
│   ├── model_saver.py           # Artifact serialisation
│   ├── recommendation_engine.py # Mental health recommendations
│   └── predictor.py             # End-to-end inference pipeline
├── dataset/
│   └── data.csv                 # Training dataset (user-provided)
├── models/                      # Saved model artifacts (auto-created)
├── logs/                        # Application logs (auto-created)
├── reports/                     # Evaluation reports (auto-created)
├── plots/                       # Generated plots (auto-created)
├── train.py                     # Training entry point
├── app.py                       # Flask API server
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## Requirements

- Python 3.12+
- pip

## Installation

```bash
git clone <repository-url>
cd MEDAI
pip install -r requirements.txt
```

## Dataset

Place your mental health CSV dataset at:

```
dataset/data.csv
```

The dataset should contain feature columns and a target column (default: `mental_health_status`).

## Training

```bash
python train.py
```

### Options

```bash
python train.py --target-column mental_health_status
python train.py --no-pytorch
python train.py --dataset-path path/to/custom.csv
```

### What happens during training

1. Dataset is loaded and validated.
2. Data is split into Train / Validation / Test sets (stratified).
3. Feature engineering pipeline is fitted on the training set.
4. Six Scikit-Learn models are trained with 5-fold cross-validation.
5. (Optional) A PyTorch deep learning model is trained with mixed precision.
6. All models are evaluated with advanced metrics and plots.
7. The best model is selected via multi-tier comparison.
8. All artifacts are saved to `models/`.

## Inference API

Start the Flask server:

```bash
python app.py
```

The server runs on `http://localhost:5000`.

### Endpoints

| Method | Path           | Description                                 |
|--------|----------------|---------------------------------------------|
| GET    | /health        | Health check                                |
| GET    | /api/info      | API metadata                                |
| GET    | /api/classes   | List supported class labels                 |
| POST   | /api/predict   | Predict mental health status                |

### Example prediction request

```bash
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"age": 25, "gender": "Male", "sleep_hours": 6, ...}'
```

### Example response

```json
{
  "success": true,
  "timestamp": "2024-01-15T12:30:00Z",
  "result": {
    "predicted_class": "anxiety",
    "confidence": 0.87,
    "probabilities": {"anxiety": 0.87, "depression": 0.08, "normal": 0.05},
    "recommendation": {
      "risk_level": "moderate",
      "immediate_actions": ["Practice slow, diaphragmatic breathing..."],
      "self_care": ["Limit caffeine...", "Practice mindfulness..."],
      "resources": ["iCall Helpline: 9152987821"]
    }
  }
}
```

## Models Trained

| Model                   | Type        |
|-------------------------|-------------|
| Logistic Regression     | Scikit-Learn|
| Decision Tree           | Scikit-Learn|
| Random Forest           | Scikit-Learn|
| K-Nearest Neighbors     | Scikit-Learn|
| Support Vector Machine  | Scikit-Learn|
| Gaussian Naive Bayes    | Scikit-Learn|
| Deep Neural Network     | PyTorch     |

## License

This project is for educational and research purposes.

## Disclaimer

This system is for informational purposes only and does not constitute medical advice.
Always consult a licensed mental health professional for diagnosis and treatment.
