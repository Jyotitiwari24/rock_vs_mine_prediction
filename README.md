# Sonar Object Classifier

This project classifies objects as **Rocks (R)** or **Mines (M)** based on sonar signals using Logistic Regression.

## Project Structure

sonar-classifier/
├── data/
├── notebooks/
├── src/
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md


## Installation

```bash
git clone <repo-url>
cd sonar-classifier
pip install -r requirements.txt
Usage
Train the model
python src/model.py
Predict new data
python src/predict.py
Docker Usage
docker build -t sonar-classifier .
docker run --rm sonar-classifier
Dataset
The sonar dataset is available in data/sonar.csv. Each row represents sonar returns; the last column is the label (R for Rock, M for Mine).