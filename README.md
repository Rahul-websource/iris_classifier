# Hello-ML: Iris Flower Classifier

A beginner-friendly machine learning project that classifies Iris flowers into three species using Python and Logistic Regression.

## Overview

This project uses the built-in Iris dataset from scikit-learn to predict the species of a flower based on four measurements.

### Flower Species
- Iris setosa
- Iris versicolor
- Iris virginica

### Features
- Loads and preprocesses the Iris dataset
- Splits data into training and testing sets
- Trains a Logistic Regression classifier
- Evaluates the model using accuracy and a classification report
- Saves the trained model using joblib
- Predicts the species of a new flower

## Technologies
- Python
- scikit-learn
- joblib

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Hello-ML.git
cd Hello-ML
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python iris_classifier.py
```

The program displays the model's evaluation metrics and a sample flower prediction.

## Dataset

The project uses the Iris dataset included with scikit-learn. It contains 150 samples, four numerical features, and three flower species.

## Model

The classifier uses Logistic Regression with feature standardization. The dataset is split into training and testing sets using an 80/20 ratio.

## Future Improvements
- Add a Streamlit web interface
- Visualize the dataset
- Compare multiple classification algorithms
- Add user input for flower measurements

## Author

Your Name

## License

This project is available for educational use.
