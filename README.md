# ANN Classification App

A machine learning application that predicts customer churn using an Artificial Neural Network (ANN).

## 🎯 Overview

This app uses a trained ANN model to predict whether a customer will leave a bank based on various features like age, credit score, balance, and more. It provides a user-friendly interface built with Streamlit for easy predictions.

## 🚀 Live Demo

Try the app here: [https://ann-classification-app.streamlit.app/](https://ann-classification-app.streamlit.app/)

## 📋 Features

- Input customer details through an interactive web interface
- Real-time predictions on customer churn probability
- Pre-trained ANN model for quick inference
- Automatic data preprocessing and scaling

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/AmeyaPadwad/ann_classification-app.git
cd ann_classification-app
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📚 Course Reference

This project is built following the [Complete Generative AI Course with LangChain and HuggingFace](https://www.udemy.com/course/complete-generative-ai-course-with-langchain-and-huggingface/) on Udemy.

## 🔧 Technologies Used

- **TensorFlow/Keras** - Neural network model
- **Scikit-learn** - Data preprocessing and scaling
- **Streamlit** - Web application framework
- **Pandas & NumPy** - Data manipulation
- **Pickle** - Model serialization
