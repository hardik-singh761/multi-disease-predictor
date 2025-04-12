# 🧠 AI-Based Multi-Disease Predictor

A Flask web application that uses trained Machine Learning models to predict **Diabetes**, **Heart Disease**, and **Parkinson's Disease** based on user input. The goal of this project is to provide a simple and beautiful UI to demonstrate how AI can assist in early diagnosis.

![Screenshot (33)](https://github.com/user-attachments/assets/fbcc8f68-1ded-4b1e-8b0e-d0809ff60dba)


---

## 💻 Features

- 🔍 Predicts 3 diseases:
  - Diabetes
  - Heart Disease
  - Parkinson's Disease
- 📦 Integrated with trained ML models using `pickle`
- 🎨 Modern, clean, sidebar-based UI (with green & white theme)
- 📋 Forms with dropdowns for better user experience
- 🧠 Backend powered by Flask
- 🔁 Keeps form data after submission
- 📊 Accurate predictions based on real medical datasets

---

## 🧠 Machine Learning Models

Each disease has its own ML model trained using real datasets:

- **Diabetes**: Trained on the PIMA Indian Diabetes dataset.
- **Heart Disease**: Trained on the Cleveland heart disease dataset.
- **Parkinson's Disease**: Trained on voice measurement data.

All models are saved as `.sav` files using `pickle.
