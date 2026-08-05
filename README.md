# 🏦 Bank Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

This project is a **Customer Churn Prediction System** developed using an **Artificial Neural Network (ANN)** and deployed with **Streamlit**. The application predicts the probability that a bank customer will leave the bank (churn) based on customer information such as credit score, age, balance, tenure, and account activity.

The model is trained on the **Bank Customer Churn Prediction** dataset and provides real-time predictions through an interactive web interface.

---

# 🚀 Features

## 🤖 Customer Churn Prediction

- Predicts whether a customer is likely to churn.
- Displays the probability of customer churn.
- Real-time prediction using a trained ANN model.

## 📊 Interactive User Interface

Users can enter customer information including:

- Country
- Gender
- Credit Score
- Age
- Balance
- Tenure
- Number of Products
- Credit Card Status
- Active Member Status
- Estimated Salary

## 🧠 Deep Learning Model

- Artificial Neural Network (ANN)
- TensorFlow & Keras implementation
- Binary Classification using Sigmoid Activation

## ⚡ Data Preprocessing

The application automatically performs:

- Label Encoding for Gender
- One-Hot Encoding for Country
- Feature Scaling using StandardScaler

---

# 🏗️ Project Workflow

## Step 1: User Input

The user enters customer details through the Streamlit interface.

↓

## Step 2: Data Preprocessing

The application:

- Encodes Gender using Label Encoder
- One-Hot Encodes Country
- Scales numerical features

↓

## Step 3: ANN Prediction

The processed data is passed to the trained ANN model.

↓

## Step 4: Probability Calculation

The model predicts the churn probability.

↓

## Step 5: Result Display

The application displays:

- Churn Probability
- Likely to Churn
or
- Not Likely to Churn

---

# 🧠 Model Architecture

The Artificial Neural Network consists of:

- Input Layer
- Hidden Layer 1 (64 neurons, ReLU)
- Hidden Layer 2 (32 neurons, ReLU)
- Output Layer (1 neuron, Sigmoid)

Loss Function:

- Binary Crossentropy

Optimizer:

- Adam Optimizer

Evaluation Metric:

- Accuracy

---

# 📊 Input Features

| Feature | Description |
|----------|-------------|
| Credit Score | Customer credit score |
| Country | France, Germany, Spain |
| Gender | Male/Female |
| Age | Customer age |
| Tenure | Number of years with the bank |
| Balance | Bank account balance |
| Number of Products | Products owned by customer |
| Credit Card | Has credit card (0/1) |
| Active Member | Active customer (0/1) |
| Estimated Salary | Annual salary |

---

# 🧰 Technologies Used

## Frontend

- Streamlit

## Backend

- Python

## Machine Learning

- TensorFlow
- Keras
- Scikit-learn

## Data Processing

- Pandas
- NumPy

## Model Serialization

- Pickle

---

# 📂 Project Structure

```bash
Customer-Churn-Prediction/
│
├── app.py
├── model.h5
├── scaler.pkl
├── label_encoder_gender.pkl
├── onehot_encoder_country.pkl
├── requirements.txt
├── Bank Customer Churn Prediction.csv
├── notebooks/
├── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

# 📈 Model Training Pipeline

The model training process includes:

1. Load dataset
2. Remove unnecessary columns
3. Label Encode Gender
4. One-Hot Encode Country
5. Split dataset into training and testing sets
6. Standardize features
7. Train Artificial Neural Network
8. Apply Early Stopping
9. Save trained model and preprocessing objects
10. Deploy using Streamlit

---

# 📊 Prediction Output

The application provides:

- Churn Probability (0–1)
- Customer is Likely to Churn
- Customer is Not Likely to Churn

Example:

```text
Churn Probability: 0.82

The customer is likely to churn.
```

---

# 🔒 Model Files

The project uses the following saved files:

| File | Purpose |
|------|----------|
| model.h5 | Trained ANN model |
| scaler.pkl | Feature scaling |
| label_encoder_gender.pkl | Gender encoding |
| onehot_encoder_country.pkl | Country encoding |

---

# 📈 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Explainable AI using SHAP
- Model deployment on AWS/Azure
- Docker containerization
- REST API using Flask/FastAPI
- Database integration
- Customer retention recommendation system

---

# 🎯 Learning Outcomes

Through this project, I learned:

- Artificial Neural Networks (ANN)
- Binary Classification
- TensorFlow & Keras
- Data Preprocessing
- Feature Engineering
- Label Encoding
- One-Hot Encoding
- Feature Scaling
- Model Serialization using Pickle
- Streamlit Deployment
- Real-time Machine Learning Applications

---

# 👩‍💻 Author

**Rutuja Shivaji Warkhade**

B.Tech Computer Engineering Student

AI/ML & Data Science Enthusiast

---

# 📜 Disclaimer

This project is developed for educational and learning purposes. The predictions generated by the model are based on historical data and should not be considered as guaranteed business decisions. Actual customer behavior may vary depending on additional factors not included in the dataset.

---

⭐ **If you found this project helpful, consider giving it a star on GitHub!**
