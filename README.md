# AI Phishing Detection Platform

A **web-based AI platform** to detect phishing in messages and URLs using machine learning. This project is built with **Python Flask**, allowing users to train models and predict whether a given text or URL is phishing or legitimate.

---

## Features

- **Train Message Model:** Train an AI model on messages dataset (text + label: phishing or legitimate).  
- **Train URL Model:** Train an AI model on URLs dataset (url + label: phishing or legitimate).  
- **Predict Message:** Enter any text (like emails, messages) to check if it is phishing.  
- **Predict URL:** Enter any URL to check if it is phishing.  
- **Confidence Scores:** Displays confidence level for legitimate and phishing predictions.  
- **Interactive Web Interface:** Single-page Flask application for easy usage.

---

## Project Structure

phishing_detection_platform/
├── app.py # Main Flask application
├── train_model.py # Script to train models
├── messages.csv # Sample dataset for messages
├── urls.csv # Sample dataset for URLs
├── msg_model.pkl # Saved model for messages
├── url_model.pkl # Saved model for URLs
├── msg_vectorizer.pkl # Vectorizer for message model
├── url_vectorizer.pkl # Vectorizer for URL model
├── templates/
│ └── index.html # Frontend HTML page
├── uploads/ # Folder to upload CSVs for training
└── README.md

---

## Installation
 **Clone the repository**
git clone https://github.com/Aradhanasingh00/AI-Phishing-Detection-Platform.git
cd AI-Phishing-Detection-Platform
##Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # On Linux/Mac
venv\Scripts\activate          # On Windows

##Install dependencies
pip install -r requirements.txt
## Run the flask app
python3 app.py
## Usage
#Phase 1 — Train Models

Train Message Model: Upload messages.csv and click Train Message Model.

Train URL Model: Upload urls.csv and click Train URL Model.

#Phase 2 — Predict

Check Text: Enter any message/email and click Check Text to see if it’s phishing.

Check URL: Enter any URL and click Check URL to verify its legitimacy.
# Example Output
Legitimate (Text)
Confidence → legitimate: 50.0% • phishing: 50.0%
# Technologies Used 
Python 3

Flask (Web Framework)

Scikit-learn (Machine Learning)

Pandas, Numpy (Data Handling)

HTML/CSS/Bootstrap (Frontend)

# 🚀 Run Locally (Linux / macOS)

Follow these simple steps to set up and run the **AI Phishing Detection Platform**:

```bash
#  Clone the repository
git clone https://github.com/Aradhanasingh00/AI-Phishing-Detection-Platform.git

# Move into the project directory
cd AI-Phishing-Detection-Platform

#  Create a virtual environment
python -m venv venv

#  Activate the virtual environment
source venv/bin/activate

#  Install the required dependencies
pip install -r requirements.txt

#  Run the application
python3 app.py


# Created by
Aradhana Singh 

GitHub: Aradhanasingh00

Email: singharadhana2004@gmail.com
