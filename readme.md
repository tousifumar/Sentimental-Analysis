# 🧠 SentimentSense: A Sentiment Analysis Web Application

## 📌 Project Overview

**SentimentSense** is a user-friendly web application that performs sentiment analysis on text input like product reviews or comments. It uses **Natural Language Processing (NLP)** to classify text into **Positive**, **Negative**, or **Neutral** categories.

> Built using **Python**, **Flask**, **TextBlob**, **Jinja2**, **HTML**, **CSS**, and **JSON**.



## 🚀 Key Features

- 🔍 Real-time sentiment analysis
- 🧠 NLP using TextBlob
- 🌐 Flask-powered backend
- 🎨 Simple front-end using HTML, CSS & Jinja2
- 📊 Outputs polarity and sentiment category



## 🧰 Tech Stack

| Component  | Technology                       |
|------------|----------------------------------|
| Frontend   | HTML, CSS, Jinja2                |
| Backend    | Python (Flask)                   |
| NLP Model  | TextBlob                         |
| Data Format| JSON                             |



## 📂 Project Structure

SentimentSense/
│
├── static/
│   └── style.css             # CSS styles
│
├── templates/
│   └── index.html            # Main frontend UI
│
├── app.py                    # Main Flask app
├── requirements.txt          # Dependencies list
└── README.md                 # Project documentation

# 💻 Installation Guide

---

## ✅ Prerequisites
* Python 3.8 or higher
* pip (Python package manager)

---

## 🔧 Setup Steps

### Clone the Repository

git clone [https://github.com/yourusername/SentimentSense.git](https://github.com/yourusername/SentimentSense.git)
cd SentimentSense 
## Create and Activate Virtual Environment (if not existed)
* python -m venv venv
* source venv/bin/activate    # On Windows: venv\Scripts\activate

## Install Required Packages
* pip install -r requirements.txt

## Download TextBlob Corpora
* python -m textblob.download_corpora

## Run the Application
* python app.py

## Open Your Browser
* Visit: http://127.0.0.1:5000 

## 📝 Example Input/Output
# Input:
* I absolutely love this product! It's fantastic.

# Output:
* Sentiment: Positive  Polarity: 0.75

