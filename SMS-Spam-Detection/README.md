# 📧 SMS Spam Detection

This project focuses on detecting **spam SMS messages** using **Natural Language Processing (NLP)** and **Machine Learning** techniques. The system classifies messages as **spam** or **ham** (not spam) using text preprocessing, TF-IDF vectorization, and Naive Bayes models.

---

## 📝 Features

- Data cleaning and preprocessing of raw SMS messages  
- Exploratory Data Analysis (EDA) for insights on spam vs ham messages  
- Text preprocessing including:
  - Lowercasing
  - Tokenization
  - Removing stopwords and punctuation
  - Stemming  
- Feature extraction using **TF-IDF Vectorization**  
- Model training using Naive Bayes variants:
  - GaussianNB
  - MultinomialNB
  - BernoulliNB  
- Model evaluation with **accuracy, precision, and confusion matrix**  

---

## 📁 Dataset

- File: `data/spam.csv`  
- Columns:
  - `v1` → label (`ham` or `spam`)  
  - `v2` → SMS message text  
- Source: [SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  

---

## 📊 Exploratory Data Analysis (EDA)

- Distribution of spam vs ham messages  
- Analysis of message length (characters, words, sentences)  
- Visualizations using Matplotlib and Seaborn  

---

## 🛠 Technologies Used

- Python 3.x  
- Pandas, NumPy  
- NLTK for text preprocessing  
- Scikit-learn for TF-IDF vectorization and Naive Bayes models  
- Matplotlib, Seaborn for visualization  

---

## 🚀 How to Run

1. **Clone the repository**  
```bash
git clone https://github.com/yourusername/sms-spam-detection.git
cd sms-spam-detection
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the script

bash
Copy code
python scripts/spam_detection.py
Output

Model performance metrics (accuracy, precision, confusion matrix)

Optionally, save trained model in models/spam_model.pkl

📌 Insights
Ham messages are generally longer than spam messages

Certain words and patterns appear more frequently in spam messages

Text preprocessing and TF-IDF significantly improve model performance
