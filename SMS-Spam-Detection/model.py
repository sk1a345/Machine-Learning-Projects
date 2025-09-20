import pandas as pd

df = pd.read_csv(
    r"C:\Users\HP\OneDrive\Machine_Learning_projects\SMS_SPAM_DETECTOR\spam.csv",
    encoding="ISO-8859-1"
)
# print(df.head(10))
# print(df['v1'].value_counts())
# print(df.shape)

# 1 Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model Building
# 5. Evaluation
# 6. Imporovement
# 7. Website
# 8. Deploy:

# Data Cleaning:

# print(df.info())
# Dropping the last 3 columns:
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True);
# print(df.sample(5))

# Renaming the columns:
df.rename(columns={'v1':'target','v2':'text'},inplace=True)

# applying the label encoder:
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# print(df['target'].unique())
# 0-ham  1: spam

# Checking the missing values:
# print(df.isnull().sum())

# Checking for the duplicate values:
# print(df.duplicated().sum())
# Dropping the duplicates:
df.drop_duplicates(keep = 'first',inplace=True)
# print(df.duplicated().sum())

# EDA exploratory data analysis:
import matplotlib.pyplot as plt
import seaborn as sns

# plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
# plt.show()
# ham: 87% and spam = 12%

import nltk #natural language toolkit

# nltk.download('punkt')

# no. of alphabets
df['num_characters'] = df['text'].apply(len)

# no. of words:
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))

# no. of sentences:
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# print(df[['num_characters','num_words','num_sentences']].describe()
# )

# print(df[df['target']==0][['num_characters','num_words','num_sentences']].describe()
# )
#
# print(df[df['target']==1][['num_characters','num_words','num_sentences']].describe()
# )

import seaborn as sns
plt.figure(figsize=(12,8))
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')
# plt.show()

sns.pairplot(df,hue='target')
# plt.show()

# Data preprocessing:
# lowercase
# tokenization
# Removing special characters:
# Removing stop words and punctuation:
# stemming:


# lowercase:

from nltk.corpus import stopwords
# print(stopwords.words("English"))

import string
# print(string.punctuation)

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("English") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)
# print(df['transform_text'])

spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for words in msg.split():
        spam_corpus.append(words)

# print(len(spam_corpus))

ham_corpus = []
for msg in df[df['target']==0]['transformed_text'].tolist():
    for w in msg.split():
        ham_corpus.append(w)

# print(len(ham_corpus))


# Model Building:
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()

x = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# train test split:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(x_train,y_train)
y_pred1 = gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

# Best model prediction:
mnb.fit(x_train,y_train)
y_pred2 = mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

bnb.fit(x_train,y_train)
y_pred3 = bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
