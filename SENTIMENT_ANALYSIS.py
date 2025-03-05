import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv('task_4/tweets.csv', encoding='ISO-8859-1')

# Data Preprocessing
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    return text

# Apply the clean_text function to the text column
df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x))

# Tokenization
nltk.download('punkt')
nltk.download('punkt_tab')

df['tokenized_text'] = df['cleaned_text'].apply(word_tokenize)

# Removing stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatization
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
df['lemmatized_text'] = df['tokenized_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Join the tokens back into a string
df['processed_text'] = df['lemmatized_text'].apply(lambda x: ' '.join(x))

# Drop rows with NaN values in the sentiment column
df = df.dropna(subset=['sentiment'])

# Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_text']).toarray()
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Implementation
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualization
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.show()

positive_words = ' '.join(df[df['sentiment'] == 'positive']['processed_text'])
negative_words = ' '.join(df[df['sentiment'] == 'negative']['processed_text'])

wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_words)
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Positive Words')
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Negative Words')
plt.show()
