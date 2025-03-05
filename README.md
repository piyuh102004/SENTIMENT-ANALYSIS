# Sentiment Analysis of Tweets

*COMPANY*:  CODTECH IT SOLUTIONS

*NAME*:  PIYUSH SINGH

*INTERN ID*:  CT08TUM

*DOMAIN*:  DATA ANALYTICS

*DURATION*:  4 WEEKS

*MENTOR*:  NEELA SANTOSH


![Image](https://github.com/user-attachments/assets/1defe561-da09-4ea5-8808-6486f980aad1)
![Image](https://github.com/user-attachments/assets/f8509e3e-3a26-48c5-93d6-5047c1568f2e)
![Image](https://github.com/user-attachments/assets/07918f72-c7f2-4f20-91c9-364ff6db865e)



## About the Program
This program performs sentiment analysis on a dataset of tweets. It classifies tweets into three categories: positive, negative, and neutral. The analysis includes data preprocessing, feature extraction using TF-IDF, and model training using Logistic Regression.

## Requirements
To run this program, you need to have the following Python libraries installed:
- pandas
- numpy
- nltk
- scikit-learn
- matplotlib
- seaborn
- wordcloud

You can install these libraries using pip:
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud
```

## Data Collection
The dataset used in this program is stored in a CSV file named `tweets.csv`. The file contains two columns:
- `text`: The content of the tweet.
- `sentiment`: The sentiment label (positive, negative, or neutral).

## Instructions to Run the Program
1. Ensure you have Python installed on your machine.
2. Install the required libraries as mentioned above.
3. Place the `tweets.csv` file in the same directory as the `SENTIMENT_ANALYSIS.py` script.
4. Run the program using the following command:
   ```bash
   python task_4/SENTIMENT_ANALYSIS.py
   ```
5. The program will output the accuracy of the model and a classification report. It will also display visualizations for sentiment distribution and word clouds for positive and negative words.

## Code Explanation
- **Data Preprocessing**: The `clean_text` function cleans the tweet text by lowercasing, removing URLs, punctuation, and unwanted characters. The text is then tokenized, stop words are removed, and lemmatization is applied.
- **Feature Extraction**: The processed text is converted into numerical features using `TfidfVectorizer`.
- **Model Implementation**: A Logistic Regression model is trained on the training data, and predictions are made on the test set. The model's performance is evaluated with accuracy and a classification report.
- **Visualization**: The program generates visualizations for sentiment distribution and word clouds for positive and negative sentiments.

## Visualization
The program generates the following visualizations:
- A count plot showing the distribution of sentiments in the dataset.
- Word clouds for positive and negative sentiments, illustrating the most frequently used words in each category.

Feel free to modify the program and explore different aspects of sentiment analysis!
