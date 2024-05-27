# British Airways Reviews Analysis

## Project Overview

This task involves web scraping, data cleaning, and analysis of customer reviews for British Airways from the Skytrax website. The goal is to extract review data, clean it, perform sentiment analysis, and visualize the results to gain insights into customer opinions.

## Tools and Libraries Used

- **BeautifulSoup**: For web scraping
- **Requests**: For making HTTP requests
- **Pandas**: For data manipulation and analysis
- **NLTK**: For Natural Language Processing (NLP) tasks including POS tagging and lemmatization
- **VADER**: For sentiment analysis
- **Matplotlib** and **Seaborn**: For data visualization
- **Wordcloud**: For generating word clouds

## Project Steps

### 1. Web Scraping
Using BeautifulSoup and Requests, we scrape review data from Skytrax's British Airways review page.

#### Example Code Snippet:
```python
import requests
from bs4 import BeautifulSoup

url = 'https://www.airlinequality.com/airline-reviews/british-airways'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

reviews = []
for review in soup.find_all('div', class_='review-container'):
    title = review.find('h2').text.strip()
    content = review.find('div', class_='text_content').text.strip()
    reviews.append({'title': title, 'content': content})
```

### 2. Data Cleaning
The scraped data is cleaned to remove unnecessary characters, stop words, and to normalize text.

### 3. POS Tagging
Using NLTK, we perform Part-of-Speech (POS) tagging on the review content to understand the grammatical structure.

#### Example Code Snippet:
```python
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

df['pos_tags'] = df['content'].apply(lambda x: pos_tag(word_tokenize(x)))
```

### 4. Lemmatization
We lemmatize the words in the review content to reduce them to their base or root form.

#### Example Code Snippet:
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_content(content):
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(content)])

df['lemmatized_content'] = df['content'].apply(lemmatize_content)
```

### 5. Sentiment Analysis
Using VADER (Valence Aware Dictionary and sEntiment Reasoner), we perform sentiment analysis on the reviews to determine their polarity (positive, negative, or neutral).

#### Example Code Snippet:
```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

df['sentiment'] = df['content'].apply(lambda x: sid.polarity_scores(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df['sentiment_label'] = df['compound'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))
```

### 6. Data Visualization
We visualize the results using various charts and word clouds to get a better understanding of the sentiment distribution and common words in the reviews.

#### Example Code Snippet:
```python
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Sentiment Distribution
sns.countplot(x='sentiment_label', data=df)
plt.title('Sentiment Distribution')
plt.show()

# Word Cloud
all_words = ' '.join([text for text in df['lemmatized_content']])
wordcloud = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

## Conclusion
This project successfully scrapes, cleans, and analyzes review data for British Airways from Skytrax. Through sentiment analysis and visualization, we gain valuable insights into customer opinions, which can be used for improving service quality and customer satisfaction.

## How to Run the Project
1. **Install Dependencies**:
   ```bash
   pip install requests beautifulsoup4 pandas nltk matplotlib seaborn wordcloud
   ```
2. **Run the Jupyter Notebook**: Open and run the notebook `face_detection_viola_jones.ipynb` in Jupyter.

## License
This project is licensed under the MIT License.