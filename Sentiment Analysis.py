#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Function to fetch headlines from Yahoo Finance
def fetch_headlines():
    url = "https://finance.yahoo.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = soup.find_all('h3')
    return [headline.get_text() for headline in headlines if headline.get_text()]

# Function for sentiment analysis using NLTK's Vader
def analyze_sentiment(headlines):
    sid = SentimentIntensityAnalyzer()
    sentiments = []
    for headline in headlines:
        sentiment_score = sid.polarity_scores(headline)
        sentiments.append((headline, sentiment_score))
    return sentiments

# Main function
def main():
    headlines = fetch_headlines()
    sentiments = analyze_sentiment(headlines)
    for headline, sentiment in sentiments:
        print(headline)
        print("Sentiment Score:", sentiment)
        print()

if __name__ == "__main__":
    main()

