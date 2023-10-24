import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from streamlit import Streamlit, Multiselect, Slider, checkbox, text_input, number_input
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS_SentimentIntensityAnalyzer
#import re
import streamlit as st

# Load pre-trained model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize VADER sentiment analyzer
vader_analyzer = VS_SentimentIntensityAnalyzer()

# Load NLTK data
nltk.download('vader_lexicon')

# Define a function to compute sentiment polarity and subjectivity
def compute_sentiment(text):
    sentiment = {'polarity': 0, 'subjectivity': 0}
    
    # Tokenize and compute sentiment polarity using the pre-trained model
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment['polarity'] = probabilities.tolist()[0][1]
    
    # Compute sentiment subjectivity using VADER
    sentiment['subjectivity'] = vader_analyzer.polarity_scores(text)['compound']
    
    return sentiment

# Define a function to display sentiment results
def display_sentiment(sentiment):
    polarity = round(sentiment['polarity'] * 100, 2)
    subjectivity = round(sentiment['subjectivity'] * 100, 2)
    return f"Polarity: {polarity}% | Subjectivity: {subjectivity}%"

# Create a Streamlit app
st.title("Fine-Tuned Transformer for Sentiment Analysis")
text = st.text_input("Enter your text here:")

if text:
    sentiment = compute_sentiment(text)
    st.write(display_sentiment(sentiment))