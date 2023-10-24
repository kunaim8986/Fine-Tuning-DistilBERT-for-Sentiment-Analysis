# Project: Development of Fine-Tuned Transformer for Sentiment Analysis
# Inventors: Mohamad Idham Razak, Mohammed Hariri Bakri, Ku Muhammad Na'im Ku Khalif,
#            Juzlinda Mohd Ghazali, Noor Hafhizah Abd Karim, Abdul karim Mohamad
# Program: Jejak Inovasi UTeM 2023



import streamlit as st 
import pandas as pd 
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import sys
import os
import torch
from torch import nn
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, AdamW
import pandas as pd
st.set_page_config(page_title='Fine-Tuning-DistilBERT-for-Sentiment-Analysis',layout='wide')

def get_filesize(file):
    size_bytes = sys.getsizeof(file)
    size_mb = size_bytes / (1024**2)
    return size_mb

def validate_file(file):
    filename = file.name
    name, ext = os.path.splitext(filename)
    if ext in ('.csv','.xlsx'):
        return ext
    else:
        return False
    
    
# Create sidebar to load dataset
with st.sidebar:
    from PIL import Image
    image = Image.open('magnum opus.jpg')
    st.image(image, caption='')
    #image = Image.open('UTEM-1024x1024.jpg')
    #st.image(image, caption='')
    #image = Image.open('logo-umpsa.png')
    #st.image(image, caption='')
    st.header("Inventors")
    st.write("Mohamad Idham Md Razak")
    st.write("Mohammed Hariri Bakri")
    st.write("Ku Muhammad Na'im Ku Khalif")
    st.write("Juzlinda Mohd Ghazali")
    st.write("Noor Hafhizah Abd Rahim")
    st.write("Abdul Karim Mohamad")
    
    st.header("Data Profiling Exploration")
    uploaded_file = st.file_uploader("Upload .csv, .xlsx files which is not exceed with 200MB.") 
    if uploaded_file is not None:
        st.write('Modes of Operation')
        minimal = st.checkbox('Do you want minimal report ?')
        display_mode = st.radio('Display mode:',
                                options=('Primary','Dark','Orange'))
        if display_mode == 'Dark':
            dark_mode= True
            orange_mode = False
        elif display_mode == 'Orange':
            dark_mode = False
            orange_mode = True
        else:
            dark_mode = False
            orange_mode = False
    
if uploaded_file is not None:
    ext = validate_file(uploaded_file)
    if ext:
        filesize = get_filesize(uploaded_file)
        if filesize <= 10: #standard file for non cloud platform
            if ext == '.csv':
                # for this project, consider load csv
                df = pd.read_csv(uploaded_file)
            else:
                xl_file = pd.ExcelFile(uploaded_file)
                sheet_tuple = tuple(xl_file.sheet_names)
                sheet_name = st.sidebar.selectbox('Select the sheet',sheet_tuple)
                df = xl_file.parse(sheet_name)
                
            
# To generate details report for MDF dataset
            with st.spinner('Generating Report'):
                pr = ProfileReport(df,
                                minimal=minimal,
                                dark_mode=dark_mode,
                                orange_mode=orange_mode
                                )
                
            st_profile_report(pr)
        else:
            st.error(f'Maximum allowed filesize is 10 MB. But received {filesize} MB')
            
    else:
        st.error('Please only submit files that meet the following criteria: .csv or .xlsx file')
        
else:
    st.title('Development of Fine Tuned – Transformer for Sentiment Analysis')
    
    st.header("Overview of Fine Tuned – Transformer for Sentiment Analysis")
    st.image("Sentiment Analyser (Transformer-DistilBERT).jpg")
    st.text('This ground-breaking research develops a fine-tuned – Transformer for Sentiment Analysis (FT-TSA). Modern text data sentiment analysis is complicated,') 
    st.text('but this work takes a novel technique. The FT-TSA model is carefully built to capture sentiment nuances and adapt to changing language and environment.')
    st.text('It achieves state-of-the-art sentiment classification using comprehensive data preparation, tokenization, and parameter tuning. FT-TSA outperforms ')
    st.text('traditional sentiment analysis models and generic Transformers in complex or context-dependent attitudes, according to empirical assessments on ') 
    st.text('standard sentiment analysis datasets. We consider the implementation to pre-train a smaller general-purpose language representation model, DistilBERT,')
    st.text('which can be fine-tuned to perform well on a wide range of tasks like its larger equivalents. This research shows FT-TSAs adaptability by applying it,') 
    st.text('to sentiment-aware recommendation systems and social media sentiment tracking. In conclusion, this study introduces a novel and highly adaptable ') 
    st.text('Transformer-based model that advances sentiment analysis and addresses the challenges of evolving sentiment expressions in modern text data.,')
    
    st.header("Data Profiling Exploration")
    st.info('Upload your data in the left sidebar to generate the related results')
    
        
    
#building the sidebar of the web app which will help us navigate through the different sections of the entire application

with st.sidebar:
    st.header("Sentiment Analysis")
    
rad=st.sidebar.radio("Transformer", ["Home","Fine-Tune Transformer"])
#rad=st.sidebar.radio("Risk Prediction Application",["Home","Diabetes Disease","Cardiovascular Disease"])
#Home Page 

if rad=="Home":
    st.header("Fine-Tuned Transformer Sentiment Analysis")
    st.image("transformer.png")
    st.text('In 2017, Vaswani et al. introduced a transformer deep learning model in “Attention is All You Need” for machine learning and NLP. The model was proposed to  ')
    st.text('improve translation systems. The name “transformer” comes from its ability to transform one sequence (input text) into another (output text) while ')
    st.text('incorporating the context at multiple levels. It was a groundbreaking model because it introduced ‘attention’, which allows the model to focus on relevant ')
    st.text('input sequence parts when producing output. Transformer models have an encoder to read text and a decoder to predict the task. Transformers are amazing at ')
    st.text('maintaining accuracy for long input sequences. Transformers can process all text input simultaneously, unlike RNNs and CNNs, which can be time-consuming. ')
    st.text('Transformers are great for NLP because they can detect long-distance dependencies in text data due to concurrent processing.')
    

import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

st.title("Fine-Tuned Transformer for Sentiment Analysis")

@st.cache_data
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    nlp_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    return nlp_model

nlp_model = load_model()

user_input = st.text_input("Enter your text here:")

if user_input:
    result = nlp_model(user_input)
    st.write(f"Sentiment: {result[0]['label']}")
    st.write(f"Confidence: {round(result[0]['score'], 4)}")