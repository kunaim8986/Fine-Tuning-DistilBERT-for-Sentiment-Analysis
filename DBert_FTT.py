# Project: Development of Fine-Tuned Transformer for Sentiment Analysis
# Inventors: Mohamad Idham Razak, Mohammed Hariri Bakri, Ku Muhammad Na'im Ku Khalif,
#            Juzlinda Mohd Ghazali, Noor Hafhizah Abd Karim, Abdul karim Mohamad
# Program: Jejak Inovasi UTeM 2023


import streamlit as st
import torch
from torch import nn
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, AdamW
import pandas as pd


# Create sidebar to load dataset
with st.sidebar:
    from PIL import Image
    image = Image.open('magnum opus.jpg')
    st.image(image, caption='')
    image = Image.open('UTEM-1024x1024.jpg')
    st.image(image, caption='')
    image = Image.open('logo-umpsa.png')
    st.image(image, caption='')
    st.header("Inventors")
    st.write("Mohamad Idham Md Razak")
    st.write("Mohammed Hariri Bakri")
    st.write("Ku Muhammad Na'im Ku Khalif")
    st.write("Juzlinda Mohd Ghazali")
    st.write("Noor Hafhizah Abd Rahim")
    st.write("Abdul Karim Mohamad")

def main():
    st.title("Fine-Tuning DistilBERT for Sentiment Analysis")

    st.header("Data Loading")
    data = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    if data is not None:
        df = pd.read_csv(data)
        st.write(df.head())

    st.header("Model Loading")
    model = st.selectbox("Choose a model", ("DistilBert",))

    if model == "DistilBert":
        st.write("You have selected the DistilBert model.")
        st.write("The current pre-trained model is: `distilbert-base-uncased`")

    st.header("Tokenization")
    sentences = df['text'].tolist()
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    inputs = tokenizer(sentences, return_tensors='pt', truncation=True, padding=True)

    st.header("Model Training")
    train_epochs = st.slider("Number of training epochs", 1, 10, 3)
    learning_rate = st.slider("Learning rate", 1e-5, 1e-2, 1e-3)

    if st.button("Start Training"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss().to(device)

        model.train()

        for epoch in range(train_epochs):
            for input_ids, attention_mask, labels in zip(inputs['input_ids'], inputs['attention_mask'], inputs['text']):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optimizer.step()

            st.write(f"Epoch {epoch + 1}/{train_epochs} - Loss: {loss.item()}")

    st.header("Model Evaluation")
    st.write("Once the model is trained, you can evaluate its performance using various metrics such as accuracy, precision, recall, and F1-score.")

if __name__ == "__main__":
    main()  