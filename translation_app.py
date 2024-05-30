import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction
from sklearn import pipeline
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import pytesseract
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os
import string
import re
import seaborn as sns

import string

from transformers import MarianMTModel, MarianTokenizer

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def remove_pun(text):
    for pun in string.punctuation:
        text = text.replace(pun, "")
    text = text.lower()
    return text


def language_indentification(txt):
    df = pd.read_csv('Language Detection.csv')
    df['Text'] = df['Text'].apply(remove_pun)

    X = df.iloc[:, 0]
    Y = df.iloc[:, 1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

    vec = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), analyzer='char')

    model_pipe2 = pipeline.Pipeline([('vec', vec), ('clf', SVC())])
    model_pipe2.fit(x_train, y_train)
    y_predict2 = model_pipe2.predict([txt])
    return y_predict2

def extract(img, restore_img):
    text = pytesseract.image_to_string(img)
    language = language_indentification(text)
    return text, language



def model_translate(sentence, target_language):
    model_name = f'Helsinki-NLP/opus-mt-{target_language}-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    # Translate the sentence into English
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    translated_sentence = tokenizer.decode(translated[0], skip_special_tokens=True)
        
    return translated_sentence



def restore_img(img):   

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (560, 900))
    adaptive_result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 5)
    # cv2.imshow("original_IMAGE", img)
    # cv2.imshow("adaptive_result", adaptive_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    text, language = extract(img ,adaptive_result)
    if language != "en":
        sentence = model_translate(text, language)
    
    return adaptive_result,text,language,sentence








def main():
    st.title("Image Extraction App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        enhaced_img, extracted_text, language, sentence = restore_img(image)

        
        st.subheader("Original Image")
        st.image(uploaded_file, caption="Original Image", use_column_width=True)
        st.subheader("Enhanced Image")
        st.image(enhaced_img, caption="Original Image", use_column_width=True)

        st.header("Language OF Text:")
        st.text(language[0])

        st.header("Extracted Text:")
        st.text(extracted_text)
        
        st.header("Translated Text:")
        st.text(sentence)
        

if __name__ == "__main__":
    main()
