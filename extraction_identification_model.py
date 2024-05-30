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
import easyocr


from transformers import MarianMTModel, MarianTokenizer
from translate import Translator



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


def extract_text_from_image(image_path, restore_img):
    # Use EasyOCR to do OCR on the image
    reader = easyocr.Reader(["ru","rs_cyrillic","be","bg","uk","mn","en"]
)
    result = reader.readtext(image_path)
    extracted_text = [entry[1] for entry in result]
    extracted_sentence = ' '.join(extracted_text)

    language = language_indentification(extracted_sentence)
    return extracted_sentence, language

# def google_translate(sentence, target_language):
#     # Initialize the Google Translate client
#     client = translate.Client()

#     # Translate the sentence into English
#     translation = client.translate(sentence, target_language=target_language, source_language='en')

#     # Get the translated text
#     translated_sentence = translation['translatedText']

#     return translated_sentence



def model_translate(sentence):
    model_name = f'Helsinki-NLP/opus-mt-en-fr'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    # Translate the sentence into English
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    translated_sentence = tokenizer.decode(translated[0], skip_special_tokens=True)
        
    return translated_sentence

# def model_translate(text):
#     lang = ['fr','it','es','ru','de']
#     translator = Translator(provider = 'libre', from_lang = 'en', to_lang = 'fr')
#     translate = translator.translate(text)
#     return translate
    # for i in lang:
    #     translator = Translator(provider = 'libre', from_lang = 'en', to_lang = i)
    #     translate = translator.translate(text)
    



def restore_img(img):   

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (560, 900))
    adaptive_result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 5)
    # cv2.imshow("original_IMAGE", img)
    # cv2.imshow("adaptive_result", adaptive_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    text, language = extract_text_from_image(img ,adaptive_result)
    # if language != "en":
    #     sentence = model_translate(text)
    
    return adaptive_result,text,language #,sentence



    
# if __name__ == "__main__":
#     image = cv2.imread("C:/Users/hp/Desktop/jupyter/test images/bound-text-2.jpg")
#     new_img, txt, lang, sentence = restore_img(image)
#     cv2.imshow("original_IMAGE", image)
#     cv2.imshow("adaptive_result", new_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print(txt)
#     print(lang)
#     print(sentence)
#     # restored_img = restore_img("C:/Users/hp/Desktop/jupyter/test images/letter-1.png")
