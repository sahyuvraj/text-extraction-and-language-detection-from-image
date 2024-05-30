import extraction_identification_model
import nagamese_translationdb
import streamlit as st
import cv2
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from translate import Translator



def main():
    st.title("Image Extraction App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        # enhaced_img, extracted_text, language, sentence = extraction_identification_model.restore_img(image)
        enhaced_img, extracted_text, language = extraction_identification_model.restore_img(image)
        without_punc_sentence = nagamese_translationdb.remove_pun(extracted_text)
        sentence = nagamese_translationdb.naga_translation(without_punc_sentence)

        
        st.subheader("Original Image")
        st.image(uploaded_file, caption="Original Image", use_column_width=True)
        

        st.header("Language OF Text:")
        st.text(language[0])

        st.header("Extracted Text:")
        st.text(extracted_text)
        
        st.header("Translated Text Into Nagamese:")
        st.text(sentence)
        

if __name__ == "__main__":
    main()
