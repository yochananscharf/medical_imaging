import streamlit as st
import torch
from streamlit import session_state as ss
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel
import joblib
from time import sleep
st.title('X-Ray classifier')
sidebar = st.sidebar



def predict_image(image):
    inputs = ss.feature_extractor(images=image_arr, return_tensors="pt")
    with torch.no_grad():
        outputs = ss.model_embeddings(**inputs)
    embeddings = outputs.last_hidden_state[:, 0].squeeze().numpy()
    label = ss.model_svm.predict(embeddings.reshape(1, -1))
    return label


    
if 'count' not in ss:

    ss.count = 0
    #load embeddings model
    model_name = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
    ss.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    ss.model_embeddings = model
    ss.model_svm = joblib.load('svm_model_final.joblib')
    st.rerun()
    #img_file = None
else:    
    img_file = sidebar.file_uploader("Upload an image (sometimes you need to upload twice!)", type=["png", "jpg", "jpeg"], key = ss.count)
    if img_file is not None:
        
          

        image_arr = Image.open(img_file).convert("RGB")
        st.image(image_arr)
        label = predict_image(image_arr)
        # inputs = ss.feature_extractor(images=image_arr, return_tensors="pt")
        # with torch.no_grad():
        #     outputs = ss.model_embeddings(**inputs)
        # embeddings = outputs.last_hidden_state[:, 0].squeeze().numpy()
        # label = ss.model_svm.predict(embeddings.reshape(1, -1))
        sidebar.write('The label is {}'.format(label+1), )
        img_file = None

        ss.count += 1
        sleep(5)
        st.rerun()