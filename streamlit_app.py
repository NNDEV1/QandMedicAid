import argparse
import torch
from torch.utils.data import DataLoader
import streamlit as st
import dataset_RAD
import base_model
import utils
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from bunch import Bunch
import numpy as np
import torchvision.transforms.functional as TF
import os
import sys
from PIL import Image
import time

st.set_page_config(page_title='Q&MedicAid',
                   layout='wide')

# MICCAI19-MedVQA

data_RAD = './data_RAD'
model_path = './BAN_MEVF'
batch_size = 1
constructor = 'build_BAN'
rnn = 'LSTM'

torch.backends.cudnn.benchmark = True
device = torch.device("cpu")

dictionary = dataset_RAD.Dictionary.load_from_file(
    os.path.join(data_RAD, 'dictionary.pkl'))

args = Bunch()
args.RAD_dir = data_RAD
args.autoencoder = True
args.maml = False
args.autoencoder = True
args.feat_dim = 64
args.op = 'c'
args.num_hid = 1024
args.rnn = rnn
args.gamma = 2
args.ae_model_path = 'pretrained_ae.pth'
args.maml_model_path = 'pretrained_maml.weights'
args.activation = 'relu'
args.dropout = 0.5
args.maml = True
args.eps_cnn = 1e-5
args.momentum_cnn = 0.05
args.map_location = "cpu"


def main(image_file, question):

    eval_dset = dataset_RAD.VQAFeatureDataset('test', args, dictionary)
    model = base_model.build_BAN(eval_dset, args)

    @st.cache(allow_output_mutation=True)
    def load_image_tensors(image_path):
        xray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        xray_ae = cv2.resize(xray, (128, 128))
        xray_maml = cv2.resize(xray, (84, 84))
        xray_ae_t = TF.to_tensor(np.array(xray_ae))
        xray_maml_t = TF.to_tensor(np.array(xray_maml))

        xray_maml_t = xray_maml_t.unsqueeze(1).to(device)
        xray_ae_t = xray_ae_t.unsqueeze(1).to(device)
        return [xray_maml_t, xray_ae_t]

    image_tensors = load_image_tensors(
        image_file)

    def tokenize(text, dataset, max_length=12):
        tokens = dataset.dictionary.tokenize(text, False)
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [dataset.dictionary.padding_idx] * \
                (max_length - len(tokens))
            tokens = tokens + padding
        utils.assert_eq(len(tokens), max_length)
        tokens = torch.tensor([tokens]).to(device)
        return tokens

    question = str(question)
    tokens = tokenize(question, eval_dset)

    model_path = './saved_models/BAN_MEVF/model_epoch19.pth'
    model_data = torch.load(model_path, map_location=args.map_location)
    model = model.to(device)
    model.load_state_dict(model_data.get('model_state', model_data))

    model.train(False)
    features, _ = model(image_tensors, tokens)
    logits = model.classifier(features)
    prediction = torch.max(logits, 1)[1].data  # argmax

    return tokens, eval_dset.label2ans[prediction.item()],


st.title('Q&MedicAid :bulb:')

st.markdown('''
    This app uses state of the art deep learning methods to answer medically related questions about an image :rocket:

    Example images to use available here: https://github.com/medtorch/MICCAI19-MedVQA/tree/master/data_RAD/images
    
    Built by [Nalin Nagar](https://github.com/NNDEV1/)
    ''')


uploaded_file = st.sidebar.file_uploader(
    "Upload your input medical image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    img.save("img.jpg")

    st.info("Uploaded image:")
    st.image(img)

    with st.form(key="my_form2"):

        question = st.text_area(
            "Question to ask about medical image", "Is there evidence of an aortic aneurysm?")

        submit_button = st.form_submit_button(label="Submit")

    if submit_button:

        tokenized_question, prediction = main("./img.jpg", question)

        st.info(f"Tokenized question: {tokenized_question}")
        st.info(f"Predicted answer: {prediction}")

else:

    st.info("Awaiting image")
