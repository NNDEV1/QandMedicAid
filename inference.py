import argparse
import torch
from torch.utils.data import DataLoader
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

eval_dset = dataset_RAD.VQAFeatureDataset('test', args, dictionary)
model = base_model.build_BAN(eval_dset, args)


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
    "./data_RAD/images/synpic42202.jpg")


def tokenize(text, dataset, max_length=12):
    tokens = dataset.dictionary.tokenize(text, False)
    tokens = tokens[:max_length]
    if len(tokens) < max_length:
        # Note here we pad in front of the sentence
        padding = [dataset.dictionary.padding_idx] * (max_length - len(tokens))
        tokens = tokens + padding
    utils.assert_eq(len(tokens), max_length)
    tokens = torch.tensor([tokens]).to(device)
    return tokens


question = 'is there a ribcage ?'
tokens = tokenize(question, eval_dset)
print(tokens)

model_path = './saved_models/BAN_MEVF/model_epoch19.pth'
model_data = torch.load(model_path, map_location=args.map_location)
model = model.to(device)
model.load_state_dict(model_data.get('model_state', model_data))

model.train(False)
features, _ = model(image_tensors, tokens)
logits = model.classifier(features)
prediction = torch.max(logits, 1)[1].data  # argmax
print(eval_dset.label2ans[prediction.item()])
