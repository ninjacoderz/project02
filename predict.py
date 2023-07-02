import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import pandas as pd 
import numpy as np
import PIL
from PIL import Image
from collections import OrderedDict
from torch.autograd import Variable
from train import load_pretrained_model
import subprocess

import json

def set_args():
    parser = argparse.ArgumentParser(description='Set arguments')
    parser.add_argument('--input_dir', default='flowers/test/13/image_05775.jpg', type=str, help='image to process and predict')
    parser.add_argument('--checkpoint', default='checkpoint.pth', type=str, help='checkpoint directory')  
    parser.add_argument('--json_dir', default='cat_to_name.json', type=str, help=' mapping the category to name') 
    parser.add_argument('--topk', default=5, type=int, help='top k value')
    parser.add_argument('--gpu', default= 'cpu', type=str, help='devices, options: gpu, cpu, mps')  
    
    return parser.parse_args() 

def load_model(checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir)
    model, ins = load_pretrained_model(checkpoint['pretrain_model'])
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']  
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = checkpoint['optimizer']
    criterion = checkpoint['criterion']
    epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    return model, optimizer, criterion, epochs

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    size = np.max(pil_image.size)*256/np.min(pil_image.size)
    pil_image.thumbnail((size,size))
    pil_image = pil_image.crop((16, 16, 240, 240))
    np_image = np.array(pil_image)/225
    
    means = np.array([0.485,0.456,0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means)/std
    
    np_image = np_image.transpose(2,0,1)
    torch_image = torch.from_numpy(np_image)
    return torch_image



def predict(image, model, gpu, json_path, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.to('cpu')
    image = process_image(image)
    image = Variable(image.unsqueeze(0).float())
    
    if torch.cuda.is_available() and gpu == 'gpu':
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else:
        device = torch.device("cpu") 
    
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    
    with torch.no_grad():
        predict = model.forward(image)
        results = torch.exp(predict).topk(topk, dim=1)
    
    probs = results[0][0].cpu().numpy()
    ids = results[1][0].cpu().numpy() 
    class_ids =  model.class_to_idx
    id_classes = {}
    for x in class_ids:
        id_classes[class_ids[x]] = x  
    classes = [id_classes[x] for x in ids.tolist()]
    
    names = []
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    for c in classes:
        names.append(cat_to_name[c])
        
    return probs, ids, classes, names

def plot_results(probs, names, ax = None):
    print('Probilities:', probs)
    print('Names:', names) 
    if ax is None:
        fig, ax = plt.subplots()
    names_df = pd.DataFrame(names)
    probs_df = pd.DataFrame(probs)
    name_prob = pd.concat([names_df, probs_df],axis =1)
    name_prob.columns = ['names', 'probs']
    name_prob.sort_values('probs', ascending = True, inplace = True)
    name_prob['probs'].plot.barh(ax =ax)
    plt.yticks(range(len(name_prob)),name_prob['names'])
    plt.show()
    return ax

def main():
    args = set_args()
    print(args)
    model, optimizer, criterion, epochs = load_model(args.checkpoint)
#     print(optimizer)
#     print(epochs)
#     image = process_image(args.input_dir)
#     imshow(image)
#     plt.show()
    probs, ids, classes, names = predict(args.input_dir, model, args.gpu, args.json_dir, args.topk)
    print(probs)
    plot_results(probs, names)
    plt.show()
    
if __name__ == '__main__':
    main()
