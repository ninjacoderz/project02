
import argparse
import sys

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

def set_args():
    parser = argparse.ArgumentParser(description='Set arguments')
    parser.add_argument('--data_dir', default='flowers', type=str, help='data directory')
    parser.add_argument('--arch', default='densenet', help='pretrained model architectures, options: vgg, densenet, alexnet')
    parser.add_argument('--hidden_units', default=[512], type=list, help='hidden layer architecture')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')    
    parser.add_argument('--epochs', default=10, type=int, help='training epochs')
    parser.add_argument('--save_dir', default='', type=str, help='save directory')
    parser.add_argument('--gpu', default= 'mps', type=str, help='devices, options: gpu, cpu, mps')     
    
    return parser.parse_args()  

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    means = [0.485, 0.456, 0.406]
    std_deviations = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, std_deviations)]),
        'val': transforms.Compose([ transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, std_deviations)]),
        'test': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, std_deviations)])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(valid_dir, data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test'])    
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val', 'test']
    }
    
    return image_datasets, dataloaders


MODEL_VGG = 'vgg'
MODEL_DENSENET = 'densenet'
MODEL_ALEXNET = 'alexnet'

def load_pretrained_model(model_name):
    if model_name == MODEL_VGG:        
        model = models.vgg16(weights='DenseNet121_Weights.DEFAULT')
        ins = 25088
        print('Loading the model vgg...')
    elif model_name == MODEL_DENSENET:
        model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        ins = 1024
        print('Loading the densenet model...')
        
    elif model_name == MODEL_ALEXNET:
        model = models.alexnet(weights='DenseNet121_Weights.DEFAULT')
        ins = 9216
        print('Loading the alexnet model...')
    else:
        print('Model not recognized')
        sys.exit
    print('Model load successed.')
    return model, ins

def build_classifier(model, ins, layer_list, outs=102,  drop_p = 0.5):
    for param in model.parameters():
        param.requires_grad = False
    net_layers = OrderedDict()
    for i in range(len(layer_list)+1):
        if i ==0:
            h1 = ins
        if i < len(layer_list):
            h2 = layer_list[i]
            net_layers.update({'fc{}'.format(i): nn.Linear(h1, h2)})
            net_layers.update({'relu{}'.format(i):nn.ReLU()})
            net_layers.update({'drop{}'.format(i):nn.Dropout(drop_p)})
            h1 = h2
        else:
            h2 = outs
            net_layers.update({'fc{}'.format(i):nn.Linear(h1, h2)})
            net_layers.update({'output':nn.LogSoftmax(dim=1)})
    
    clf= nn.Sequential(net_layers)
    model.classifier = clf
    return model 

def train_model(model, criterion, optimizer, epochs, device, dataloaders):    
    model.to(device)
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)                
            optimizer.zero_grad()        
            logps = model(inputs)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()        
            running_loss +=loss.item()
        else:
            valid_loss = 0
            accuracy = 0        
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['val']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    logps = model(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print("Training Loss: {:.3f}.. ".format(running_loss/len(dataloaders['train'])),
                  "Validation Loss: {:.3f}.. ".format(valid_loss/len(dataloaders['val'])),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders['val'])))
    
    return model, accuracy/len(dataloaders['val'])

def save_model(image_datasets, model, optimizer, criterion, epochs, save_dir=''):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = { 
        'pretrain_model': 'densenet',
        'classifier': model.classifier,
        'optimizer' : optimizer,
        'criterion' : criterion,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict() ,
        'epochs':epochs,
        'class_to_idx': model.class_to_idx
        }
    torch.save(checkpoint, save_dir+'checkpoint.pth')
    
def main():
    args = set_args()
    print(args)
    image_datasets, dataloaders = load_data(args.data_dir)
    model, ins = load_pretrained_model(args.arch)    
    model = build_classifier(model, ins, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)
    model, accuracy = train_model(model, criterion, optimizer, args.epochs, args.gpu, dataloaders)
    save_model(image_datasets, model, optimizer, criterion, args.epochs, args.save_dir)
    
if __name__ == '__main__':
    main()
