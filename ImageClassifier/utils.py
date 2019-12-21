# Imports here
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
from collections import OrderedDict
import argparse

model_name = {"vgg16":25088, "alexnet":9216}

def transformation(data):
    data_dir = data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Transforming the training, validation, and testing sets.
    train_data_transforms = transforms.Compose ([transforms.RandomRotation(30),
                                                 transforms.RandomResizedCrop(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    valid_data_transforms = transforms.Compose ([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    test_data_transforms = transforms.Compose ([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                               ])

    # Loading datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir,transform=train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir,transform=valid_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir,transform=test_data_transforms)

    return train_image_datasets, valid_image_datasets, test_image_datasets

def loader(data):

    data_dir = data
    train_image_datasets, valid_image_datasets, test_image_datasets = transformation(data_dir)

    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=64, shuffle=True)

    return train_dataloaders, valid_dataloaders, test_dataloaders

train_image_datasets, valid_image_datasets, test_image_datasets = transformation('./flowers/')
train_dataloaders, valid_dataloaders, test_dataloaders = loader('./flowers/')

def network(arch='vgg16', dropout=0.3, hidden_units=2048, lr=0.001, device='gpu'):

    # Defining models
    if arch == 'vgg16':
       model = models.vgg16(pretrained=True)
       model    
    elif arch == 'alexnet':
       model = models.alexnet(pretrained=True)
       model
    else:
       print("Use vgg16 or alexnet")  

    for param in model.parameters():
        param.requires_grad=False
            
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(model_name[arch], 4096)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(4096, hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=dropout)),
                          ('fc3', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    if torch.cuda.is_available() and device == 'gpu':
        model.to('cuda')
    else:    
        model.to('cpu')
        
    return model, optimizer, criterion

def validation(model, valid_dataloaders, criterion, device):
    with torch.no_grad():
        if device == 'gpu':
            model.to('cuda')
        else:   
            model.to('cpu')
        loss = 0
        accuracy = 0

        for inputs, labels in valid_dataloaders:
            if torch.cuda.is_available() and device == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                inputs, labels = inputs.to('cpu'), labels.to('cpu')
                
            output = model.forward(inputs)
            loss += criterion(output, labels).item()
            ps = torch.exp(output)  
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return loss, accuracy

def Train(optimizer, criterion, model, epochs, device):
    if device == 'gpu':
        model.to('cuda')
    else:    
        model.to('cpu')
    steps = 0
    print_every = 40

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_dataloaders):
            steps += 1
            if device == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:    
                inputs, labels = inputs.to('cpu'), labels.to('cpu') 
            optimizer.zero_grad()
        
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                
                loss, accuracy = validation(model, valid_dataloaders, criterion, device)

                print("Epoch: {}/{}".format(e+1, epochs),
                      "Training Loss: {:.3f}".format(running_loss/print_every),
                      "Valid Loss: {:.3f}".format(loss/len(valid_dataloaders)),
                      "Accuracy Loss: {:.3f}".format(accuracy/len(valid_dataloaders)))

                running_loss = 0
                model.train()

    print("\nProcess is complete!!")
    
def testing(model, test_dataloaders, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_dataloaders:
            inputs, labels = data
            if device == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:    
                inputs, labels = inputs.to('cpu'), labels.to('cpu')
            outputs = model(inputs)
            _, predicted = torch.max (outputs.data,1)
            total += labels.size (0)
            correct += (predicted == labels).sum().item()
            model.eval()

    print('Accuracy: %d %%' % (100 * correct / total))
    return total

def save_checkpoint(model, path="checkpoint.pth", arch="vgg16", epochs=6, hidden_units=4096, dropout=0.3, lr=0.001):

  
    model.class_to_idx = train_image_datasets.class_to_idx
    model.cpu
    
    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                  }

    torch.save(checkpoint, 'checkpoint.pth')

def loading_checkpoint(path):
    
    model,_,_ = network()
    model.cpu
    checkpoint = torch.load(path)

    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    return model
    

def process_image(image):
    
    im = Image.open(image)
    
    width, height = im.size 
    
    if width > height: 
        height = 256
        im.thumbnail ((50000, height), Image.ANTIALIAS)
    else: 
        width = 256
        im.thumbnail ((width, 50000), Image.ANTIALIAS)
        
    width, height = im.size
    reduce = 224
    left = (width - reduce) / 2 
    top = (height - reduce) / 2
    right = left + 224 
    bottom = top + 224
    im = im.crop ((left, top, right, bottom))
      
    np_image = np.array (im)/255 
    np_image -= np.array ([0.485, 0.456, 0.406]) 
    np_image /= np.array ([0.229, 0.224, 0.225])
    
    np_image= np_image.transpose ((2,0,1))
    return np_image


def predict(image_path, model, topk, device, json_file):
    
    model.eval()
    #model,_,_ = network()
    if torch.cuda.is_available and device == 'gpu':
        model.to('cuda')
    else:    
        model.to('cpu')
    pred_img = process_image(image_path)
    # Converting to tensor
    image = torch.from_numpy(pred_img).type(torch.FloatTensor)
    # Removing RunTimeError for missing batch size - add batch size of 1 
    image = image.unsqueeze_(dim=0)
           
    with torch.no_grad():
        if device == 'gpu':
            output = model.forward(image.cuda())
        else:    
            output = model.forward(image)
        probability_output = torch.exp(output) #Converted to probability        
        probs, indices = probability_output.topk(topk)    
        # Detach top probabilities into a numpy list
        probs = probs.cpu().numpy().tolist()[0]
        
        # Change top indices into a list
        indices = indices.tolist()[0]
        
        #Converting to Classes
        class_to_idx = model.class_to_idx
        class_to_idx = {value: key for key, value in model.class_to_idx.items()}        
        classes = [class_to_idx[item] for item in indices]
        classes = np.array(classes)
        
        cat_to_name = cat_name(json_file)
        
        class_names = [cat_to_name [i] for i in classes]
        for a, b in zip(class_names, probs):
            result = print("- {} with a probability of {}".format(a, b))
        
        return result
    
def cat_name(json_file):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
