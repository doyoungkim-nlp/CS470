# import linear algebra and data manipulation libraries
import numpy as np
import pandas as pd

# import matplotlib for plotting
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# import helper libraries
import requests
from io import BytesIO # Use When expecting bytes-like objects
import pickle
from collections import OrderedDict
import os
from os import path
import time
import argparse

# import PIL for image manipulation
from PIL import Image

# import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import image_utils
from image_utils import add_flipped_and_rotated_images

from models.simple_conv_nn import SimpleCNN
from models.SimpleNet import SimpleNet_v1
from models.VGG8b import vgg8b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device : ", device)
def get_preds(model, input, architecture = 'nn'):
    """
    Function to get predicted probabilities from the model for each class.

    INPUT:
        model - pytorch model
        input - (tensor) input vector

    OUTPUT:
        ps - (tensor) vector of predictions
    """
    # Turn off gradients to speed up this part
    with torch.no_grad():
        if architecture == 'nn':
            logits = model.forward(input)
        else : 
            image = input.numpy()
            image = image.reshape(image.shape[0], 1, 28, 28)
            #logits = model.forward(torch.from_numpy(image).float().to(device))
            logits = model.forward(torch.from_numpy(image).float().to("cpu"))
    ps = F.softmax(logits, dim=1)
    return ps

def get_labels(pred):
    """
        Function to get the vector of predicted labels for the images in
        the dataset.

        INPUT:
            pred - (tensor) vector of predictions (probabilities for each class)
        OUTPUT:
            pred_labels - (numpy) array of predicted classes for each vector
    """

    pred_np = pred.numpy()
    pred_values = np.amax(pred_np, axis=1, keepdims=True)
    pred_labels = np.array([np.where(pred_np[i, :] == pred_values[i, :])[0] for i in range(pred_np.shape[0])])
    pred_labels = pred_labels.reshape(len(pred_np), 1)

    return pred_labels

def evaluate_model(model, train, y_train, test, y_test, architecture = 'nn'):
    """
    Function to print out train and test accuracy of the model.

    INPUT:
        model - pytorch model
        train - (tensor) train dataset
        y_train - (numpy) labels for train dataset
        test - (tensor) test dataset
        y_test - (numpy) labels for test dataset

    OUTPUT:
        accuracy_train - accuracy on train dataset
        accuracy_test - accuracy on test dataset
    """
    train_pred = get_preds(model, train, architecture)
    train_pred_labels = get_labels(train_pred)

    test_pred = get_preds(model, test, architecture)
    test_pred_labels = get_labels(test_pred)

    accuracy_train = accuracy_score(y_train, train_pred_labels).item()
    accuracy_test = accuracy_score(y_test, test_pred_labels).item()

    #print("Accuracy score for train set is {} \n".format(accuracy_train))
    #print("Accuracy score for test set is {} \n".format(accuracy_test))

    return round(accuracy_train, 5), round(accuracy_test, 5)

def shuffle(X_train, y_train):
    """
    Function which shuffles training dataset.
    INPUT:
        X_train - (tensor) training set
        y_train - (tensor) labels for training set

    OUTPUT:
        X_train_shuffled - (tensor) shuffled training set
        y_train_shuffled - (tensor) shuffled labels for training set
    """
    X_train_shuffled = X_train.numpy()
    y_train_shuffled = y_train.numpy().reshape((X_train.shape[0], 1))

    permutation = list(np.random.permutation(X_train.shape[0]))
    X_train_shuffled = X_train_shuffled[permutation, :]
    y_train_shuffled = y_train_shuffled[permutation, :].reshape((X_train.shape[0], 1))

    X_train_shuffled = torch.from_numpy(X_train_shuffled).float()
    y_train_shuffled = torch.from_numpy(y_train_shuffled).long()

    return X_train_shuffled, y_train_shuffled


# fitting function for simple CNN
def fit_conv(model, X_train, y_train, test, y_test, architecture, epochs = 100, n_chunks = 1000, learning_rate = 0.003, weight_decay = 0, optimizer = 'SGD'):
    print("fit_conv")
    print("Fitting model with epochs = {epochs}, learning rate = {lr}\n"\
    .format(epochs = epochs, lr = learning_rate))

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    #m = nn.LogSoftmax(dim=1)

    #torchsummary.summary(model, (1, 28, 28))

    if (optimizer == 'SGD'):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay= weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)

    print_every = 100

    steps = 0

    train_acc = []
    test_acc = []
    loss_list = []

    for e in range(epochs):
        running_loss = 0

        X_train, y_train = shuffle(X_train, y_train)

        images = torch.chunk(X_train, n_chunks)
        labels = torch.chunk(y_train, n_chunks)

        for i in range(n_chunks):
            steps += 1

            optimizer.zero_grad()

            # Forward and backward passes
            np_images = images[i].numpy()
            np_images = np_images.reshape(images[i].shape[0], 1, 28, 28)
            img = torch.from_numpy(np_images).float()

            output = model.forward(img)
            # CrossEntropy 
            loss = criterion(output, labels[i].squeeze())
            # NLLLoss
            #loss = criterion(m(output), labels[i].squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
                #loss_list.append(running_loss/print_every)
                running_loss = 0
    

        ## codes for evaluation & make accuracy .csv file
        #accuracy_train, accuracy_test = evaluate_model(model, X_train, y_train, x_test, y_test)
    
        #print("train acc : ", accuracy_train)
        #print("loss length : , ", len(loss_list))
    #accuracy_train, accuracy_test = evaluate_model(model, X_train, y_train, test, y_test, architecture=architecture)
    #print(accuracy_train)
    #train_acc.append(accuracy_train)
    #test_acc.append(accuracy_test)

    """
    ts = time.time()
    dir_path = os.path.join("/content/drive/My Drive/CS470/Final/quick-draw-image-recognition-master/model",architecture)
    #df = pd.DataFrame.from_dict({'train' : train_acc, 'test' :test_acc})
    #file_path = 'learning_curve_' + str(ts) + "_" + architecture + '.csv'
    #df.to_csv(os.path.join(dir_path, file_path))
    df2 = pd.DataFrame.from_dict({'loss' : loss_list})
    file_path2 = 'train_loss_curve_' + str(ts) + "_" + architecture + '.csv'
    df2.to_csv(os.path.join(dir_path, file_path2))
    """


                
# fitting function for feed-forward neural network
def fit_model(model, X_train, y_train, x_test, y_test, architecture, epochs = 100, n_chunks = 1000, learning_rate = 0.003, weight_decay = 0, optimizer = 'SGD'):
    print("fit_model")
    print("Fitting model with epochs = {epochs}, learning rate = {lr}\n"\
    .format(epochs = epochs, lr = learning_rate))

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    m = nn.LogSoftmax(dim=1)

    if (optimizer == 'SGD'):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay= weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)

    print_every = 100

    steps = 0

    train_acc = []
    test_acc = []
    loss_list = []
    for e in range(epochs):
        running_loss = 0

        X_train, y_train = shuffle(X_train, y_train)

        images = torch.chunk(X_train, n_chunks)
        labels = torch.chunk(y_train, n_chunks)

        for i in range(n_chunks):
            steps += 1

            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images[i])
            # CrossEntropyLoss
            #loss = criterion(output, labels[i].squeeze())
            #NLLLoss
            loss = criterion(m(output), labels[i].squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
                loss_list.append(running_loss/print_every)
                running_loss = 0

        ## codes for evaluation & make accuracy .csv file
        #accuracy_train, accuracy_test = evaluate_model(model, X_train, y_train, x_test, y_test)
        #print("accuracy_train type : ", type(accuracy_train))
        #train_acc.append(accuracy_train)
        #test_acc.append(accuracy_test)
    
        #print("loss length : ", len(loss_list))
    """
    ts = time.time()
    dir_path = os.path.join("/content/drive/My Drive/CS470/Final/quick-draw-image-recognition-master/model",architecture)
    df = pd.DataFrame.from_dict({'train' : train_acc, 'test' :test_acc})
    file_path = 'learning_curve_' + str(ts) + "_" + architecture + '.csv'
    df2 = pd.DataFrame.from_dict({'loss' : loss_list})
    file_path2 = 'train_loss_curve_' + str(ts) + "_" + architecture + '.csv'
    df2.to_csv(os.path.join(dir_path, file_path2))
    """  

# fitting function for VGG8b, SimpleNet
def fit_other(model, X_train, y_train, x_test, y_test, architecture, epochs = 100, n_chunks = 1000, learning_rate = 0.003, weight_decay = 0, optimizer = 'SGD'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    #torchsummary.summary(model, (1, 28, 28))

    print("fit_other")
    print("Fitting model with epochs = {epochs}, learning rate = {lr}\n"\
    .format(epochs = epochs, lr = learning_rate))
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    #m = nn.LogSoftmax(dim=1)

    if (optimizer == 'SGD'):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay= weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)

    print_every = 100

    steps = 0
    train_acc = []
    test_acc = []
    loss_list = []
    for e in range(epochs):
        print("fitting epoch : ", e)
        running_loss = 0

        X_train, y_train = shuffle(X_train, y_train)

        images = torch.chunk(X_train, n_chunks)
        labels = torch.chunk(y_train, n_chunks)

        for i in range(n_chunks):
            steps += 1

            optimizer.zero_grad()

            
            # Forward and backward passes
            np_images = images[i].numpy()
            np_images = np_images.reshape(images[i].shape[0], 1, 28, 28)
            img = torch.from_numpy(np_images).float().to(device)
            
            output = model(img)
            
            # CrossEntropyLoss
            loss = criterion(output, labels[i].to(device).squeeze())
            # NLLLoss
            #loss = criterion(m(output), labels[i].to(device).squeeze())

            #losses.update(loss.item(), inputs.size(0))


            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
                loss_list.append(running_loss/print_every)
                running_loss = 0
    
        ## codes for evaluation & make accuracy .csv file
        #accuracy_train, accuracy_test = evaluate_model(model, X_train, y_train, x_test, y_test)
        #print("loss length : , ", len(loss_list))
    #accuracy_train, accuracy_test = evaluate_model(model, X_train, y_train, x_test, y_test, architecture=architecture)
    #print(accuracy_train)
    #train_acc.append(accuracy_train)
    #test_acc.append(accuracy_test)

    """
    ts = time.time()
    dir_path = os.path.join("/content/drive/My Drive/CS470/Final/quick-draw-image-recognition-master/model",architecture)
    #df = pd.DataFrame.from_dict({'train' : train_acc, 'test' :test_acc})
    #file_path = 'learning_curve_' + str(ts) + "_" + architecture + '.csv'
    #df.to_csv(os.path.join(dir_path, file_path))
    df2 = pd.DataFrame.from_dict({'loss' : loss_list})
    file_path2 = 'train_loss_curve_' + str(ts) + "_" + architecture + '.csv'
    df2.to_csv(os.path.join(dir_path, file_path2))
    """
