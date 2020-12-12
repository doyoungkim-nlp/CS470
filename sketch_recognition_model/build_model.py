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

from fitting_function import fit_conv, fit_model, fit_other

def load_data():
    print("Loading data \n")

    # Check for already loaded datasets
    if not(path.exists('xtrain_doodle.pickle')):
        # Load from web
        print("Loading data from the web \n")

        # Classes we will load
        categories = ['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword']

        # Dictionary for URL and class labels
        URL_DATA = {}
        for category in categories:
            URL_DATA[category] = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/' + category +'.npy'

        # Load data for classes in dictionary
        classes_dict = {}
        for key, value in URL_DATA.items():
            response = requests.get(value)
            classes_dict[key] = np.load(BytesIO(response.content))

        # Generate labels and add labels to loaded data
        for i, (key, value) in enumerate(classes_dict.items()):
            value = value.astype('float32')/255.
            if i == 0:
                classes_dict[key] = np.c_[value, np.zeros(len(value))]
            else:
                classes_dict[key] = np.c_[value,i*np.ones(len(value))]

        # Create a dict with label codes
        label_dict = {0:'cannon',1:'eye', 2:'face', 3:'nail', 4:'pear',
                      5:'piano',6:'radio', 7:'spider', 8:'star', 9:'sword'}

        lst = []
        for key, value in classes_dict.items():
            lst.append(value[:3000])
        doodles = np.concatenate(lst)

        # Split the data into features and class labels (X & y respectively)
        y = doodles[:,-1].astype('float32')
        X = doodles[:,:784]

        # Split each dataset into train/test splits
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    else:
        # Load data from pickle files
        print("Loading data from pickle files \n")

        file = open("xtrain_doodle.pickle",'rb')
        X_train = pickle.load(file)
        file.close()

        file = open("xtest_doodle.pickle",'rb')
        X_test = pickle.load(file)
        file.close()

        file = open("ytrain_doodle.pickle",'rb')
        y_train = pickle.load(file)
        file.close()

        file = open("ytest_doodle.pickle",'rb')
        y_test = pickle.load(file)
        file.close()

    return X_train, y_train, X_test, y_test

def save_data(X_train, y_train, X_test, y_test, force = False):
    print("Saving data \n")

    # Check for already saved files
    if not(path.exists('xtrain_doodle.pickle')) or force:
        # Save X_train dataset as a pickle file
        with open('xtrain_doodle.pickle', 'wb') as f:
            pickle.dump(X_train, f)

        # Save X_test dataset as a pickle file
        with open('xtest_doodle.pickle', 'wb') as f:
            pickle.dump(X_test, f)

        # Save y_train dataset as a pickle file
        with open('ytrain_doodle.pickle', 'wb') as f:
            pickle.dump(y_train, f)

        # Save y_test dataset as a pickle file
        with open('ytest_doodle.pickle', 'wb') as f:
            pickle.dump(y_test, f)

def build_model(input_size, output_size, hidden_sizes, architecture = 'nn', dropout = 0.0):
    if (architecture == 'nn'):
        # Build a feed-forward network
        model = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                              ('bn2', nn.BatchNorm1d(num_features=hidden_sizes[1])),
                              ('relu2', nn.ReLU()),
                              ('dropout', nn.Dropout(dropout)),
                              ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                              ('bn3', nn.BatchNorm1d(num_features=hidden_sizes[2])),
                              ('relu3', nn.ReLU()),
                              ('logits', nn.Linear(hidden_sizes[2], output_size))]))

    else:
        if (architecture == 'conv'):
            # Build a simple convolutional network
            model = SimpleCNN(64, output_size)
        elif (architecture == 'simpleNet') : 
            model = SimpleNet_v1(output_size)
        elif (architecture == 'vgg8b') : 
            model = vgg8b()

    return model

def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    ts = time.time()
    plt.savefig('prediction' + str(ts) + '.png')

def save_model(model, architecture, input_size, output_size, hidden_sizes, dropout, filepath = 'checkpoint.pth'):
    if architecture == 'nn':
        checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_sizes,
                  'dropout': dropout,
                  'state_dict': model.state_dict()}

        torch.save(checkpoint, filepath)
    else:
        if architecture == 'conv' : 
            filepath = 'checkpoint_conv.pth'
        elif architecture == "simpleNet" : 
            filepath = "checkpoint_simpleNet.pth"
        elif architecture == "vgg8b" : 
            filepath = "checkpoint_vgg8b.pth"
        checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_sizes,
                  'dropout': dropout,
                  'state_dict': model.state_dict()}
        torch.save(checkpoint, filepath)
    print("Saving model to {}\n".format(filepath))

def load_model(architecture = 'nn', filepath = 'checkpoint.pth'):
    print("Loading model from {} \n".format(filepath))

    if architecture == 'nn':
        checkpoint = torch.load(filepath)
        input_size = checkpoint['input_size']
        output_size = checkpoint['output_size']
        hidden_sizes = checkpoint['hidden_layers']
        dropout = checkpoint['dropout']
        model = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                              ('bn2', nn.BatchNorm1d(num_features=hidden_sizes[1])),
                              ('relu2', nn.ReLU()),
                              ('dropout', nn.Dropout(dropout)),
                              ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                              ('bn3', nn.BatchNorm1d(num_features=hidden_sizes[2])),
                              ('relu3', nn.ReLU()),
                              ('logits', nn.Linear(hidden_sizes[2], output_size))]))
        model.load_state_dict(checkpoint['state_dict'])

    elif architecture == 'SimpleCNN':
        filepath = "checkpoint_conv.pth"
        checkpoint = torch.load(filepath)
        model = SimpleCNN()
        model.load_state_dict(checkpoint['state_dict'])
    elif architecture == 'SimpleNet_v1':
        filepath = "checkpoint_simpleNet.pth"
        checkpoint = torch.load(filepath)
        model = SimpleNet_v1(output_size)
        model.load_state_dict(checkpoint['state_dict'])
    elif architecture == 'vgg8b':
        filepath = "checkpoint_vgg8b.pth"
        checkpoint = torch.load(filepath)
        model = vgg8b()
        model.load_state_dict(checkpoint['state_dict'])

    return model

def test_model(model, img, architecture = 'nn'):
    # Convert 2D image to 1D vector
    img = img.resize_(1, 784)

    ps = get_preds(model, img, architecture = architecture)
    view_classify(img.resize_(1, 28, 28), ps)

def get_preds(model, input, architecture = 'nn'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Turn off gradients to speed up this part
    with torch.no_grad():
        if architecture == 'nn':
            logits = model.forward(input)
        else : 
            image = input.numpy()
            image = image.reshape(image.shape[0], 1, 28, 28)
            logits = model.forward(torch.from_numpy(image).float().to(device))
            #logits = model.forward(torch.from_numpy(image).float().to("cpu"))
    ps = F.softmax(logits, dim=1)
    return ps

def get_labels(pred):
    pred_np = pred.numpy()
    pred_values = np.amax(pred_np, axis=1, keepdims=True)
    pred_labels = np.array([np.where(pred_np[i, :] == pred_values[i, :])[0] for i in range(pred_np.shape[0])])
    pred_labels = pred_labels.reshape(len(pred_np), 1)

    return pred_labels


## evaluate_model can cause GPU memory out => Test dataset should be small enough
def evaluate_model(model, train, y_train, test, y_test, architecture = 'nn'):
    train_pred = get_preds(model, train, architecture)
    train_pred_labels = get_labels(train_pred)

    test_pred = get_preds(model, test, architecture)
    test_pred_labels = get_labels(test_pred)

    accuracy_train = accuracy_score(y_train, train_pred_labels)
    accuracy_test = accuracy_score(y_test, test_pred_labels)

    print("Accuracy score for train set is {} \n".format(accuracy_train))
    print("Accuracy score for test set is {} \n".format(accuracy_test))

    return accuracy_train, accuracy_test


def plot_learning_curve(model, input_size, output_size, hidden_sizes, train, labels, y_train, test, y_test, architecture = "nn", learning_rate = 0.003, weight_decay = 0.0, dropout = 0.0, n_chunks = 1000, optimizer = 'SGD'):
    train_acc = []
    test_acc = []

    for epochs in np.arange(5, 30, 5):
        # create model
        #model = build_model(input_size, output_size, hidden_sizes, dropout = dropout)

        # fit model
        fit_model(model, train, labels, epochs = epochs, n_chunks = n_chunks, learning_rate = learning_rate, weight_decay = weight_decay, optimizer = optimizer)
        # get accuracy
        accuracy_train, accuracy_test = evaluate_model(model, train, y_train, test, y_test)

        train_acc.append(accuracy_train)
        test_acc.append(accuracy_test)

    #return train_acc, test_acc
    
    """
    # Plot curve
    x = np.arange(10, 210, 10)
    plt.plot(x, train_acc)
    plt.plot(x, test_acc)
    plt.legend(['train', 'test'], loc='upper left')
    plt.title('Accuracy, learning_rate = ' + str(learning_rate), fontsize=20)
    plt.xlabel('Number of epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    ts = time.time()
    plt.savefig('learning_curve' + str(ts) + "_" + architecture + '.png')
    """
    ts = time.time()
    dir_path = os.path.join("/content/drive/My Drive/CS470/Final/quick-draw-image-recognition-master/model",architecture)
    df = pd.DataFrame.from_dict({'train' : train_acc, 'test' :test_acc})
    file_path = 'learning_curve_' + str(ts) + "_" + architecture + '.csv'
    df.to_csv(os.path.join(dir_path, file_path))



def plot_learning_curve_conv(model, input_size, output_size, hidden_sizes, train, labels, y_train, test, y_test, architecture="conv", learning_rate = 0.003, weight_decay = 0.0, dropout = 0.0, n_chunks = 1000, optimizer = 'SGD'):
    train_acc = []
    test_acc = []

    for epochs in np.arange(2, 10, 2):
        print("epoch : ", epochs)
        # create model
        model = build_model(input_size, output_size, hidden_sizes, architecture=architecture, dropout = dropout)
        print("model build complete")

        # fit model
        #fit_conv(model, train, labels, test, y_test, architecture, epochs = epochs, n_chunks = n_chunks, learning_rate = learning_rate, weight_decay = weight_decay, optimizer = "SGD")
        if architecture == "conv" : 
          fit_conv(model, train, labels, epochs = epochs, n_chunks = n_chunks, learning_rate = learning_rate, weight_decay = weight_decay, optimizer = "SGD")
        else : 
          fit_other(model, train, labels, test, y_test, architecture, epochs = epochs, n_chunks = n_chunks, learning_rate = learning_rate, weight_decay = weight_decay, optimizer = optimizer)

        # get accuracy
        accuracy_train, accuracy_test = evaluate_model(model, train, y_train, test, y_test, architecture=architecture)

        print("accuracy_train type : ", type(accuracy_train))
        train_acc.append(accuracy_train)
        test_acc.append(accuracy_test)

    #return train_acc, test_acc

    """
    # Plot curve
    x = np.arange(2, 12, 2)
    plt.plot(x, train_acc)
    plt.plot(x, test_acc)
    plt.legend(['train', 'test'], loc='upper left')
    plt.title('Accuracy, learning_rate = ' + str(learning_rate), fontsize=20)
    plt.xlabel('Number of epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    ts = time.time()
    plt.savefig('learning_curve' + str(ts) + "_" + architecture + '.png')
    """
    ts = time.time()
    df = pd.DataFrame.from_dict({'train' : train_acc, 'test' :test_acc})
    df.to_csv('learning_curve_' + str(ts) + "_" + architecture + '.csv')


def compare_hyperparameters(input_size, output_size, hidden_sizes, train, labels, y_train, test, y_test, learning_rate, architecture = 'nn', n_chunks = 1000, optimizer = 'SGD', dropout=0.0, weight_decay=0.0):
    """
    Function which evaluates the accyracy of the model on set of hyperparameters dropout and weight_decay.
    """
    # define hyperparameters grid
    #weight_decays = [0.0, 0.5, 1.0, 2.0]
    #dropouts = [0.0]
    #optimizers = ["SGD", "ADAM"]
    learning_rates = [0.001, 0.003, 0.005]
    #learning_rates = [0.003]

    epochs = np.arange(2, 21, 2)

    results = {}
    params = []
    dir_path = os.path.join("/content/drive/MyDrive/CS470/Final/quick-draw-image-recognition-master/model",architecture)

    # train and evaluate models with different hyperparameters
    for learning_rate in learning_rates:

        test_acc = []
        train_acc = []

        for e in epochs:
            model = build_model(input_size, output_size, hidden_sizes, architecture = architecture, dropout = dropout)
            
            if architecture == 'nn' : 
                fit_model(model, train, labels, test, y_test, architecture, epochs = e, n_chunks = n_chunks, learning_rate = learning_rate, weight_decay = weight_decay, optimizer = optimizer)
            elif architecture == 'conv' : 
                fit_conv(model, train, labels, test, y_test, architecture, epochs = e, n_chunks = n_chunks, learning_rate = learning_rate, weight_decay = weight_decay, optimizer = optimizer)
            else : 
                fit_other(model, train, labels, test, y_test, architecture, epochs = epochs, n_chunks = n_chunks, learning_rate = learning_rate, weight_decay = weight_decay, optimizer = optimizer)
            accuracy_train, accuracy_test = evaluate_model(model, train, y_train, test, y_test)

            train_acc.append(accuracy_train)
            test_acc.append(accuracy_test)

        results['test lr: ' + str(learning_rate)] = test_acc
        results['train lr: ' + str(learning_rate)] = train_acc
        params.append('lr: ' + str(learning_rate))

        # save intermediate results
        df = pd.DataFrame.from_dict(results)
        filename = "comparison_loss_function_NLLLoss" + ".csv"
        df.to_csv(os.path.join(dir_path, filename))

    print(results)

    ts = time.time()

    # save results as csv
    df = pd.DataFrame.from_dict(results)
    filename = 'comparison_' + str(ts) + "_lr_" + str(learning_rate)+ '.csv'
    df.to_csv(filename)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Argument parser')

    parser.add_argument('--save_dir', action='store', default = ' ',
                        help='Directory to save model checkpoint')

    parser.add_argument('--learning_rate', type = float, action='store', default = 0.003,
                        help='Model hyperparameters: learning rate')

    parser.add_argument('--epochs', type = int, action='store', default = 3,
                        help='Model hyperparameters: epochs')

    parser.add_argument('--weight_decay', type = float, action='store', default = 0,
                        help='Model hyperparameters: weight decay (regularization)')

    parser.add_argument('--dropout', type = float, action='store', default = 0.0,
                        help='Model hyperparameters: dropout')

    parser.add_argument('--architecture', action='store', default = 'nn',
                        help='Model architecture: nn - feed forward neural network with 1 hidden layer.',
                        choices = ['nn', 'conv', 'simpleNet', 'vgg8b'])

    parser.add_argument('--add_data', action='store_true',
                        help='Add flipped and rotated images to the original training set.')

    parser.add_argument('--mini_batches', type = int, action='store', default = 1000,
                        help='Number of minibatches.')

    parser.add_argument('--optimizer', action='store', default = 'SGD',
    choices=['SGD', 'Adam'],
    help='Optimizer for fitting the model.')

    parser.add_argument('--gpu', action='store_true',
                        help='Run training on GPU')
    results = parser.parse_args()

    learning_rate = results.learning_rate
    epochs = results.epochs
    weight_decay = results.weight_decay
    dropout = results.dropout
    architecture = results.architecture
    n_chunks = results.mini_batches
    optimizer = results.optimizer

    if (results.gpu == True):
        device = 'cuda'
    else:
        device = 'cpu'

    """
    if (results.save_dir == ' '):
        save_path = 'checkpoint.pth'
    else:
        save_path = results.save_dir + '/' + 'checkpoint.pth'
    """


    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Add flipped and rotated images to the dataset
    if (results.add_data == True):
        X_train, y_train = add_flipped_and_rotated_images(X_train, y_train)

    # Save datasets to disk if required
    save_data(X_train, y_train, X_test, y_test, force = results.add_data)

    # Convert to tensors
    train = torch.from_numpy(X_train).float()
    labels = torch.from_numpy(y_train).long()
    test = torch.from_numpy(X_test).float()
    test_labels = torch.from_numpy(y_test).long()

    # Hyperparameters for our network
    input_size = 784
    hidden_sizes = [128, 100, 64]
    output_size = 10

    # Build model
    model = build_model(input_size, output_size, hidden_sizes, architecture = architecture, dropout = dropout)

    
    # Fit model
    
    if (architecture == 'nn'):
        fit_model(model, train, labels, test, y_test, architecture, epochs = epochs, n_chunks = n_chunks, learning_rate = learning_rate, weight_decay = weight_decay, optimizer = optimizer)
    elif (architecture == "conv"):
        fit_conv(model, train, labels, test, y_test, architecture, epochs = epochs, n_chunks = n_chunks, learning_rate = learning_rate, weight_decay = weight_decay, optimizer = optimizer)
    else : 
        fit_other(model, train, labels, test, y_test, architecture, epochs = epochs, n_chunks = n_chunks, learning_rate = learning_rate, weight_decay = weight_decay, optimizer = optimizer)
    
    # Save the model
    save_model(model,architecture, input_size, output_size, hidden_sizes, dropout)
    
    """
    # plot learning curve
    if (architecture == 'nn') : 
        plot_learning_curve(model, input_size, output_size, hidden_sizes, train, labels, y_train, test, y_test, learning_rate = learning_rate, dropout = dropout, weight_decay = weight_decay, n_chunks = n_chunks, optimizer = optimizer)
    else : 
        plot_learning_curve_conv(model, input_size, output_size, hidden_sizes, train, labels, y_train, test, y_test, architecture=architecture, learning_rate = learning_rate, dropout = dropout, weight_decay = weight_decay, n_chunks = n_chunks, optimizer = optimizer)
    """

    # Evaluate model
    #evaluate_model(model, train, y_train, test, y_test, architecture = architecture)
    #test_model(model, test[0], architecture = architecture)
    

    #compare_hyperparameters(input_size, output_size, hidden_sizes, train, labels, y_train, test, y_test, learning_rate, architecture = architecture, n_chunks = n_chunks, optimizer = optimizer)

    #loaded_model = load_model(architecture)
    #loaded_model.eval()
    #pred = test_model(loaded_model, test[0], architecture = architecture)

if __name__ == '__main__':
    main()
