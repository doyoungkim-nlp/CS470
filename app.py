import os

from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

import cv2

from sketch_recognition_model.models.simple_conv_nn import SimpleCNN
from sketch_recognition_model.models.RMDL import RMDL
from sketch_recognition_model.models.SimpleNet import SimpleNet_v1
from sketch_recognition_model.models.VGG8b import vgg8b

import numpy as np
from PIL import Image
import base64
import re
from io import BytesIO
import base64
import io
import time
from collections import OrderedDict
import json

# import matplotlib for plotting
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# import Plotly
import plotly
#import plotly.plotly as py
import chart_studio.plotly as py
import plotly.graph_objs as go

# import Flask
from flask import Flask, render_template, request, jsonify

# import image processing
import sys
sys.path.insert(0, '../')
import sketch_recognition_model.image_utils
from sketch_recognition_model.image_utils import crop_image, normalize_image, convert_to_rgb, convert_to_np

# import pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms


from datetime import datetime




# import the required libraries
import numpy as np ## Downgrade numpy to 1.16.3
import time
import random
#import cPickle
import codecs
import collections
import os
import math
import json
import random
import tensorflow as tf
from six.moves import xrange
import svgwrite

# import matplotlib for plotting
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

# import Flask
from flask import Flask
from flask import render_template, request

from sketch_rnn.sketch_rnn_train import *
from sketch_rnn.model import *
from sketch_rnn.utils import *
from sketch_rnn.rnn import *

from sketch_generation_demo.functions import draw_strokes, make_grid_svg, load_env_compatible, load_model_compatible, encode, decode




# Dictionary with label codes
label_dict = {0:'cannon',1:'eye', 2:'face', 3:'nail', 4:'pear',
              5:'piano',6:'radio', 7:'spider', 8:'star', 9:'sword'}

def load_model(filepath = 'sketch_recognition_model/checkpoints/checkpoint_simpleNet.pth'):
    """
    Function loads the model from checkpoint.

    INPUT:
        filepath - path for the saved model

    OUTPUT:
        model - loaded pytorch model
    """

    print("Loading model from {} \n".format(filepath))

    checkpoint = torch.load(filepath)
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']


    
    """
    # 1. nn
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
    """

    """
    # 2. SimpleCNN
    model = SimpleCNN()
    model.load_state_dict(checkpoint['state_dict'])
    """

    
    # 3. SimpleNet
    model = SimpleNet_v1(output_size)
    model.load_state_dict(checkpoint['state_dict'])
    

    """
    # 4. RMDL
    model = RMDL(output_size)
    model.load_state_dict(checkpoint['state_dict'])
    """
    
    """
    # 5. VGG8b
    model = vgg8b()
    model.load_state_dict(checkpoint['state_dict'])
    """
    return model, input_size, output_size

def get_prediction(model, input):
    """
    Function to get prediction (label of class with the greatest probability).

    INPUT:
        model - pytorch model
        input - (numpy) input vector

    OUTPUT:
        label - predicted class label
        label_name - name of predicted class
    """
    # Convert input to tensor
    input = torch.from_numpy(input).float()
    input = input.resize(1, 784)

    # Turn off gradients to speed up this part
    with torch.no_grad():
        """
        # 1. nn
        logits = model.forward(input)
        """
        # 2. conv, simpleNet, RMDL, VGG8b
        image = input.numpy()
        image = image.reshape(image.shape[0], 1, 28, 28)
        logits = model.forward(torch.from_numpy(image).float())


    ps = F.softmax(logits, dim=1)

    # Convert to numpy
    preds = ps.numpy()

    # Return label corresponding to the max probability
    label = np.argmax(preds)
    label_name = label_dict[label] # get class name from dictionary

    return label, label_name, preds

def view_classify(img, preds):
    """
    Function for viewing an image and it's predicted classes
    with matplotlib.

    INPUT:
        img - (numpy) image file
        preds - (numpy) predicted probabilities for each class
    """
    preds = preds.squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,3), ncols=2)
    ax1.imshow(img.squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), preds)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    ts = time.time()

    plt.savefig('static/history/prediction' + str(ts) + '.png')
    plt.savefig('static/prediction.png')


    label_1 = np.argsort(preds)[-1]
    label_2 = np.argsort(preds)[-2]
    label_3 = np.argsort(preds)[-3]

    label_list = ['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword']

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), label_1, label_2, label_3)
    predicted = {
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            "category": "sketchmind",
            "image":'static/history/prediction' + str(ts) + '.png',
            "predicted": label_list[label_1] + ';' + label_list[label_2] + ';' + label_list[label_3],
            "correctness": "right!"
        }
    import json 
    
    
    # function to add to JSON 
    def write_json(data, filename='static/history.json'): 
        with open(filename,'w') as f: 
            json.dump(data, f, indent=4) 
        
        
    with open('static/history.json') as json_file: 
        data = json.load(json_file) 

                
        temp = data['history'] 
        # appending data to emp_details  
        temp.append(predicted) 
    write_json(data)  

def main() : 
    models_root_dir = './sketch_generation_demo/pretrained_model'
    model_list = ["sheep/layer_norm", "flamingo/lstm", "owl/lstm"]
    model_name = random.choice(model_list)
    model_dir = os.path.join(models_root_dir, model_name)
    base_model_dir = os.path.join(models_root_dir, "sheep/layer_norm")
    data_dir = 'http://github.com/hardmaru/sketch-rnn-datasets/raw/master/aaron_sheep/'

    [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env_compatible(data_dir, base_model_dir)
    [hps_model, eval_hps_model, sample_hps_model] = load_model_compatible(model_dir)

    reset_graph()
    model = Model(hps_model, tf.compat.v1.disable_eager_execution())
    eval_model = Model(eval_hps_model, reuse=True)
    sample_model = Model(sample_hps_model, reuse=True)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    load_checkpoint(sess, model_dir)

    
    N = 1
    reconstructions = []
    for i in range(N):
        reconstructions.append([decode(sess, sample_model, eval_model, temperature=0.5, draw_mode=False), [0, i]])

    stroke_grid = make_grid_svg(reconstructions)
    draw_strokes(stroke_grid)

    drawing = svg2rlg("./tmp/svg/sample.svg")

    ts = time.time()
    # filename = "./sketch_generation_demo/tmp/png/test" + str(ts) + ".png"
    # renderPM.drawToFile(drawing, filename, fmt="PNG")

    renderPM.drawToFile(drawing, "./static/history/revenge"+str(ts)+".png", fmt="PNG")
    import shutil

    original = "./static/history/revenge"+str(ts)+".png"
    target = "./static/revenge.png"
    shutil.copyfile(original, target)
    # label + revenge + time.png
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    predicted = {
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            "category": "revenge",
            "image":'static/history/revenge' + str(ts) + '.png',
            "predicted": "apple",
            "correctness": "right!"
        }
    import json 
    
    
    # function to add to JSON 
    def write_json(data, filename='static/history.json'): 
        with open(filename,'w') as f: 
            json.dump(data, f, indent=4) 
        
        
    with open('static/history.json') as json_file: 
        data = json.load(json_file) 

                
        temp = data['history'] 
        # appending data to emp_details  
        temp.append(predicted) 
    write_json(data) 

    return model_name.split('/')[0]





app = Flask(__name__)

# load model
model, input_size, output_size = load_model()
model.eval() # set to evaluation

modelName = ''

@app.route('/')
def start():
	return render_template('start.html')

@app.route('/canvas')
def canvas():
	return render_template('canvas.html')

@app.route('/gallery')
def gallery():
	return render_template('gallery.html')

@app.route('/result_revenge')
def result_revenge():
	return render_template('result_revenge.html')

@app.route('/how_to_play')
def how_to_play():
	return render_template('how_to_play.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        image_b64 = request.values['imageBase64']
        encoded_data = image_b64.split(',')[1]
        decoded = base64.b64decode(encoded_data)

        image_string = BytesIO(decoded)
        image_PIL = Image.open(image_string)
        image_np = np.array(image_PIL)
        # image_np에 저장됨.

        dateTime = request.values['dateTime']
        print("dateTime: ", dateTime)

        with open('some_image.jpg', 'wb') as f: 
            f.write(decoded)
        with open('label.txt', 'wb') as f: 
            print("about to write to label.txt")
            f.write(("apple4" + ';' + dateTime).encode())
        return ''
    else:
        label = ''
        time.sleep(0.1) # 0.03 works 0.02 not
        with open('label.txt', 'rb') as f: 
            firstline = f.readline().decode()
            label = firstline.split(';')[0]
            dateTime = firstline.split(';')[1]
            return render_template('result.html', label=label, dateTime=dateTime)


@app.route('/go/<dataURL>')
def pred(dataURL):
    """
    Render prediction result.
    """

    # decode base64  '._-' -> '+/='
    dataURL = dataURL.replace('.', '+')
    dataURL = dataURL.replace('_', '/')
    dataURL = dataURL.replace('-', '=')

    # get the base64 string
    image_b64_str = dataURL
    # convert string to bytes
    byte_data = base64.b64decode(image_b64_str)
    image_data = BytesIO(byte_data)
    # open Image with PIL
    img = Image.open(image_data)

    # save original image as png (for debugging)
    ts = time.time()
    #img.save('image' + str(ts) + '.png', 'PNG')

    # convert image to RGBA
    img = img.convert("RGBA")

    # preprocess the image for the model
    image_cropped = crop_image(img) # crop the image and resize to 28x28
    image_normalized = normalize_image(image_cropped) # normalize color after crop

    # convert image from RGBA to RGB
    img_rgb = convert_to_rgb(image_normalized)

    # convert image to numpy
    image_np = convert_to_np(img_rgb)

    # apply model and print prediction
    label, label_num, preds = get_prediction(model, image_np)
    print("This is a {}".format(label_num))

    # save classification results as a diagram
    view_classify(image_np, preds)

    # create plotly visualization
    graphs = [
        #plot with probabilities for each class of images
        {
            'data': [
                go.Bar(
                        x = preds.ravel().tolist(),
                        y = list(label_dict.values()),
                        orientation = 'h')
            ],

            'layout': {
                'title': 'Class Probabilities',
                'yaxis': {
                    'title': "Classes"
                },
                'xaxis': {
                    'title': "Probability",
                }
            }
        }]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)


    preds = preds.squeeze()
    label_1 = np.argsort(preds)[-1]
    label_2 = np.argsort(preds)[-2]
    label_3 = np.argsort(preds)[-3]

    label_list = ['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword']

    label_str_1 = label_list[label_1]
    label_str_2 = label_list[label_2]
    label_str_3 = label_list[label_3]

    # render the hook.html passing prediction resuls
    return render_template(
        'result.html',
        result = label_num, # predicted class label
        ids=ids, # plotly graph ids
        graphJSON=graphJSON, # json plotly graphs
        dataURL = dataURL, # image to display with result
        label_str_1 = label_str_1,
        label_str_2 = label_str_2,
        label_str_3 = label_str_3
    )

@app.route('/revenge')
def revenge():
    return render_template('revenge.html', modelName=main())

@app.route('/statistics')
def statistics():
	return render_template('statistics.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)