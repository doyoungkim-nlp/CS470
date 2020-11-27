import os
from flask import Flask, render_template, request, jsonify
from flask import render_template
from models import db

from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

import numpy as np
from PIL import Image
import base64
from io import BytesIO
import cv2
import time

import re
# import Plotly
import plotly
#import plotly.plotly as py
import chart_studio.plotly as py
import plotly.graph_objs as go

# import Flask
from flask import Flask
from flask import render_template, request

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
try:
       from cStringIO import StringIO  # Py2 C accelerated version
except ImportError:
       try:
           from StringIO import StringIO  # Py2 fallback version
       except ImportError:
           from io import StringIO  # Py3 version



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
from flask import Flask
from flask import render_template, request

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
    plt.savefig('history/prediction' + str(ts) + '.png')

app = Flask(__name__)

# load model
model, input_size, output_size = load_model()
model.eval() # set to evaluation

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

    # render the hook.html passing prediction resuls
    return render_template(
        'result.html',
        result = label_num, # predicted class label
        ids=ids, # plotly graph ids
        graphJSON=graphJSON, # json plotly graphs
        dataURL = dataURL # image to display with result
    )

@app.route('/revenge')
def revenge():
	return render_template('revenge.html')

@app.route('/statistics')
def statistics():
	return render_template('statistics.html')

if __name__ == '__main__':
    basedir = os.path.abspath(os.path.dirname(__file__))
    dbfile = os.path.join(basedir, 'db.sqlite')
    
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + dbfile 
    app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True 
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
    
    db.init_app(app) # app 설정값들을 초기화한다.
    db.app = app # models.py에서 db를 가져와서 db.app에 app을 명시적으로 넣어준다.
    db.create_all()
    
    app.run(host='127.0.0.1', port=5000, debug=True)