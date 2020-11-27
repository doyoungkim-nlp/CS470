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
try:
       from cStringIO import StringIO  # Py2 C accelerated version
except ImportError:
       try:
           from StringIO import StringIO  # Py2 fallback version
       except ImportError:
           from io import StringIO  # Py3 version

app = Flask(__name__)

@app.route('/')
def start():
	return render_template('start.html')

@app.route('/canvas')
def canvas():
	return render_template('canvas.html')

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
            f.write("apple2".encode())
        return ''
    else:
        label = ''
        time.sleep(0.1)
        with open('label.txt', 'rb') as f: 
            label = f.readline().decode()
            return render_template('result.html', label=label)

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