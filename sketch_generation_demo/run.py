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

from functions import draw_strokes, make_grid_svg, load_env_compatible, load_model_compatible, encode, decode


def main() : 
    models_root_dir = './pretrained_model'
    model_list = ["aaron_sheep/layer_norm", "catbus/lstm", "elephantpig/lstm", "flamingo/lstm", "owl/lstm"]
    model_name = random.choice(model_list)
    model_dir = os.path.join(models_root_dir, model_name)
    base_model_dir = os.path.join(models_root_dir, "aaron_sheep/layer_norm")
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
    renderPM.drawToFile(drawing, "./tmp/png/test.png", fmt="PNG")



if __name__ == '__main__':
    main()

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

from functions import draw_strokes, make_grid_svg, load_env_compatible, load_model_compatible, encode, decode


def main() : 
    models_root_dir = './pretrained_model'
    model_list = ["aaron_sheep/layer_norm", "catbus/lstm", "elephantpig/lstm", "flamingo/lstm", "owl/lstm"]
    model_name = random.choice(model_list)
    model_dir = os.path.join(models_root_dir, model_name)
    base_model_dir = os.path.join(models_root_dir, "aaron_sheep/layer_norm")
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
    renderPM.drawToFile(drawing, "./tmp/png/test.png", fmt="PNG")



if __name__ == '__main__':
    main()