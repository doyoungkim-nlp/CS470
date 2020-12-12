# Sketch recognition

This is the directory that contains codes for sketch recognition which is the first function of Sketchmind.   
We consider sketch recognition is similar with MNIST classification, so we tried various networks which are usually used or have good performance for MNIST classification.   

## Directory structure
This structure only contains meaningful codes or directories
<pre><code>
sketch_recognition_model
|- app
    |- run.py
|- checkpoints
    |- checkpoint_baseline.pth
    |- checkpoint_simpleNet.pth
|- models
    |- simple_conv_nn.py    # for simple CNN
    |- SimpleNet.py         # for SimpleNet
    |- VGG8b.py             # for VGG8b
|- build_model.py
|- fitting_function.py
|- image_utils.py

</code></pre>

We used 4 networks : Feed-Forward Neural Network(fully connected), simple CNN, VGG8b, SimpleNet   
These networks are in `models` directory (each network is separated to a single python file), and we call them when build the model.   

In `app` directory, there are codes for sketch recognition demo.   
In `checkpoints` directory, there are pre-trained models that can be used in demo.   

`build_model.py` contains codes for training, evaluation, prediction and visualize the predicted result.   
`fitting_function.py` contains functions that fit the model.   
`image_utils.py` contains functions to add flipped and rotated images to the original training set.


## Running the codes
Before you run the codes, you should build the conda environment which builds with `environment.yml` in root directory(CS470).

### 1. Training the model (model build)
<pre><code>
python build_model.py [arguments]

[Example]

python build_model.py --architecture simpleNet --epochs 20

</code></pre>

> #### Arguments
><pre><code>
> --save_dir : Directory to save model checkpoint
> --learning_rate : Value of learning rate (type : float)
> --epochs : Number of epochs (type : int)
> --weight_decay : Value of weight decay (type : float)
> --dropout : Value of dropout (type : float)
> --architecture : Model architecture (choices : ['nn', 'conv', 'simpleNet','vgg8b'])
> --mini_batches : Number of mini-batches (type : int)
> --add_data : Add flipped and rotated images to the original training set
> --optimizer : Optimizer for fitting the model (choices : ['SGD', 'Adam'])
> 
></code></pre>


### 2. Start sketch recognition demo   
In sketch_recognition_directory, type the commands below
<pre><code>
cd app
python run.py

</code></pre>
Then, you can run the demo for sketch recognition.