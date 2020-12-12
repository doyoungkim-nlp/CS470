# Sketch generation demo
This demo code is based on magenta github(https://github.com/magenta/magenta/tree/master/magenta/models/sketch_rnn).   

Because of lack of computational power, we used pre-trained model provided by magenta. And we used their demo code for our Sketchmind demo, but this code uses python2 and low version of tensorflow, so it is not fitted to our environment or Google Colaboratory environment.   
So we modified and simplified this code to run our demo.   
We modified this code to use in python3 and latest version (or high version) of tensorflow. And simplified this code to use without installing magenta library.

## Code structure
This structure only contains meaningful codes or directories
<pre><code>
|- pretrained_model
|- sketch_rnn
|- functions.py

</code></pre>

This directory is for running `app.py` in root directory(CS470).   
`functions.py` contains necessary functions for run `app.py`.   
In `pretrained_model` directory, there are pre-trained model provided from magenta(original repository).   
In `sketch_rnn` directory, there are python codes that used in functions in  demo(`app.py`) and `functions.py`.

