# Sketch-RNN (with PyTorch)


## Development notes
The provided code runs, however, there are a few "to-do" items to correctly match the official implementation:

1. __Encoder LSTM__: This model have not yet implemented recurrent dropout and layer normalization for the bi-directional encoder LSTM.
2. __Input/output dropout__: The magenta library offers optional input/output dropout for the decoder LSTM, although they were not used in the Sketch-RNN paper. This model have not implemented either.

## Example usage

We've provided a script `train_sketch_rnn.py` showing how to train the model. 
You must provide an argument `--data_dir` specifying the root path where your `.npz` dataset files are located. 
To checkpoint the model and losses during training, specify a save directory (to be created) with `--save_dir`.
To save the images that are generated for each epochs, specify a save directory (to be created) with `--save_image_dir`.
You can specify the number of epochs using `--num_epochs`

```
python train_sketch_rnn.py --data_dir=/path/to/data/root --save_dir=model1_save --save_image_dir=imgage1_save --num_epochs=100
```

Optionally, Sketch-RNN hyperparameters can also be specified via command line arguments. 
The parameter names and default values are as follows:

```python
# architecture params
parser.add_argument('--max_seq_len', type=int, default=250) # will be updated based on dataset
parser.add_argument('--enc_model', type=str, default='lstm')
parser.add_argument('--dec_model', type=str, default='layer_norm')
parser.add_argument('--enc_rnn_size', type=int, default=256)
parser.add_argument('--dec_rnn_size', type=int, default=512)
parser.add_argument('--z_size', type=int, default=128)
parser.add_argument('--num_mixture', type=int, default=20)
parser.add_argument('--r_dropout', type=float, default=0.1)
# loss params
parser.add_argument('--kl_weight', type=float, default=0.5)
parser.add_argument('--kl_weight_start', type=float, default=0.01) # eta_min
parser.add_argument('--kl_tolerance', type=float, default=0.2) # kl_min
parser.add_argument('--kl_decay_rate', type=float, default=0.99995) # R
parser.add_argument('--reg_covar', type=float, default=1e-6) # covariance shrinkage
# training params
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.9999)
parser.add_argument('--min_lr', type=float, default=0.00001) # Unused at the moment
parser.add_argument('--grad_clip', type=float, default=1.0)
# dataset & data augmentation params
parser.add_argument('--data_set', type=str, default='cat.npz')
parser.add_argument('--random_scale_factor', type=float, default=0.15)
parser.add_argument('--augment_stroke_prob', type=float, default=0.10)
```
