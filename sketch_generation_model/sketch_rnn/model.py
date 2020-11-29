import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.nn.utils.rnn as rnn_utils

import numpy as np
import matplotlib.pyplot as plt
import PIL
import os

from .rnn import _cell_types, LSTMLayer, init_orthogonal_
from .param_layer import ParameterLayer
from .objective import KLLoss, DrawingLoss
from .utils import sample_gmm

__all__ = ['SketchRNN', 'model_step', 'sample_conditional',
           'sample_unconditional']


class Encoder(nn.Module):
    def __init__(self, hidden_size, z_size):
        super().__init__()
        self.rnn = nn.LSTM(5, hidden_size, bidirectional=True, batch_first=True)
        self.output = nn.Linear(2*hidden_size, 2*z_size)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(2):
            weight_ih, weight_hh, bias_ih, bias_hh = self.rnn.all_weights[i]
            nn.init.xavier_uniform_(weight_ih)
            init_orthogonal_(weight_hh, hsize=self.hidden_size)
            nn.init.zeros_(bias_ih)
            nn.init.zeros_(bias_hh)
        nn.init.normal_(self.output.weight, 0., 0.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = rnn_utils.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (x, _) = self.rnn(x) # [2,batch,hid]
        x = x.permute(1,0,2).flatten(1).contiguous() # [batch,2*hid]
        z_mean, z_logvar = self.output(x).chunk(2, 1)
        z = z_mean + torch.exp(0.5*z_logvar) * torch.randn_like(z_logvar)
        return z, z_mean, z_logvar


class SketchRNN(nn.Module):
    def __init__(self, hps):
        super().__init__()
        # check inputs
        assert hps.enc_model in ['lstm', 'layer_norm', 'hyper']
        assert hps.dec_model in ['lstm', 'layer_norm', 'hyper']
        if hps.enc_model in ['layer_norm', 'hyper']:
            raise NotImplementedError('LayerNormLSTM and HyperLSTM not yet '
                                      'implemented for bi-directional encoder.')
        # encoder modules
        self.encoder = Encoder(hps.enc_rnn_size, hps.z_size)
        # decoder modules
        cell_fn = _cell_types[hps.dec_model]
        self.cell = cell_fn(5+hps.z_size, hps.dec_rnn_size, r_dropout=hps.r_dropout)
        self.decoder = torch.jit.script(LSTMLayer(self.cell, batch_first=True))
        self.init = nn.Linear(hps.z_size, self.cell.state_size)
        self.param_layer = ParameterLayer(hps.dec_rnn_size, k=hps.num_mixture)
        # loss modules
        self.loss_kl = KLLoss(
            hps.kl_weight,
            eta_min=hps.kl_weight_start,
            R=hps.kl_decay_rate,
            kl_min=hps.kl_tolerance)
        self.loss_draw = DrawingLoss(reg_covar=hps.reg_covar)
        self.max_len = hps.max_seq_len
        self.z_size = hps.z_size
        self.reset_parameters()

    def reset_parameters(self):
        reset = lambda m: (hasattr(m, 'reset_parameters') and
                           not isinstance(m, torch.jit.ScriptModule))
        for m in filter(reset, self.children()):
            m.reset_parameters()
        nn.init.normal_(self.init.weight, 0., 0.001)
        nn.init.zeros_(self.init.bias)

    def _forward(self, enc_inputs, dec_inputs, enc_lengths=None):
        # encoder forward
        z, z_mean, z_logvar = self.encoder(enc_inputs, enc_lengths)

        # initialize decoder state
        state = torch.tanh(self.init(z)).chunk(2, dim=-1)

        # append z to decoder inputs
        z_rep = z[:,None].expand(-1,self.max_len,-1)
        dec_inputs = torch.cat((dec_inputs, z_rep), dim=-1)

        # decoder forward
        output, state = self.decoder(dec_inputs, state)

        # mixlayer outputs
        params = self.param_layer(output) 

        self.params = params
        self.state = state
        # => (pi, (mu_x, mu_y),(sigma_x, sigma_y),rho_xy, q)
        #print("params[1] shape : ", params[1].shape)
        #print("state type : ", type(state))

        return params, z_mean, z_logvar, state

    def forward(self, data, lengths=None):
        enc_inputs = data[:,1:self.max_len+1,:] # remove sos
        dec_inputs = data[:,:self.max_len,:] # keep sos
        return self._forward(enc_inputs, dec_inputs, lengths)

  #------------------------------------------------------------------------#
    def conditional_generation(self, epoch, max_len, save_image_dir):
        if torch.cuda.is_available():
            #sos = Variable(torch.Tensor([0,0,1,0,0]).view(1,1,-1).cuda())
            sos = torch.Tensor([0,0,1,0,0]).view(1,1,-1).cuda()
        else:
            #sos = Variable(torch.Tensor([0,0,1,0,0]).view(1,1,-1))
            sos = torch.Tensor([0,0,1,0,0]).view(1,1,-1)
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        for i in range(max_len):
            # decode:

            #self.params, self.hidden, self.cell, self.state = self._forward(enc_inputs, dec_inputs, lengths)
            #params, _, _, state = self._forward(enc_inputs, dec_inputs, lengths)
            self.pi, self.means, self.scales, self.rho_xy, self.q = self.params
            # means = (mu_x, mu_y) / scales = (sigma_x, sigma_y)
            hidden_cell = self.state

            pi = self.pi
            (self.mu_x, self.mu_y) = torch.split(self.means, 1, dim=3)
            (self.sigma_x, self.sigma_y) = torch.split(self.scales, 1, dim=3)
            mu_x = self.mu_x
            mu_y = self.mu_y
            sigma_x = self.sigma_x
            sigma_y = self.sigma_y
            rho_xy = self.rho_xy
            q = self.q
            
            pi = pi.transpose(0,1).squeeze().contiguous().view(1,-1,20)
            mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(1,-1,20) #20 : args.num_mixture
            mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(1,-1,20)
            sigma_x = sigma_x.transpose(0,1).squeeze().contiguous().view(1,-1,20)
            sigma_y = sigma_y.transpose(0,1).squeeze().contiguous().view(1,-1,20)
            rho_xy = rho_xy.transpose(0,1).squeeze().contiguous().view(1,-1,20)
            


            # sample from parameters:
            #s, dx, dy, pen_down, eos = sample_next_state(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)
            s, dx, dy, pen_down, eos = self.sample_next_state()
            #------
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            """
            if eos:
                print(i)
                #break
                continue
            """

        #print("seq x length : ", seq_x)
        #print("seq_y length : ", seq_y)
        #print("seq_z length : ", seq_z)
        # visualize result:
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        z_sample = np.array(seq_z)
        sequence = np.stack([x_sample,y_sample,z_sample]).T
        #print("sequence : ", sequence)
        make_image(sequence, epoch, save_image_dir)
    

    def sample_next_state(self):
        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf)/0.5
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0,0,:].cpu().numpy()
        #pi = adjust_temp(pi)
        pi = np.nan_to_num(pi, copy=False)
        pi /= pi.sum()
        pi_idx = np.random.choice(20, p=pi)

        # get pen state:
        q = self.q.data[0,0,:].cpu().numpy()
        #q = adjust_temp(q)
        q = np.nan_to_num(q, copy=False)
        q = q / q.sum()
        q_idx = np.random.choice(3, p=q)
        # get mixture params:
        mu_x = self.mu_x.data[0,0,pi_idx]
        mu_y = self.mu_y.data[0,0,pi_idx]
        sigma_x = self.sigma_x.data[0,0,pi_idx]
        sigma_y = self.sigma_y.data[0,0,pi_idx]
        rho_xy = self.rho_xy.data[0,0,pi_idx]
        x,y = sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy,greedy=False)
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx+2] = 1
        return torch.Tensor(next_state).view(1,1,-1),x,y,q_idx==1,q_idx==2
        """
        if torch.cuda.is_available():
            return torch.Tensor(next_state.cuda()).view(1,1,-1),x,y,q_idx==1,q_idx==2
        else:
            return torch.Tensor(next_state).view(1,1,-1),x,y,q_idx==1,q_idx==2
        """

def sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy, greedy=False):
    # inputs must be floats
    greedy = True
    mu_x = float(mu_x.cpu().numpy()[0])
    mu_y = float(mu_y.cpu().numpy()[0])
    sigma_x = float(sigma_x.cpu().numpy()[0])
    sigma_y = float(sigma_y.cpu().numpy()[0])
    if greedy:
        return mu_x,mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(0.4)
    sigma_y *= np.sqrt(0.4)
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],[rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    #print("mean[0][0] : ", mean[0][0], " / cov[0][0] : ", cov[0][0])
    
    ## Fix it!! (TypeError) => fixed it (change the tensor -> numpy array -> float)
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

# ---- Image Making ---- #
def make_image(sequence, epoch, save_image_dir, name='_output_'):
    """plot drawing with separated strokes"""
    #print("sequence : ", sequence)
    strokes = np.split(sequence, np.where(sequence[:,2]>0)[0]+1)
    #print("strokes : ", strokes)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:,0],-s[:,1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                 canvas.tostring_rgb())
    name = str(epoch)+name+'.jpg'
    file_path = save_image_dir
    pil_image.save(os.path.join(file_path, name),"JPEG")
    plt.close("all")


  #----------------------------------------------------------------------------------------------------#

# ---- model step code (for train/eval) ----

def model_step(model, data, lengths=None):
    # model forward
    params, z_mean, z_logvar, states = model(data, lengths)

    # prepare targets
    targets = data[:,1:model.max_len+1,:]
    x, v_onehot = targets.split([2,3], -1)
    assert torch.all(v_onehot.sum(-1) == 1)
    v = v_onehot.argmax(-1)

    # compute losses
    loss_kl = model.loss_kl(z_mean, z_logvar)
    loss_draw = model.loss_draw(x, v, params)
    loss = loss_kl + loss_draw

    return loss



# ---- Sampling code -----

@torch.no_grad()
def sample_from_z(model, z, T=1):
    # initialize decoder state
    state = torch.tanh(model.init(z)).chunk(2, dim=-1)

    # decode target sequences w/ attention
    x = torch.zeros(1, 2, dtype=torch.float32, device=z.device)
    v = torch.zeros(1, dtype=torch.long, device=z.device)
    x_samp, v_samp = [x], [v]
    for t in range(model.max_len):
        # compute parameters for next step
        v = F.one_hot(v,3).float()
        dec_inputs = torch.cat((x,v,z), -1)
        output, state = model.decoder.cell(dec_inputs, state)
        mix_logp, means, scales, corrs, v_logp = model.param_layer(output, T=T)
        # sample next step
        v = D.Categorical(v_logp.exp()).sample() # [1]
        if v.item() == 2:
            break
        x = sample_gmm(mix_logp, means, scales, corrs) # [1,2]
        # append sample and continue
        x_samp.append(x)
        v_samp.append(v)
    return torch.cat(x_samp), torch.cat(v_samp)

@torch.no_grad()
def sample_unconditional(model, T=1, z_scale=1, device=torch.device('cpu')):
    model.eval().to(device)
    z = z_scale * torch.randn(1, model.z_size, dtype=torch.float, device=device)
    return sample_from_z(model, z, T=T)

@torch.no_grad()
def sample_conditional(model, data, lengths, T=1, device=torch.device('cpu')):
    model.eval().to(device)
    data, lengths = check_sample_inputs(data, lengths, device)
    enc_inputs = data[:,1:,:]
    z, _, _ = model.encoder(enc_inputs, lengths)
    return sample_from_z(model, z, T=T)

def check_sample_inputs(data, lengths, device):
    assert isinstance(data, torch.Tensor)
    assert data.dim() == 2
    assert isinstance(lengths, torch.Tensor) or isinstance(lengths, numbers.Integral)
    data = data.unsqueeze(0)
    if torch.is_tensor(lengths):
        assert lengths.dim() == 0
        lengths = lengths.unsqueeze(0)
    else:
        lengths = torch.tensor([lengths])
    return data.to(device), lengths.to(device)
