import warnings
import os
import shutil

import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sketch_rnn.hparams import hparam_parser
from sketch_rnn.utils import AverageMeter, ModelCheckpoint
from sketch_rnn.dataset import SketchRNNDataset, load_strokes, collate_drawings
from sketch_rnn.model import SketchRNN, model_step
from torch.utils.tensorboard import SummaryWriter



def train_epoch(model, data_loader, optimizer, scheduler, device,
                grad_clip=None):
    model.train()
    loss_meter = AverageMeter()
    with tqdm(total=len(data_loader.dataset)) as progress_bar:
        for data, lengths in data_loader:
            data = data.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            # training step
            optimizer.zero_grad()
            loss = model_step(model, data, lengths)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            # update loss meter and progbar
            loss_meter.update(loss.item(), data.size(0))
            progress_bar.set_postfix(loss=loss_meter.avg)
            progress_bar.update(data.size(0))

    return loss_meter.avg


@torch.no_grad()
def eval_epoch(model, data_loader, device):
    model.eval()
    loss_meter = AverageMeter()
    for data, lengths in data_loader:
        data = data.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        loss = model_step(model, data, lengths)
        loss_meter.update(loss.item(), data.size(0))
    return loss_meter.avg


def train_sketch_rnn(args):
    writer = SummaryWriter(args.save_dir)

    torch.manual_seed(884)
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    saver = ModelCheckpoint(args.save_dir) if (args.save_dir is not None) else None

    # initialize train and val datasets
    train_strokes, valid_strokes, test_strokes = load_strokes(args.data_dir, args)
    train_data = SketchRNNDataset(
        train_strokes,
        max_len=args.max_seq_len,
        random_scale_factor=args.random_scale_factor,
        augment_stroke_prob=args.augment_stroke_prob
    )
    val_data = SketchRNNDataset(
        valid_strokes,
        max_len=args.max_seq_len,
        scale_factor=train_data.scale_factor,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0
    )

    # initialize data loaders
    collate_fn = lambda x : collate_drawings(x, args.max_seq_len)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=use_gpu,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=use_gpu,
        num_workers=args.num_workers
    )

    # model & optimizer
    model = SketchRNN(args).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    if os.path.exists(args.save_image_dir):
        print("save_image is exists")
        shutil.rmtree(args.save_image_dir)
    os.mkdir(args.save_image_dir)

    #print("model state dict keys : ", list(model.state_dict().keys()))
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_clip)
        val_loss = eval_epoch(model, val_loader, device)
        print('Epoch %0.3i, Train Loss: %0.4f, Valid Loss: %0.4f' %
              (epoch+1, train_loss, val_loss))

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        if saver is not None:
            print("saver is not none")
            saver(epoch, model, optimizer, train_loss, val_loss)
            #model.conditional_generation(epoch, args.max_seq_len, args.save_image_dir)
        
        if epoch == 0 or epoch%5 == 4 : 
            print("conditional generation")
            model.conditional_generation(epoch, args.max_seq_len, args.save_image_dir)

        time.sleep(0.5) # avoids progress bar issue
    #model.load("/content/drive/MyDrive/CS470/Final/sketch_generation/Sketch-RNN-master/Sketch-RNN-master/model1_save/model.pt")
    #model.conditional_generation(0, args.max_seq_len)


if __name__ == '__main__':
    hp_parser = hparam_parser()
    parser = argparse.ArgumentParser(parents=[hp_parser])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_image_dir', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    train_sketch_rnn(args)

