import argparse
import logging
from torch import optim
from tqdm import tqdm
from utils.Figure import draw_accuracy, draw_f1, draw_dice, draw_loss
from eval import eval_net
from unet import UNet
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import torch
from torch import nn
import pandas as pd
import math

dir_img = 'data/training/img/'
dir_mask = 'data/training/full_mask/'
dir_checkpoint = 'checkpoints/'

def fill_abnormal_data(data):
    df = pd.DataFrame(data)
    return df.fillna(0)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill(0.01)


def train_net(net1,
              device1,
              epochs,
              batch_size,
              lr,
              val_percent,
              save_cp=True,
              img_scale = 0.5,
              ):
    accmean = []
    dmean = []
    f1mean = []
    loss1= []
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    logging.info(f'Creating dataset with {len(dataset)} examples')

    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True)
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device1.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net1.parameters(), lr=lr, weight_decay=1e-8)
    criterion = nn.BCEWithLogitsLoss()

    # train loop
    for epoch in range(epochs):
        # define parameters
        net1.train()
        epoch_loss = 0
        mean_acc = 0
        mean_dice = 0
        mean_f1 = 0
        sum_dice = 0
        sum_acc = 0
        sum_f1 = 0
        count = 0

    # set progress bar, start iteration
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            # training
            for batch in train_loader:
                img = batch['image']
                true_mask = batch['mask']
                # to CUDA
                imgs = img.to(device=device1, dtype=torch.float32)
                true_masks = true_mask.to(device=device1, dtype=torch.float32)
                masks_pred = net1(imgs)
                loss = criterion(masks_pred, true_masks)

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1

                # testing
                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    dice, acc, f1, img, true_mask, masks_pred = eval_net(net1, val_loader, device1, n_val)
                    count = count+1

                    sum_dice += dice
                    mean_dice = sum_dice / count
                    sum_acc += acc
                    mean_acc = sum_acc / count
                    sum_f1 += f1
                    mean_f1 = sum_f1 / count

                    logging.info('Dice coeff: {}'.format(dice))
                    logging.info('Accuracy: {}'.format(acc))
                    logging.info('F1-score: {}'.format(f1))

        epoch_loss = epoch_loss/len(dataset)
        logging.info('loss:{}'.format(epoch_loss))
        while(math.isnan(epoch_loss)):
            epoch_loss = 0
        loss1 = np.append(loss1, epoch_loss)

        logging.info('Mean validation Dice Coeff: {}'.format(mean_dice))
        while(math.isnan(mean_dice)):
            mean_dice = 0
        dmean = np.append(dmean, mean_dice)

        logging.info('Mean accuracy: {}'.format(mean_acc))
        while(math.isnan(mean_acc)):
            mean_acc = 0
        accmean = np.append(accmean, mean_acc)


        logging.info('Mean F1-score: {}'.format(mean_f1))
        while(math.isnan(mean_f1)):
            mean_f1 = 0
        f1mean = np.append(f1mean, mean_f1)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    draw_loss(loss1, epochs)
    draw_dice(dmean, epochs)
    draw_accuracy(accmean, epochs)
    draw_f1(f1mean, epochs)

if __name__ == '__main__':
    # log basic configurations
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')

    net = UNet(1,1)
    net.apply(init_weights)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    net.to(device=device)


    train_net(net1=net,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              device1=device,
              img_scale=args.scale,
              val_percent=args.val / 100)
