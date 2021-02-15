import argparse
import time
import numpy as np
from tqdm import tqdm
import librosa

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.utils.data import DataLoader

from models.unet import UNet
from data_loader import WavDatasetForDereverb
from utils import create_folder

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

MODEL_PATH = '../pt/20210214/'
MODEL_NAME = 'model-22-0.022722.pt'


def train(model, criterion, train_loader, optimizer, epoch):
    model.train()
    criterion.train()

    train_loss = 0
    for batch_idx, samples in enumerate(train_loader):
        data, target = samples

        data = data.to(DEVICE)
        target = target.to(DEVICE)

        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Epoch: {:3d}\tBatch Index: {:2d}\tLoss: {:.6f}'.format(epoch, batch_idx, loss.item()))

    train_loss /= len(train_loader.dataset)

    return train_loss



def validate(model, criterion, val_loader, epoch):
    model.eval()
    criterion.eval()

    val_loss = 0
    with torch.no_grad():
        for samples in tqdm(val_loader):
            data, target = samples

            data = data.to(DEVICE)
            target = target.to(DEVICE)

            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    val_loss /= len(val_loader.dataset)
    #print('Epoch:{}\tVal Loss:{:.6f}'.format(epoch, val_loss))

    return val_loss



def evaluate(model, criterion, test_loader):
    model.eval()
    criterion.eval()

    test_loss = 0
    with torch.no_grad():
        for samples in tqdm(test_loader):
            data, target = samples

            data = data.to(DEVICE)
            target = target.to(DEVICE)

            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    #print('Test Loss:{:.6f}'.format(test_loss))

    return test_loss


def save_model(modelpath, model):
    torch.save(model.state_dict(), modelpath)

    print('model saved')




def load_model(modelpath, model):
    state = torch.load(modelpath, map_location=torch.device(DEVICE))
    model.load_state_dict(state)

    print('model loaded')


def main(args):
    # train
    if args.mode == 'train':
        train_dataset = WavDatasetForDereverb(mode='train', type=args.type)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.worker)
        #train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker)

        val_dataset = WavDatasetForDereverb(mode='val', type=args.type)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker)
        print(train_dataloader, val_dataloader)

        model = UNet()
        # set optimizer
        optimizer = AdamW(
            [param for param in model.parameters() if param.requires_grad], lr=args.lr)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

        # load model
        #modelpath = MODEL_PATH + MODEL_NAME
        #load_model(modelpath, model)
        '''
        if torch.cuda.device_count() > 1:
            print('multi gpu used!')
            model = nn.DataParallel(model)
        '''
        model = model.to(DEVICE)

        criterion = torch.nn.L1Loss()
        criterion = criterion.to(DEVICE)

        for epoch in range(args.start_epoch, args.epoch):
            # train set
            train_loss = train(model, criterion, train_dataloader, optimizer, epoch)
            # validate set
            val_loss = validate(model, criterion, val_dataloader, epoch)

            print('Epoch:{}\tTrain Loss:{:.6f}\tVal Loss:{:.6f}'.format(epoch, train_loss, val_loss))

            if epoch == args.start_epoch:
                create_folder(MODEL_PATH + args.type)
            modelpath = MODEL_PATH + args.type + '/model-{:d}-{:.6f}-{:.6f}.pt'.format(epoch, train_loss, val_loss)
            save_model(modelpath, model)

            # scheduler update
            scheduler.step()
    # evaluate
    if args.mode == 'test':
        test_dataset = WavDatasetForDereverb(mode='test', type=args.type)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker)
        print(test_dataloader)

        model = UNet()
        # set optimizer
        optimizer = AdamW(
            [param for param in model.parameters() if param.requires_grad], lr=args.lr)

        # load model
        modelpath = MODEL_PATH + MODEL_NAME
        load_model(modelpath, model)
        '''
        if torch.cuda.device_count() > 1:
            print('multi gpu used!')
            model = nn.DataParallel(model)
        '''
        model = model.to(DEVICE)

        criterion = torch.nn.L1Loss()
        criterion = criterion.to(DEVICE)

        # validate set
        test_loss = evaluate(model, criterion, test_dataloader)

        print('[Evaluation]\tTest Loss:{:.6f}'.format(test_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=16,
        type=int)
    parser.add_argument(
        '--epoch',
        help='the number of training iterations',
        default=10,
        type=int)
    parser.add_argument(
        '--start_epoch',
        help='the number of start epoch',
        default=0,
        type=int)
    parser.add_argument(
        '--seed',
        help='random seed',
        default=2021,
        type=int)
    parser.add_argument(
        '--lr',
        help='learning rate',
        default=1e-4,
        type=float)
    parser.add_argument(
        '--shuffle',
        help='True, or False',
        default=True,
        type=bool)
    parser.add_argument(
        '--mode',
        help='train or test',
        default='train',
        type=str)
    parser.add_argument(
        '--type',
        help='waveglow or melgan',
        default='waveglow',
        type=str)
    parser.add_argument(
        '--worker',
        help='the number of cpu workers',
        default=0,
        type=int)


    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)

    start_t = time.time()
    main(args)
    print('elapsed time:', time.time() - start_t)



