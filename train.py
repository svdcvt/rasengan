import torch
from torch import nn
from models import Generator, Discriminator
from losses import reconstruction_loss, regularization_term, disc_loss, gen_loss, segsnr

from scipy.io import wavfile
from pesq import pesq

import torchaudio
import matplotlib.pyplot as plt

import numpy as np
import resampy
import IPython
from tqdm import tqdm
import shutil

import sys
sys.path += ["../"]
from segan_pytorch.segan.datasets import se_dataset
from utils import *
from torch.utils.tensorboard import SummaryWriter


def main(args=None):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    # shutil.rmtree("./data")
    G = Generator()
    D = Discriminator()
    optimizerG = torch.optim.Adam(G.parameters(), lr=1e-4)
    optimizerD = torch.optim.RMSprop(D.parameters(), lr=1e-4)

    win = lambda x: torch.ones(x) if not args.hann else torch.hann_window(x)

    dataset = se_dataset.SEDataset(args.clean_dir, args.noisy_dir,
                                   0.95,
                                   cache_dir=args.cache_dir,
                                   slice_size=args.slice_size,
                                   max_samples=1000)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    G = Generator().to(device)
    D = Discriminator().to(device)
    optimizerG = torch.optim.Adam(G.parameters(), lr=1e-4)
    optimizerD = torch.optim.RMSprop(D.parameters(), lr=1e-4)


    torch.autograd.set_detect_anomaly(True)

    G.to(device)
    D.to(device)

    g_loss = []
    d_loss = []

    writer = SummaryWriter("./logs")
    train = False
    if train:
        for epoch in range(args.num_epochs):
            print("EPOCH", epoch)
            if epoch == (args.num_epochs // 3):
                optimizerD.param_groups[0]['lr'] = optimizerD.param_groups[0]['lr'] / 10
                optimizerG.param_groups[0]['lr'] = optimizerG.param_groups[0]['lr'] / 10
            
            for bidx, batch in tqdm(enumerate(dloader, 1), total=len(dloader)):
                uttname, clean, noisy, slice_idx = batch
                clean, noisy = clean.to(device), noisy.to(device)

                # Get real data
                if args.task == 'sr':
                    real_full = full_lps(clean).to(device)
                    lf = lowres_lps(clean, args.rate_div).to(device)
                else:
                    real_full = full_lps(clean).detach().clone().to(device)
                    lf = full_lps(noisy)[:,:129,:].detach().clone().to(device)

                # Update D
                if epoch >= (args.num_epochs // 3):
                    for p in D.parameters():
                        p.requires_grad = True
                    fake_hf = G(lf).to(device)
                    fake_full = torch.cat([lf, fake_hf], 1)
                    fake_logit, fake_prob = D(fake_full)
                    real_logit, real_prob = D(real_full)

                    optimizerD.zero_grad()
                    gan_loss_d = disc_loss(fake_prob, real_prob)
                    reg_d = regularization_term(fake_prob, real_prob, fake_logit, real_logit)
                    lossD = gan_loss_d + reg_d
                    lossD.backward()
                    writer.add_scalar("loss/D_loss", lossD)
                    writer.add_scalar("loss/D_loss_reg", reg_d)
                    writer.add_scalar("loss/D_loss_gan", gan_loss_d)
                    optimizerD.step()
                    d_loss.append(lossD.item())

                # Update G
                for p in D.parameters():
                    p.requires_grad = False

                fake_hf = G(lf)
                fake_full = torch.cat([lf, fake_hf], 1)
                fake_logit, fake_prob = D(fake_full)
                real_logit, real_prob = D(real_full)
                gan = None
                if epoch >= (args.num_epochs // 3):
                    gan = gen_loss(fake_prob)
                    rec_loss = reconstruction_loss(real_full, fake_full)
                    lossG = args.lambd * rec_loss - gan
                else:
                    rec_loss = reconstruction_loss(real_full, fake_full)
                    lossG = args.lambd * rec_loss
                writer.add_scalar("loss/G_loss", lossG)
                writer.add_scalar("rec_loss/rec_loss", rec_loss)
                lossG.backward()
                optimizerG.step()
                g_loss.append(lossG.item())
            with open("result.pth", "wb") as f:
                torch.save({
                    "g_state_dict": G.state_dict(),
                    "d_state_dict": D.state_dict(),
                }, f)
    else:
        with open("./result.pth", "rb") as f:
            G.load_state_dict(torch.load(f)["g_state_dict"])
        for bidx, batch in tqdm(enumerate(dloader, 1), total=len(dloader)):
            uttname, clean, noisy, slice_idx = batch
            clean, noisy = clean.to(device), noisy.to(device)
            real_full = full_lps(clean).to(device)
            lf = lowres_lps(clean, args.rate_div).to(device)
            fake_hf = G(lf)
            fake_full = torch.cat([lf, fake_hf], 1)
            break
        
        
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == "__main__":
    args = AttrDict(
        batch_size=256,
        slice_size=270,
        gamma=2,
        lambd=0.5,
        task='sr',
        rate_div=2,
        batch=64,
        hann=False,
        power=2.0,
        win_length=32,
        hop_length=8,
        T=32,
        num_epochs = 3,
        cache_dir='./data',
        clean_dir='/home/sorain/my_wham/train/clean',
        noisy_dir='/home/sorain/my_wham/train/noisy',
    )
    main(args)