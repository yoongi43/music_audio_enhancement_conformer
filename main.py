import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange

from tqdm import tqdm
import soundfile as sf
from datetime import datetime
from glob import glob
import os; opj=os.path.join
import numpy as np
import argparse

from model.models import TFCModel

""" """
from data.dataset import MUSDB18Dataset, ReverbDataset, NoiseDataset
from data.dataset import MedleyDBSolosDataset

import importlib

from utils import metrics
from utils import utils
from utils import augmentation
from configs.exp1 import Configs
import wandb
# from sweetdebug import sweetdebug ; sweetdebug()

EPS = torch.finfo(float).eps

def parse_argas():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='v1')
    args = parser.parse_args()
    return args

class Solver:
    def __init__(self, args):
        cfg = args.configs
        self.cfg = importlib.import_module(f'configs.{cfg}')
        self.cfg = Configs()
        cfg = self.cfg
        
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
        
        """ Dataset """
        if cfg.dataset == 'solo':
            self.MusicDataset = MedleyDBSolosDataset
            self.dataset_path = cfg.dataset_medleydbsolos_path
            self.instruments = ['clarinet', 'female singer', 'flute', 'piano', 'tenor saxophone', 'trumpet', 'violin']
        elif cfg.dataset == 'musdb':
            self.MusicDataset = MUSDB18Dataset
            self.dataset_path = cfg.dataset_musdb_path
            self.instruments = None
        else:
            raise NotImplementedError(f'Not implemented dataset: {cfg.dataset}')
        """ Model """
        
        self.net = TFCModel(num_channel=64,
                          num_features=cfg.n_fft//2+1,
                          n_conformers=cfg.n_conformers,
                          attn_mask=None)
        
        wandb.init(project='music_enhancement_conformers')
        
        # torch.save(ckpt, opj(cfg.save_dir, 'ckpt', str(epoch).zfill(4)+'.pt'))
        # ckpt = torch.load('./results/ckpt/0012.pt')
        if cfg.resume:
            assert cfg.ckpt_path
            ckpt = torch.load(cfg.ckpt_path)
            self.net.load_state_dict({k[7:]: v for k, v in ckpt['net'].items()})
            self.start_epoch = int(cfg.ckpt_path.split('/')[-1].split('.')[0])
            print(f"ckpt loaded from {cfg.ckpt_path}")
        else:
            self.start_epoch = 0
        
        # ckpt = torch.load('/home/yoongi43/music_enhancement_conformers/ckpt/2022_10_14_18_09_20_CPq_solo/ckpt/0050.pt')
        # self.net.load_state_dict(ckpt['net'])
        ## ddp load
        
        """ Save dir """
        self.save_dir = cfg.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        """ Loaders """
        self.loaders = self.init_dataloaders()
    
    def solve(self, net, noisy, device):
        noisy = noisy.float().to(device)
        pred = net(noisy)
        return pred
    
    def train(self):
        cfg = self.cfg
        
        os.makedirs(opj(cfg.save_dir, 'ckpt'), exist_ok=True)

        """ Data """
        loaders = self.loaders
        train_music_loader, train_rir_loader, train_noise_loader = loaders['training']
        valid_music_loader, valid_rir_loader, valid_noise_loader = loaders['validation']
        
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        net = nn.DataParallel(self.net)
        net = net.to(device)
        
        """ Metrics"""
        loss_fwsnr = metrics.fwssnr
        loss_mrs = metrics.multi_resolution_spectrogram
        loss_spec_l1 = metrics.spectrogram_l1
        
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
        
        """ Augmentations """
        eq_model = augmentation.MicrophoneEQ(rate=cfg.sr).to(device)
        low_cut_filter = augmentation.LowCut(35, rate=cfg.sr).to(device)
        
        for epoch in range(self.start_epoch, cfg.total_epochs):            
            """ Train Loop """
            wandb.log({"epoch": epoch})
            net.train()
            scaler = torch.cuda.amp.GradScaler()
            pbar = tqdm(train_music_loader, desc=f'train loop: {epoch}')
            for idx, x_waveform in enumerate(pbar):
                # if idx > 1: break
                """ Augment sample """
                # batch_size = x_waveform.shape[0]
                x_waveform, noise, rir = x_waveform.to(device), next(train_noise_loader).to(device), next(train_rir_loader).to(device)
                x_waveform, x_aug_waveform = augmentation.augment(
                    x_waveform, rir=rir, noise=noise, eq_model=eq_model, low_cut_model=low_cut_filter, rate=cfg.sr, normalize=True)
                x_waveform, x_aug_waveform = x_waveform[:,:,:(x_waveform.shape[2]//256)*256], x_aug_waveform[:,:,:(x_aug_waveform.shape[2]//256)*256]
                
                
                
                clean = rearrange(x_waveform, 'b 1 t -> b t')
                noisy = rearrange(x_aug_waveform, 'b 1 t -> b t')
                
                optimizer.zero_grad()
                
                """ Spectrograms """
                noisy_spec = torch.stft(noisy, cfg.n_fft, cfg.hop_len, window=torch.hamming_window(cfg.n_fft).to(device), onesided=True)
                clean_spec = torch.stft(clean, cfg.n_fft, cfg.hop_len, window=torch.hamming_window(cfg.n_fft).to(device), onesided=True)
                
                noisy_spec = utils.power_compress(noisy_spec)  # b f t c -> b c f t
                noisy_spec = rearrange(noisy_spec, 'b c f t -> b c t f')
                clean_spec = utils.power_compress(clean_spec)
                clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
                clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

                """ Estimate """
                # pred_dict = net(noisy_spec) # out: b c t f
                with torch.cuda.amp.autocast():
                    pred_dict = self.solve(net, noisy=noisy_spec, device=device) # output: b c t f
                    est_mag, est_real, est_imag, est_mask, est_correction = pred_dict['mag'], pred_dict['real'], pred_dict['imag'], pred_dict['mask'], pred_dict['correction']
                    est_real, est_imag = rearrange(est_real, 'b c t f -> b c f t'), rearrange(est_imag, 'b c t f -> b c f t')
                    est_mag = rearrange(est_mag, 'b c t f -> b c f t')
                    est_mask, est_correction = rearrange(est_mask, 'b 1 t f -> b f t'), rearrange(est_correction, 'b c t f -> b c f t')
                    clean_mag = torch.sqrt(clean_real**2 + clean_imag**2 + EPS)  # b c f t
                    
                    loss_mag = F.mse_loss(est_mag, clean_mag)
                    loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)
                    
                    est_spec_uncompress = utils.power_uncompress(est_real, est_imag).squeeze(1)
                    est_audio = torch.istft(est_spec_uncompress, cfg.n_fft, cfg.hop_len,
                                            window=torch.hamming_window(cfg.n_fft).to(device), onesided=True)                    

                    time_loss = torch.mean(torch.abs(est_audio - clean))

                    # losses: loss_ri, loss_mag, time_loss

                    loss = cfg.loss_weights[0] * loss_ri + cfg.loss_weights[1] * loss_mag + cfg.loss_weights[2] * time_loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # loss.backward()
                
                optimizer.step()
                
                """ METRICS """
                est_audio = est_audio.detach().cpu()
                clean = clean.detach().cpu()

                
                clean, est_audio = rearrange(clean, 'b t -> b 1 t'), rearrange(est_audio, 'b t -> b 1 t')
                # fwsnr_train = loss_fwsnr(clean=clean, estimated=est_audio, batch_size=cfg.batch_size)
                # mrs_train = loss_mrs(clean=clean, estimated=est_audio, batch_size=cfg.batch_size)
                # spec_l1_train = loss_spec_l1(clean=clean, estimated=est_audio, batch_size=cfg.batch_size)
                # # pbar.set_description('fwnsr: {:.2f}'.format(fwsnr_train))
                # pbar.set_description(f'fwsnr: {fwsnr_train:.2f}|mrs: {mrs_train:.2f}|spec_l1: {spec_l1_train:.2f}')
                pbar.set_description(f'epoch: {epoch}|loss_ri: {loss_ri.item():.2f}|loss_mag: {loss_mag.item():.2f}|time_loss: {time_loss.item():.2f}')
                wandb.log({"loss_ri": loss_ri.item(), "loss_mag": loss_mag.item(), "time_loss": time_loss.item()})
                
                if epoch % cfg.ckpt_per == 0:
                    # ckpt = dict(net=net.state_dict(), opt=optimizer.state_dict())
                    ckpt = dict(net=net.state_dict())
                    torch.save(ckpt, opj(cfg.save_dir, 'ckpt', str(epoch).zfill(4)+'.pt'))
        
            """ Valid Loop """      
            if epoch % cfg.valid_per == 0:
                with torch.no_grad():
                    net.eval()
                    pbar = tqdm(valid_music_loader, desc=f'valid loop: {epoch}')
                    for idx, x_waveform in enumerate(pbar):
                        x_waveform, noise, rir = x_waveform.to(device), next(valid_noise_loader).to(device), next(valid_rir_loader).to(device)                       
                        x_waveform, x_aug_waveform = augmentation.augment(
                            x_waveform, rir=rir, noise=noise, eq_model=eq_model, low_cut_model=low_cut_filter, rate=cfg.sr, normalize=True)
                        x_waveform, x_aug_waveform = x_waveform[:,:,:(x_waveform.shape[2]//256)*256], x_aug_waveform[:,:,:(x_aug_waveform.shape[2]//256)*256]
                        
                        x_waveform, noise, rir = rearrange(x_waveform, 'b 1 t -> b t'), rearrange(noise, 'b 1 t -> b t'), rearrange(rir, 'b 1 t -> b t')
                        x_aug_waveform = rearrange(x_aug_waveform, 'b 1 t -> b t')
                        
                        clean = x_waveform
                        noisy = x_aug_waveform
                        
                        """ Spectrograms """
                        noisy_spec = torch.stft(noisy, cfg.n_fft, cfg.hop_len, window=torch.hamming_window(cfg.n_fft).to(device), onesided=True)
                        clean_spec = torch.stft(clean, cfg.n_fft, cfg.hop_len, window=torch.hamming_window(cfg.n_fft).to(device), onesided=True)
                        
                        noisy_spec = utils.power_compress(noisy_spec)  # b f t c -> b c f t
                        noisy_spec = rearrange(noisy_spec, 'b c f t -> b c t f')
                        clean_spec = utils.power_compress(clean_spec)
                        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
                        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
                        
                        """ Estimate """
                        # pred_dict = net(noisy_spec) # out: b c t f
                        pred_dict = self.solve(net, noisy=noisy_spec, device=device) # output: b c t f
                        est_mag, est_real, est_imag, est_mask, est_correction = pred_dict['mag'], pred_dict['real'], pred_dict['imag'], pred_dict['mask'], pred_dict['correction']
                        est_real, est_imag = rearrange(est_real, 'b c t f -> b c f t'), rearrange(est_imag, 'b c t f -> b c f t')
                        est_mag = rearrange(est_mag, 'b c t f -> b c f t')
                        est_mask, est_correction = rearrange(est_mask, 'b 1 t f -> b f t'), rearrange(est_correction, 'b c t f -> b c f t')
                        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2+EPS)  # b c f t
                        
                        loss_mag = F.mse_loss(est_mag, clean_mag)
                        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)
                        
                        est_spec_uncompress = utils.power_uncompress(est_real, est_imag).squeeze(1)
                        est_audio = torch.istft(est_spec_uncompress, cfg.n_fft, cfg.hop_len,
                                                window=torch.hamming_window(cfg.n_fft).to(device), onesided=True)
                        
                        # print('clean', clean.shape)
                        # print('noisy', noisy.shape)
                        # print('est_mag', est_mag.shape)
                        # print('est_audio', est_audio.shape)
                        time_loss = torch.mean(torch.abs(est_audio - clean))
                        length = est_audio.size(-1)
                        # losses: loss_ri, loss_mag, time_loss
                        loss = cfg.loss_weights[0] * loss_ri + cfg.loss_weights[1] * loss_mag + cfg.loss_weights[2] * time_loss

                        """ METRICS """
                        est_audio = est_audio.detach().cpu()
                        clean = clean.detach().cpu()
                        
                        clean, est_audio = rearrange(clean, 'b t -> b 1 t'), rearrange(est_audio, 'b t -> b 1 t')
                        fwsnr_valid = loss_fwsnr(clean=clean, estimated=est_audio, batch_size=cfg.batch_size)
                        mrs_valid = loss_mrs(clean=clean, estimated=est_audio, batch_size=cfg.batch_size)
                        spec_l1_valid = loss_spec_l1(clean=clean, estimated=est_audio, batch_size=cfg.batch_size)
                        
                        ## print metrics
                        pbar.set_description(f'fwsnr: {fwsnr_valid:.2f}|mrs: {mrs_valid:.2f}|spec_l1: {spec_l1_valid:.2f}')
                        wandb.log({"fwsnr_valid": fwsnr_valid, "mrs_valid": mrs_valid, "spec_l1_valid": spec_l1_valid})
                        
                        
                        """ Validation epoch iteration END"""
    
    def init_dataloaders(self):
        print('Loading datasets...')
        cfg = self.cfg
        train_dataset = self.MusicDataset(cfg.dataset_musdb_path, split='training', rate=cfg.sr, audio_length=cfg.audio_length, samples_per_epoch=5000, instruments=self.instruments)
        val_dataset = self.MusicDataset(cfg.dataset_musdb_path, split='validation', rate=cfg.sr, audio_length=cfg.audio_length, samples_per_epoch=1000, instruments=self.instruments)
        test_dataset = self.MusicDataset(cfg.dataset_musdb_path, split='test', rate=cfg.sr, audio_length=cfg.audio_length, samples_per_epoch=100, instruments=self.instruments)
        
        print('Train set num: ', len(train_dataset))
        print('Valid set num: ', len(val_dataset))
        print('Test set num: ', len(test_dataset))

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, drop_last=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, drop_last=True, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, drop_last=True, shuffle=False)
        
        if cfg.rir_path:
            train_reverb = ReverbDataset(cfg.rir_path, split='training')
            val_reverb = ReverbDataset(cfg.rir_path, split='validation')
            test_reverb = ReverbDataset(cfg.rir_path, split='test')
            train_reverb_loader = iter(DataLoader(train_reverb, batch_size=cfg.batch_size))
            val_reverb_loader = iter(DataLoader(val_reverb, batch_size=cfg.batch_size))
            test_reverb_loader = iter(DataLoader(test_reverb, batch_size=cfg.batch_size))
        else:
            train_reverb_loader = val_reverb_loader = test_reverb_loader = None
        
        if cfg.noise_path:
            train_noise = NoiseDataset(cfg.noise_path, split='training', sample_length=cfg.audio_length)
            val_noise = NoiseDataset(cfg.noise_path, split='validation', sample_length=cfg.audio_length)
            test_noise = NoiseDataset(cfg.noise_path, split='test', sample_length=cfg.audio_length)
            train_noise_loader = iter(DataLoader(train_noise, batch_size=cfg.batch_size))
            val_noise_loader = iter(DataLoader(val_noise, batch_size=cfg.batch_size))
            test_noise_loader = iter(DataLoader(test_noise, batch_size=cfg.batch_size))
        else:
            train_noise_loader = val_noise_loader = test_noise_loader = None

        return dict(
            training=(train_loader, train_reverb_loader, train_noise_loader),
            validation=(val_loader, val_reverb_loader, val_noise_loader),
            test=(test_loader, test_reverb_loader, test_noise_loader)
        ) 
        

if __name__=='__main__':
    args = parse_argas()
    solver = Solver(args)
    solver.train()