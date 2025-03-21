import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import torchaudio as ta
import torchaudio.transforms as T
from einops import rearrange

from model.models import TFCModel

from data.dataset import ReverbDataset, NoiseDataset

from utils import metrics, utils, augmentation

import argparse
import importlib
import os; opj=os.path.join
from glob import glob
import stempeg
from tqdm import tqdm
from sweetdebug import sweetdebug; sweetdebug()


def load_model(configs, path_ckpt, remove_module=True):
    ckpt = torch.load(path_ckpt)
    model = TFCModel(
        num_channel=64,
        num_features=configs.n_fft//2+1,
        n_conformers=configs.n_conformers,
        attn_mask=None
    )

    if remove_module:
        ckpt = {k[7:]: v for k, v in ckpt['net'].items()}
    model.load_state_dict(ckpt)
    return model

def enhance_full_audio_batch(model, audio_batch, configs, augment=True, 
                             reverb_loader=None, noise_loader=None, eq_model=None, low_cut_filter=None,
                             enhance_random_segment=True):
    """
    audio_batch: (B, T)
    """
    segment_length = configs.audio_length
    # segment_length = (segment_length//256)*256
    overlap_length = segment_length // 4
    
    fade = T.Fade(fade_in_len=0, fade_out_len=0, fade_shape='linear')
    
    total_length = audio_batch.size(1)
    ## contain last segment
    num_segments = (total_length - segment_length) // (segment_length - overlap_length) + 1
    start_idx_list = [i * (segment_length - overlap_length) for i in range(num_segments)]
    end_idx_list = [i + segment_length for i in start_idx_list]
    overlap_list = [overlap_length for i in range(num_segments)]
    if end_idx_list[-1] != total_length:
        start_idx_list += [total_length - segment_length]
        end_idx_list += [total_length]
        overlap_list += [end_idx_list[-2]-total_length+segment_length]
        
    assert total_length == end_idx_list[-1]
    assert any([end_idx_list[i]-start_idx_list[i] == segment_length for i in range(len(start_idx_list))])
    
    noisy_batch = torch.zeros_like(audio_batch)
    est_batch = torch.zeros_like(audio_batch)
    
    if enhance_random_segment:
        start_idx_list = [0]
        end_idx_list = [segment_length]
        overlap_list = [0]
        noisy_batch = torch.zeros((audio_batch.size(0), segment_length))
        est_batch = torch.zeros((audio_batch.size(0), segment_length))
        new_audio_batch = torch.zeros((audio_batch.size(0), segment_length))
    
    # import pdb; pdb.set_trace()
        
    # if augment:
    #     assert reverb_loader is not None and noise_loader is not None and eq_model is not None and low_cut_filter is not None
    for ii in range(audio_batch.size(0)):
        audio = audio_batch[ii]
        
        if enhance_random_segment:
            start_idx = torch.randint(0, total_length-segment_length, (1,))
            end_idx = start_idx + segment_length
            
            audio = audio[start_idx:end_idx]
            new_audio_batch[ii] = audio
        
        audio = 0.95 * audio / audio.abs().max()
        
        noisy = torch.zeros_like(audio)
        est = torch.zeros_like(audio)
        
        
        for jj, (start, end, overlap) in enumerate(zip(tqdm(start_idx_list, leave=False, desc=f'(Aug) Enhancing stem {ii}'), end_idx_list, overlap_list)):
            audio_segment = audio[start:end]
            audio_segment = rearrange(audio_segment, 't -> 1 1 t')
            if augment:
                assert reverb_loader is not None and noise_loader is not None and eq_model is not None and low_cut_filter is not None
                audio_segment, noisy_segment = augmentation.augment(audio_segment,
                                                        rir=next(reverb_loader).to(audio.device), 
                                                        noise=next(noise_loader).to(audio.device), 
                                                        eq_model=eq_model, 
                                                        low_cut_model=low_cut_filter,
                                                        rate=configs.sr,
                                                        normalize=True)
            else:
                noisy_segment = audio_segment.clone()
                
            noisy_segment = rearrange(noisy_segment, '1 1 t -> 1 t')  # (B, T)
            noisy_spec = torch.stft(noisy_segment, configs.n_fft, configs.hop_len, window=torch.hamming_window(configs.win_len).to(audio.device), onesided=True, return_complex=False)
            noisy_spec = utils.power_compress(noisy_spec)
            noisy_spec = rearrange(noisy_spec, '1 c f t -> 1 c t f')
            
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    pred_dict = model(noisy_spec.float().to(audio.device))
            est_real, est_imag = pred_dict['real'], pred_dict['imag']
            est_real, est_imag = rearrange(est_real, 'b c t f -> b c f t'), rearrange(est_imag, 'b c t f -> b c f t')
            est_spec_uncompress = utils.power_uncompress(est_real, est_imag).squeeze(1)
            est_audio = torch.istft(est_spec_uncompress, configs.n_fft, configs.hop_len,
                                    window=torch.hamming_window(configs.win_len).to(audio.device), onesided=True)
            
            
            noisy_segment = noisy_segment.squeeze(0)
            est_audio = est_audio.squeeze(0)

            
            if jj == 0:
                if len(start_idx_list) >= 2:
                    fade.fade_out_len = overlap_list[jj+1]
                noisy[start:end] += fade(noisy_segment)
                est[start:end] += fade(est_audio)
            
            elif jj < len(start_idx_list)-1:
                fade.fade_out_len = overlap_list[jj+1]
                fade.fade_in_len = overlap
                noisy[start:end] += fade(noisy_segment)
                est[start:end] += fade(est_audio)
            else:
                fade.fade_in_len = overlap
                fade.fade_out_len = 0
                noisy[start:end] += fade(noisy_segment)
                est[start:end] += fade(est_audio)
            
        noisy_batch[ii] = noisy
        est_batch[ii] = est
    
    if enhance_random_segment:
        audio_batch = new_audio_batch
    return audio_batch, noisy_batch, est_batch
        
                
                
                

            
            

    
    

# def evaluate_musdb18(model, path_ckpt, path_musdb, path_noise, path_rir, sample_rate, audio_length, save_dir):
def evaluate_samples(configs, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # dir_list= [opj(save_dir, 'audio'), opj(save_dir, 'est'), opj(save_dir, 'noisy')]
    # for dir_ in dir_list:
    #     os.makedirs(dir_, exist_ok=True)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    path_ckpt = glob(opj(configs.save_dir, 'ckpt', '*.pt'))[-1]
    
    model = load_model(configs, path_ckpt)
    model = model.to(device)
    model.eval()
    
    configs.audio_length = (configs.audio_length//256)*256
    
    augment = True if args.eval_mode == 'musdb' else False

    if augment:
        test_reverb = ReverbDataset(configs.rir_path, split='test')
        test_noise = NoiseDataset(configs.noise_path, split='test', sample_length=configs.audio_length)
        reverb_loader = iter(DataLoader(test_reverb, batch_size=1, shuffle=False))
        noise_loader = iter(DataLoader(test_noise, batch_size=1, shuffle=False))
    else:
        reverb_loader = None
        noise_loader = None
    
    eq_model = augmentation.MicrophoneEQ(rate=configs.sr).to(device)
    low_cut_filter = augmentation.LowCut(cutoff_freq=35, rate=configs.sr).to(device)
    
    if args.eval_mode == 'musdb':
        data_path_list = glob(opj(configs.dataset_musdb_path, 'test', '*.stem.mp4'))
        stem_ids = [0, 1, 2, 3, 4]
    elif args.eval_mode == 'sample':
        data_path_list = [args.sample_path]  # 
        stem_ids = None
    else:
        raise ValueError('Invalid eval mode')
    
    pbar = tqdm(data_path_list, desc='Enhancing full song')
    
    
    for ii, path in enumerate(pbar):
        if ii > 15:
            break
        audio_batch = []
        if args.eval_mode == 'musdb':
            for stem_id in stem_ids:
                audio, rate = stempeg.read_stems(
                    filename=path,
                    stem_id=stem_id,
                    sample_rate=configs.sr,
                )
                # import pdb; pdb.set_trace()
                audio = torch.from_numpy(audio.T).float()
                audio = audio.mean(0, keepdim=True)
                audio_batch.append(audio)
        else:
            audio, rate = ta.load(path)

            if rate != configs.sr:
                resampler = T.Resample(orig_freq=rate, new_freq=configs.sr)
                audio = resampler(audio)
                # print(f"Resampled audio from {rate} Hz to {configs.sr} Hz")
            
            audio = audio.mean(0, keepdim=True)
            audio_batch.append(audio)
            # import pdb; pdb.set_trace()
        audio_batch = torch.cat(audio_batch, dim=0).to(device)
        
        augment = True if args.eval_mode == 'musdb' else False
        
        audio_batch, noisy_batch, est_batch = \
            enhance_full_audio_batch(model, audio_batch, configs, augment=augment, 
                                    reverb_loader=reverb_loader, noise_loader=noise_loader,
                                    eq_model=eq_model, low_cut_filter=low_cut_filter,
                                    enhance_random_segment=True if args.eval_mode == 'musdb' else False)
        
        
        for jj, (audio, noisy, est) in enumerate(zip(audio_batch, noisy_batch, est_batch)):
            audio = audio.unsqueeze(0).detach().cpu()
            noisy = noisy.unsqueeze(0).detach().cpu()
            est = est.unsqueeze(0).detach().cpu()
            ## torchaudio.save should have shape (1, T)
            if args.eval_mode == 'musdb':
                os.makedirs(opj(save_dir, str(ii), f'{stem_ids[jj]}'), exist_ok=True)
                ta.save(opj(save_dir, str(ii), f'{stem_ids[jj]}', 'audio.wav'), audio, sample_rate=configs.sr)
                ta.save(opj(save_dir, str(ii), f'{stem_ids[jj]}', 'noisy.wav'), noisy, sample_rate=configs.sr)
                ta.save(opj(save_dir, str(ii), f'{stem_ids[jj]}', 'est.wav'), est, sample_rate=configs.sr)
            elif args.eval_mode == 'sample':
                sample_name = os.path.basename(path)
                sample_name = os.path.splitext(sample_name)[0]
                os.makedirs(opj(save_dir, f'{sample_name}'), exist_ok=True)
                ta.save(opj(save_dir, f'{sample_name}', 'audio.wav'), audio, sample_rate=configs.sr)
                ta.save(opj(save_dir, f'{sample_name}', 'noisy.wav'), noisy, sample_rate=configs.sr)
                ta.save(opj(save_dir, f'{sample_name}', 'est.wav'), est, sample_rate=configs.sr)
            else:
                raise ValueError('Invalid eval mode')
            
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='exp2')
    parser.add_argument('--eval-mode', type=str, default='musdb', choices=['musdb', 'sample'])
    parser.add_argument('--sample-path', type=str, default=None, help='If musdb, path is in configs')
    parser.add_argument('--save-dir', type=str, default=None, help='path to save')
    args = parser.parse_args()
    
    cfg = importlib.import_module(f'configs.{args.configs}').Configs
    evaluate_samples(configs=cfg, args=args)
