"""
Using downloaded data, split into training, validation, and test sets
1. Download data
    - ACE: http://www.ee.ic.ac.uk/naylor/ACEweb/index.html
        - https://zenodo.org/records/6257551 (Single-channel: ACE_Corpus_RIRN_Single.tbz2)
            - https://zenodo.org/records/6257551/files/ACE_Corpus_RIRN_Single.tbz2?download=1
    - RIR: DNS Challenge
        - git clone https://github.com/microsoft/DNS-Challenge
    - Medley-db solos
        - https://zenodo.org/records/1344103#.Yg__Yi-B1QI
    - Musdb18
        - https://zenodo.org/records/1117372
2. Noise: We only use babble and ambient noise from ACE dataset
3. Reverb: We only use smallroom and mediumroom from SLR26


"""
## Source: https://github.com/nkandpa2/music_enhancement/tree/master
## Process and split data
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data.dataset import ReverbDataset, NoiseDataset
import torch
import numpy as np
from glob import glob
import argparse
# from sweetdebug import sweetdebug; sweetdebug()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir-rir', 
                        type=str,
                        default=None,
                        help = 'path to rir dataset. i.e., .../SLR26/')
    parser.add_argument('--data-dir-noise',
                        type=str,
                        default = None,
                        help='path to noise data (ACE). i.e., .../Single/')

    parser.add_argument('--output-dir', 
                        type=str,
                        default=None,
                        help='path to output directory to save the processed npz files')


    parser.add_argument('--sample-rate', type=int, default=16000, help='sample rate')
    parser.add_argument('--sample-length', type=int, default=47555, help='length to cut noise samples to')
    parser.add_argument('--valid-fraction', type=float, default=0.1, help='fraction of data to reserve for validation')
    parser.add_argument('--test-fraction', type=float, default=0.1, help='fraction of data to reserve for testing')
    args = parser.parse_args()
    return args
    

def split_from_data(data, valid_fraction, test_fraction):
    data = data[torch.where(~torch.all(data == 0, dim=2))].unsqueeze(1)
    num_samples = data.shape[0]
    data = data[torch.randperm(num_samples)]
    splits = [int((1-valid_fraction-test_fraction)*num_samples), int(test_fraction*num_samples)]
    splits.append(num_samples - splits[0] - splits[1])
    train, valid, test = torch.split(data, splits)
    train, valid, test = train.numpy(), valid.numpy(), test.numpy()
    return train, valid, test
    

def process_rir(data_dir_rir, sample_rate:int, valid_fraction, test_fraction, save_root_dir):
    folder_list = ['simulated_rirs_48k/smallroom', 'simulated_rirs_48k/mediumroom']
    data_dir_list_rir = [os.path.join(data_dir_rir, folder) for folder in folder_list]
    # import pdb; pdb.set_trace() 
    dataset = ReverbDataset(data_dir_list_rir, rate=sample_rate)
    out_file = f'reverb_samples_{sample_rate}_hz.npz'
    train, valid, test = split_from_data(dataset.data, valid_fraction, test_fraction)
    np.savez(os.path.join(save_root_dir, out_file), training=train, validation=valid, test=test)


def process_noise(data_dir_noise, sample_length, sample_rate, valid_fraction, test_fraction, save_root_dir):
    ## If there is no separate directory named ace-ambient and ace-bubble, make it from the full ACE dataset
    ## data_dir_noise = '...Single'. i.e., ACE dataset path
    noise_names = ['Ambient', 'Babble']
    total_wav_path_list = glob(os.path.join(data_dir_noise, '**', '*.wav'), recursive=True)
    noise_wav_list = [wav for wav in total_wav_path_list if 
                      any([name in os.path.basename(wav) for name in noise_names])]
    dataset = NoiseDataset(noise_wav_list, rate=sample_rate, sample_length=sample_length)
    out_file = f'noise_samples_{sample_length}_length_{sample_rate}_hz.npz'
    
    train, valid, test = split_from_data(dataset.data, valid_fraction, test_fraction)
    np.savez(os.path.join(save_root_dir, out_file), training=train, validation=valid, test=test)


if __name__=='__main__':
    """
    data_dir_rir: data dir to  'SLR26' folder of DNS Challenge dataset
    data_dir_noise: data dir to 'Single' folder of ACE Dataset
    save_root_dir: root dir to save the processed data
    -> in save_root_dir, the code will save the processed data as npz files
    """
    args = parse_args()

    sample_rate = args.sample_rate
    sample_length = args.sample_length
    valid_fraction = args.valid_fraction
    test_fraction = args.test_fraction
    save_root_dir = args.output_dir
    data_dir_rir = args.data_dir_rir
    data_dir_noise = args.data_dir_noise

    process_rir(data_dir_rir, sample_rate, valid_fraction, test_fraction, save_root_dir)
    process_noise(data_dir_noise, sample_length, sample_rate, valid_fraction, test_fraction, save_root_dir)
    