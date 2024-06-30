import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import pandas as pd
from glob import glob

import logging
import pdb
from tqdm import tqdm
import itertools
import os ; opj=os.path.join

from sklearn.model_selection import train_test_split

from utils import utils
import random 
import stempeg

random.seed(0)

EPS = torch.finfo(float).eps
  
class MedleyDBSolosDataset(Dataset):
    def __init__(self, data_dir, split='training', instruments=None, rate=16000, normalize=True, data_num=None, audio_length=47500):
        """
        data_dir: path to MedleyDB Solos data directory
        split: training, validation, or test split of dataset
        instruments: list of instruments to use or None for all instruments
        rate: sampling rate
        """
        assert split in ['training', 'test', 'validation']
        self.data_dir = data_dir
        self.rate = rate
        self.audio_length=audio_length
        
        if instruments[0] == 'None':
            instruments=None
            
        self.full_metadata = pd.read_csv(os.path.join(self.data_dir, 'Medley-solos-DB_metadata.csv'))
        self.instruments = instruments if not instruments is None else self.full_metadata.instrument.unique()
        self.metadata = self.full_metadata[(self.full_metadata.subset == split) & (self.full_metadata.instrument.isin(self.instruments))]
        self.file_list = [os.path.join(self.data_dir, f'Medley-solos-DB_{subset}-{instrument_id}_{uuid4}.wav') \
                          for subset, instrument_id, uuid4 in zip(self.metadata.subset, self.metadata.instrument_id, self.metadata.uuid4)]
        
        if data_num is not None:
            # self.file_list = self.file_list[:data_num]
            self.file_list = random.sample(self.file_list, data_num)
            
        logging.info(f'Reading MedleyDB Solos Samples from split {split}')
        self.samples = []
        for f in tqdm(self.file_list):
            self.samples.append(utils.load(f, self.rate)[:, :self.audio_length])
        self.samples = torch.stack(self.samples)
        logging.info(f'Read {self.samples.shape[0]} music samples')   
        
        if normalize:
            self.samples = 0.95*self.samples/self.samples.abs().max(dim=2, keepdim=True)[0]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index]

        
class InfiniteDataset(IterableDataset):
    def __init__(self):
        """
        Base class for datasets that repeats forever
        """
        self.index = 0
        self.data = []
        
    def __iter__(self):
        assert len(self.data) > 0, 'Must initialize data before iterating'
        while True:
            yield self.data[self.index]
            self.index = (self.index + 1) % len(self.data)


class NoiseDataset(InfiniteDataset):
    def __init__(self, noise_source, sample_length=47500, rate=16000, split='training'):
        """
        noise_source: either list of directories with .wav files or path to .npy file
        sample_length: length to cut noise samples to
        rate: sampling rate
        """
        assert split in ['training', 'test', 'validation']
        self.sample_length = sample_length
        self.rate = rate
        self.index = 0
        
        if isinstance(noise_source, list):
            self.noise_files = noise_source
            logging.info('Loading individual noise sample files')
            self.data = []
            for f in tqdm(self.noise_files):
                noise_sample = utils.load(f, self.rate)
                noise_sample = noise_sample.mean(0, keepdim=True)
                noise_sample = torch.stack(torch.split(noise_sample, sample_length, dim=1)[:-1])
                self.data.append(noise_sample)
            self.data = torch.cat(self.data)
        else:
            logging.info(f'Batch loading from noise sample file for split {split}')
            noises = np.load(noise_source)[split]
            self.data = torch.from_numpy(noises[...,:sample_length])

        
        logging.info(f'Read {self.data.shape[0]} noise samples')

    def save(self, path):
        """
        Save data for batch loading later
        path: path to save data at
        """
        logging.info(f'Saving noise samples to {path}')
        np.save(path, self.data.numpy()) 


class ReverbDataset(InfiniteDataset):
    def __init__(self, reverb_source, rate=16000, split='training', trim=True):
        """
        reverb_source: either list of directories with .wav files or path to .npy file
        rate: sampling rate
        """
        assert split in ['training', 'test', 'validation']
        self.rate = rate
        self.index = 0
        # import pdb; pdb.set_trace()
        
        if isinstance(reverb_source, list):
            self.reverb_files = \
                list(itertools.chain.from_iterable([glob(os.path.join(d, '**/*.wav'), recursive=True) for d in reverb_source]))
            # self.reverb_files = reverb_source
            logging.info('Loading individual room impulse response files')
            self.data = []
            for f in tqdm(self.reverb_files):
                r = utils.load(f, self.rate)
                if trim:
                    direct_impulse_index = r.argmax().item()
                    window_len = int(2.5/1000*self.rate)
                    if direct_impulse_index < window_len:
                        r = torch.cat((torch.zeros(1, window_len - direct_impulse_index), r), dim=1)
                    r = r[:, direct_impulse_index - window_len:]
                self.data.append(r)
            max_ir_length = max([d.shape[1] for d in self.data])
            self.data = [torch.cat((d, torch.zeros(1, max_ir_length-d.shape[1])), dim=1) for d in self.data]
            self.data = torch.stack(self.data)
        else:
            logging.info(f'Batch loading from room impulse response file for split {split}')
            self.data = torch.from_numpy(np.load(reverb_source)[split])

        logging.info(f'Read {self.data.shape[0]} room impulse response samples')

    def save(self, path):
        """
        Save data for batch loading later
        path: path to save data at
        """
        logging.info(f'Saving reverb samples to {path}')
        np.save(path, self.data.numpy()) 



class MUSDB18Dataset(Dataset):
    def __init__(self, data_dir, split='training', rate=16000, normalize=True, audio_length=47500, 
                 data_num=None, instruments=None, samples_per_epoch=10000):
        
        assert split in ['training', 'test', 'validation']
        self.rate = rate
        self.data_dir = data_dir
        self.audio_length = audio_length
        if samples_per_epoch is not None:
            self.samples_per_epoch = samples_per_epoch
        else:
            self.samples_per_epoch = 10000
        self.normalize = normalize
        
        
        if split in ['training', 'validation']:
            data_dir_list = glob(opj(data_dir, 'train', '*.mp4'))
            # import pdb; pdb.set_trace()
            path_train_list, path_valid_list = train_test_split(data_dir_list, test_size=0.1, shuffle=True, random_state=1)
            if split == 'training':
                self.path_list = path_train_list
            elif split == 'validation':
                self.path_list = path_valid_list
        elif split == 'test':
            ## not used
            self.path_list = glob(opj(data_dir, 'test', '*.mp4'))
            
        self.num_data = len(self.path_list)
        
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, index):
        # return self.samples[index]
        index = index % self.num_data
        path = self.path_list[index]
        # stem_id = random.choice([0, 1, 2, 3, 4])
        stem_id = random.choices([0, 1, 2, 3, 4], weights=[0.6, 0.1, 0.1, 0.1, 0.1], k=1)[0]
        # audio = utils.load(path, self.rate)
        audio, rate = stempeg.read_stems(
            filename=path,
            stem_id=stem_id
        )
        audio = torch.from_numpy(audio.T).float()
        audio = audio.mean(0, keepdim=True)
        ## random crop from audio (sample length)
        if audio.shape[-1] < self.audio_length:
            audio = torch.cat([audio, torch.zeros(1, self.audio_length - audio.shape[-1])], dim=-1)
        else:
            start = random.randint(0, audio.shape[-1] - self.audio_length)
            audio = audio[:, start:start+self.audio_length]
        
        if self.normalize:
            audio = 0.95 * audio / (audio.abs().max(dim=-1, keepdim=True)[0]+EPS)
        
        # import pdb; pdb.set_trace()
        return audio