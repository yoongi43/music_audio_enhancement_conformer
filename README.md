# Exploiting Time-Frequency Conformers for Music Audio Enhancement
This repository is the implementation of the paper "Exploiting Time-Frequency Conformers for Music Audio Enhancement" presented in ACM MM 2023.

## Conda Environment
- Python 3.8
- Pytorch 1.13.1
  - `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
- Install libraries
  - `pip3 install -r requirements.txt`

## Dataset 
1. Download dataset from each link.
   - MUSDB18
     - https://zenodo.org/records/1117372
   - Noise dataset: ACE Challenge 
     - (http://www.ee.ic.ac.uk/naylor/ACEweb/index.html)
     - https://zenodo.org/records/6257551 (Single-channel: ACE_Corpus_RIRN_Single.tbz2)
   - Rir dataset: DNS Challenge
     - https://github.com/microsoft/DNS-Challenge
     - We use the SLR26 dataset.
2. Prepare dataset.
   Process the dataset using the `data_preproc.py` script.
   - `python scripts/data_preproc.py --data-dir-rir .../SLR26/ ---data-dir-noise .../Single/ --output-dir path/to/save/files/`

3. Final dataset directory structure should be like below.
```
+-- data
|  + musdb18
|  |  + train
|  |  + test
|  + noise_samples_47555_length_16000_hz.npz
|  + reverb_samples_16000_hz.npz
```

## Training code
- Set your configurations in configs/exp{num}.yaml
- Especially, set the dataset directiory in the configuration file.
  - `data_rootdir = '/path/to/data/dirs/'`
- python main.py --configs exp2

## Inference MUSDB18
Checkpoints are contained in the 'results' directory.

`python evaluate.py --configs exp2 --eval-mode musdb --save-dir path/to/save/`

## Inference an audio file
`python evaluate.py --configs exp2 --eval-mode sample --sample-path path/to/sample.wav --save-dir path/to/save/`

## Reference
This repository is based on the official implementations of the following papers.
- CMGAN: Conformer-Based Metric GAN for Monaural Speech Enhancement
  - https://github.com/ruizhecao96/CMGAN
- Music Enhancement via Image Translation and Vocoding
  - https://github.com/nkandpa2/music_enhancement

## Citation
If you find this code useful, please consider citing our paper.
```
@inproceedings{10.1145/3581783.3612269,
author = {Chae, Yunkee and Koo, Junghyun and Lee, Sungho and Lee, Kyogu},
title = {Exploiting Time-Frequency Conformers for Music Audio Enhancement},
year = {2023},
isbn = {9798400701085},
doi = {10.1145/3581783.3612269},
pages = {2362–2370},
}
```
