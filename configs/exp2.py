from dotted_dict import DottedDict
import os; opj = os.path.join

class Configs:
    gpus = '6, 7'
    dataset = 'musdb' # or 'solo'
    data_rootdir = '/data/yoongidata/temp/data/me_github'
    dataset_medleydbsolos_path = opj(data_rootdir, 'medley-solos-db')
    dataset_musdb_path = opj(data_rootdir, 'musdb18')
    rir_path = opj(data_rootdir, 'reverb_samples_16000_hz.npz')
    noise_path = opj(data_rootdir, 'noise_samples_47555_length_16000_hz.npz')
    
    # for training exp
    # rir_path = '/data/yoongidata/temp/data/dns_rir/impulse_responses/reverb_samples_16000_hz.npz'

    save_dir = './results/exp2'
    
    audio_length = 47555
    batch_size = 16
    
    # model_type = 'CPq'
    n_conformers = 2
    # n_conformers = 2
    n_fft = 1024
    win_len=1024
    hop_len=256
    sr = 16000
    
    total_epochs = 50
    valid_per = 1
    lr = 1e-4
    ckpt_per = 2
    
    loss_weights = [0.1, 0.9, 0.1]
    
    # resume = True
    # ckpt_path = './results/ckpt/0032.pt'
    # start_epoch = 32