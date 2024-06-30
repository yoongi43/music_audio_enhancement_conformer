# Dataset 
- Download
  - MUSDB18
    - .../musdb18/
  - noise: ACE
    - .../Single
  - rir: DNS
    - .../SLR26

- Prepare
  - data_preproc.py
  - .../noise_samples_47555_length_16000_hz.npz
  - .../reverb_samples_16000_hz.npz
  
- File path setting

# Train code
python scripts/run_cmd_train.py

# Inference MUSDB18
python evaluate.py --configs exp2 --eval-mode musdb --save-dir path/to/save/

# Inference a file
python evaluate.py --configs exp2 --eval-mode sample --sample-path path/to/sample.wav --save-dir path/to/save/

# Strong Reference
These codes are based on official implementation of CMGAN paper and Music Enhacnement paper. 
- CMGAN
- MusicEnhancement