# Training
## 1. Dataset
- ESD
## 2. Preprocess
### 2.1 Preprocess for FastSpeech2
- Get wav2net emb (Used to train IIV emb), mel (Not used), wav, and metadata.json from ESD dataset
  - metadata.json includes text, length, emotion, phonemes, and group which is appended in next step
```
python3 preprocess_ESD.py --config prepared_esd
```

- Prepare psd emb (used for cluster), group id and updated into metadata_new.json
  - Check optimal cluster number and get contributing prosody with F-value in Annova test
  - Cluster by k-means
```
python3 preprocess_ESD.py --config ?
```

- Pair wav (normalized) and text (Cleaned) for alignment in raw_data dir 
  - (Not needed if TextGrid dir is available)

```
python3 prepare_align.py config/LJSpeech/preprocess.yaml
```

- Download aligned TextGrid dir, and unzip the files in ``preprocessed_data/ESD/TextGrid/``.
  - Alternately, you can align the corpus by yourself. Download the official MFA package and run
  ```
  ./montreal-forced-aligner/bin/mfa_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt english preprocessed_data/LJSpeech
  ```
    - or
  ```
  ./montreal-forced-aligner/bin/mfa_train_and_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt preprocessed_data/LJSpeech
  ```

    - to align the corpus and then run the preprocessing script.
  ```
  python3 preprocess.py config/LJSpeech/preprocess.yaml
  ```
- Get mel, energy, duration, pitch given wav and textGrid.
  - mel is get by con1d ?
  - Wav is segmented by phoneme, given by textGrid
  - pitch extracted by pw.dio from segmented wav
  - energy extracted by the amptitude of mel from segmented wav
```
python3 preprocess.py config/ESD/preprocess.yaml
```


## 3. Start Train
- Train IIV model
```
cd IIV
python3 train.py
```

- Get IIV embeddings by best iiv models
```
cd IIV
python3 get_trained_IIV.py config/ESD/preprocess.yaml

# visualized by 
python3 visualization.py config/ESD/preprocess.yaml
```

- Train FastIIV model, given iiv trained embeds
```
python3 train.py -p config/ESD/preprocess.yaml -m config/ESD/model.yaml -t config/ESD/train.yaml
```


## 4. TensorBoard

Use
```
tensorboard --logdir output/log/ESD
```

## Loss
$
Loss = w * L_{softmax} + (1-w) * L_{IIV_triplet} + ?
$

$
L_{IIV_triplet} = L_{inter}(a^{p}, x^{p}, x^{n}) + \sum_{g=1}^{G}{L_{intra}(a_{g}^{p}, x_{g}^{p}, x_{g}^{n})}
$


## 5. Paper evaluation
- 


# Inference
- get iiv stats 
  - Extract mean_anchors_distance
  - Extract_mean_anchors (equal to representative embeddings)
    - mean_anchors is used for iiv embeddings training
    - repr embeddings is used for inference
```
python3 get_iiv_stats.py
```

## 1. preprocess
- process_utterance
    - read TextGrid (alignment) file
    - pitch, energy, duration, and mel extraction
        - Extraction
        - pitch interpolation (averaging)
        - get phone-level pitch
        - remove outliear
- normalization of pitch, energy
- write to meta data 
    id | speaker | phone | text

## 2. Data format
- original_data (Any Format)

- raw_path   <- Get by prepare_align
  - speaker1
    - wav1.wav
    - wav1.lab
  - ...

- preprocessed_path   <- Get by preprocess
  - Textgrid
    - speaker1
      - wav1.TextGrid
