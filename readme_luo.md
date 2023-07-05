1. Show IIV embeddings visually
    - See inter-emo descrimitive
    - see intra-emo descrimitive (by samples)
2. Show wav2net embeddings


## prepare_align.py
- wav normalization
- text clean


# Training
## 1. Dataset
- ESD
## 2. Preprocess
Previously, get wav2net and opensmile embeddings from ESD dataset by run
```
python3 
```


First, Pair wav and text for alignment by running 
```
python3 prepare_align.py config/LJSpeech/preprocess.yaml
```
Second, Obtain aligned TextGrid file on [](), and unzip the files in ``preprocessed_data/ESD/TextGrid/``.

Alternately, you can align the corpus by yourself. 
Download the official MFA package and run
```
./montreal-forced-aligner/bin/mfa_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt english preprocessed_data/LJSpeech
```
or
```
./montreal-forced-aligner/bin/mfa_train_and_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt preprocessed_data/LJSpeech
```

to align the corpus and then run the preprocessing script.
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

Third, run the preprocessing script by
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

## 3. Start Train
- Train IIV model
Train your model with
```
cd IIV
python3 train.py
```

- Get trained IIV embeddings
```
cd IIV
python3 get_trained_IIV.py config/ESD/preprocess.yaml

# visualized by 
python3 visualization.py config/ESD/preprocess.yaml
```

- Train FastIIV model
```
python3 train.py -p config/ESD/preprocess.yaml -m config/ESD/model.yaml -t config/ESD/train.yaml
```


## 4. TensorBoard

Use
```
tensorboard --logdir output/log/ESD
```


# Inference

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
