1. Show IIV embeddings visually
    - See inter-emo descrimitive
    - see intra-emo descrimitive (by samples)
2. Show wav2net embeddings


## prepare_align.py
- wav normalization
- text clean

## preprocess
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

## Data format
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



In total, 62 parameters are contained in the Geneva
Minimalistic Standard Parameter Set.

F0semitoneFrom27.5Hz_sma3nz_amean
F0semitoneFrom27.5Hz_sma3nz_stddevNorm
F0semitoneFrom27.5Hz_sma3nz_percentile20.0
F0semitoneFrom27.5Hz_sma3nz_percentile50.0
F0semitoneFrom27.5Hz_sma3nz_percentile80.0
F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2
F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope
F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope
F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope
F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope
loudness_sma3_amean
loudness_sma3_stddevNorm
loudness_sma3_percentile20.0
loudness_sma3_percentile50.0
loudness_sma3_percentile80.0
loudness_sma3_pctlrange0-2
loudness_sma3_meanRisingSlope
loudness_sma3_stddevRisingSlope
loudness_sma3_meanFallingSlope
loudness_sma3_stddevFallingSlope
jitterLocal_sma3nz_amean
jitterLocal_sma3nz_stddevNorm
shimmerLocaldB_sma3nz_amean
shimmerLocaldB_sma3nz_stddevNorm
HNRdBACF_sma3nz_amean
HNRdBACF_sma3nz_stddevNorm
logRelF0-H1-H2_sma3nz_amean
logRelF0-H1-H2_sma3nz_stddevNorm
logRelF0-H1-A3_sma3nz_amean
logRelF0-H1-A3_sma3nz_stddevNorm
F1frequency_sma3nz_amean
F1frequency_sma3nz_stddevNorm
F1bandwidth_sma3nz_amean
F1bandwidth_sma3nz_stddevNorm
F1amplitudeLogRelF0_sma3nz_amean
F1amplitudeLogRelF0_sma3nz_stddevNorm
F2frequency_sma3nz_amean
F2frequency_sma3nz_stddevNorm
F2amplitudeLogRelF0_sma3nz_amean
F2amplitudeLogRelF0_sma3nz_stddevNorm
F3frequency_sma3nz_amean
F3frequency_sma3nz_stddevNorm
F3amplitudeLogRelF0_sma3nz_amean
F3amplitudeLogRelF0_sma3nz_stddevNorm
alphaRatioV_sma3nz_amean
alphaRatioV_sma3nz_stddevNorm
hammarbergIndexV_sma3nz_amean
hammarbergIndexV_sma3nz_stddevNorm
slopeV0-500_sma3nz_amean
slopeV0-500_sma3nz_stddevNorm
slopeV500-1500_sma3nz_amean
slopeV500-1500_sma3nz_stddevNorm
alphaRatioUV_sma3nz_amean
hammarbergIndexUV_sma3nz_amean
slopeUV0-500_sma3nz_amean
slopeUV500-1500_sma3nz_amean
loudnessPeaksPerSec
VoicedSegmentsPerSec
MeanVoicedSegmentLengthSec
StddevVoicedSegmentLengthSec
MeanUnvoicedSegmentLength
StddevUnvoicedSegmentLength

Emotion distribution: 

3197 audios in Happy
3197 audios in Angry
3195 audios in Neutral
3197 audios in Surprise
3199 audios in Sad

emo_clusterN = {
    "Angry": 3,
    "Surprise": 3,
    "Sad": 3,
    "Neutral": 3,
    "Happy": 3
}

emo_optimal_clusterN = {
    "Angry": 3,
    "Surprise": 2,
    "Sad": 4,
    "Neutral": 2,
    "Happy": 2
}



start : Happy
Rejected dimension Nums are 43
(7, (8272.97917983235, 0.0, 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope'))
(6, (5957.670711125929, 0.0, 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope'))


start : Angry
Rejected dimension Nums are 41
(51, (250307.2986151871, 0.0, 'slopeV500 - 1500_sma3nz_stddevNorm'))
(7, (3998.97808142092, 0.0, 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope'))
(6, (2285.123573831205, 0.0, 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope'))

start : Neutral
Rejected dimension Nums are 4
(27, (44521.43236623274, 0.0, 'logRelF0 - H1 - H2_sma3nz_stddevNorm'))

start : Surprise
Rejected dimension Nums are 3
(27, (309532.61506744905, 0.0, 'logRelF0 - H1 - H2_sma3nz_stddevNorm'))

start : Sad
Rejected dimension Nums are 55
(51, (79907.38863334325, 0.0, 'slopeV500 - 1500_sma3nz_stddevNorm'))
(7, (2154.4426057002097, 0.0, 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope'))
(6, (1528.2080025670027, 0.0, 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope'))
(36, (1499.098732559102, 0.0, 'F2frequency_sma3nz_amean'))


- **logRelF0**: 
The measure H1-H2, the difference in amplitude between the first and second harmonics, is frequently used to distinguish phonation types and to characterize differences across voices and genders. While H1-H2 can differentiate voices and is used by listeners to perceive changes in voice quality, its relation to voice articulation is less straightforward. 

- F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope (**F0_riseSlope_SD**)
- F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope (**F0_riseSlope_mean**)
mean and standard deviation of the slope of rising/falling signal parts


- **slopeV500 - 1500_sma3nz_stddevNorm**
linear regression slope of the logarithmic power spectrum within the two given bands.


reference paper of opensmile (page 5):
https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf



Spectral slope description powerpoint
https://www.isca-speech.org/archive_open/archive_papers/spkd2008/material/spkd_007_p.pdf



- Contributable dimension per cluster..
/home/rosen/anaconda3/envs/fastts/bin/python /home/rosen/project/FastSpeech2/preprocess_ESD.py 
start : Happy
{0: {7: 950.6254, 6: 716.818}, 1: {7: 76.40261, 6: 91.59863}}


start : Angry
{0: {51: -1.932109, 7: 1193.8047}, 1: {51: -1.7930673, 7: 132.99066}, 2: {51: -37295.926, 7: 113.0735}}

start : Neutral
{0: {27: 0.46332216}, 1: {27: -22376.957}}

start : Surprise
{0: {27: 0.8328492}, 1: {27: 29302.355}}

start : Sad
{0: {51: -11.148005, 7: 1049.5886}, 1: {51: -1.7410944, 7: 122.96218}, 2: {51: -18559.893, 7: 141.74426}, 3: {51: -2.5960681, 7: 74.79896}}
Process finished with exit code 0


        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }
        
        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )


          def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        style_emb=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        #speech=None,
    ):


      return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

