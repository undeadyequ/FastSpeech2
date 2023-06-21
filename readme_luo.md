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



