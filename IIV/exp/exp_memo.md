# Problem
- IIV embeddings distribution is not intra- descrimitive
  - even use intra-only training ?
  - The distr of intra-only training is converge to less point than that of inter and intra training
  - The w=0.3 separete intra better than w=0.7
- Style of ESD is not that clear
  - EmoV-DB is not parallel, is ok?
- Set separation evaluation


# Task
- Test attention score given different emo/cluster
- Better cluster separation (Loss) -> only 
  - Intra_margin 0.2 -> 0.1
- Better synthesized emotional audio
  - ESD -> EmoV-DB
- Intensity

## K-means clustering result
- samples distribution in cluster
- contribute prosody by fp value
- mean of contribute prosody for each cluster

```
Happy: Total 3197, Cluster_distr:{0: 3051, 1: 146}
Angry: Total 3197, Cluster_distr:{0: 3196, 1: 1}, one_element_id:['0016_000482']
Neutral: Total 3195, Cluster_distr:{0: 3194, 1: 1}, one_element_id:['0011_000244']
Surprise: Total 3197, Cluster_distr:{0: 3196, 1: 1}, one_element_id:['0011_001657']
Sad: Total 3199, Cluster_distr:{0: 3074, 1: 124, 2: 1}, one_element_id:['0013_001189']

Rejected dimension:43
emo:Happy
dim_fpValue_name
(7, (8272.979179832355, 0.0, 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope'))
(6, (5957.670711125929, 0.0, 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope'))
cls_contribDim_mean:
{0: {7: 76.40261, 6: 91.59863}, 1: {7: 950.6254, 6: 716.818}}

Rejected dimension:4
emo:Angry
dim_fpValue_name
(51, (500771.1534533934, 0.0, 'slopeV500 - 1500_sma3nz_stddevNorm'))
(12, (6.702843905665723, 0.00966958642050155, 'loudness_sma3_percentile20.0'))
cls_contribDim_mean:
{0: {51: -1.8005936, 12: 0.033792235}, 1: {51: -37295.926, 12: 0.09066387}}

Rejected dimension:4
emo:Neutral
dim_fpValue_name
(27, (44521.43236623274, 0.0, 'logRelF0 - H1 - H2_sma3nz_stddevNorm'))
(20, (13.731169552216194, 0.0002145019349511059, 'jitterLocal_sma3nz_amean'))
cls_contribDim_mean:
{0: {27: 0.46332216, 20: 0.031113993}, 1: {27: -22376.957, 20: 0.07964794}}

Rejected dimension:3
emo:Surprise
dim_fpValue_name
(27, (309532.61506744905, 0.0, 'logRelF0 - H1 - H2_sma3nz_stddevNorm'))
(17, (5.094824155279172, 0.02406461080226749, 'loudness_sma3_stddevRisingSlope'))
cls_contribDim_mean:
{0: {27: 0.8328492, 17: 4.583166}, 1: {27: 29302.355, 17: 9.090978}}

Rejected dimension:37
emo:Sad
dim_fpValue_name
(51, (119877.89445524792, 0.0, 'slopeV500 - 1500_sma3nz_stddevNorm'))
(7, (3101.447741250889, 0.0, 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope'))
(6, (2191.6572932804465, 0.0, 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope'))
cls_contribDim_mean:
{0: {51: -2.1296902, 7: 100.56144, 6: 113.49736}, 1: {51: -10.926141, 7: 1039.4904, 6: 1102.7955}, 2: {51: -18559.893, 7: 141.74426, 6: 211.20401}}
```


## IIV embedding training
- Hyper, loss, and triplet sample nums

- iiv_conv2d_anchorFalse_02_02_w07_-1

Epoch 1 Iteration 0: Loss = 0.15, inter:0.18, intra:0.07-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.18,0.0,0.18, Number of inter- and intra-triplets = 11545165, 0_0_21129_0_14684
Epoch 1 Iteration 10: Loss = 0.10, inter:0.12, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.12, Number of inter- and intra-triplets = 4196414, 0_0_11069_0_4562
Epoch 1 Iteration 20: Loss = 0.11, inter:0.12, intra:0.08-> Angry,Surprise,Sad,Neutral,Happy : 0.11,0.0,0.13,0.0,0.13, Number of inter- and intra-triplets = 3468359, 1058_0_12930_0_8762
Epoch 1 Iteration 30: Loss = 0.10, inter:0.12, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.12,0.0,0.12, Number of inter- and intra-triplets = 4208136, 0_0_3935_0_11903
best acc: 0.1, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 2 Iteration 0: Loss = 0.10, inter:0.12, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.14,0.0,0.11, Number of inter- and intra-triplets = 3518206, 0_0_6428_0_4388
Epoch 2 Iteration 10: Loss = 0.10, inter:0.12, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.13,0.0,0.10, Number of inter- and intra-triplets = 3049914, 0_0_11073_0_4127
Epoch 2 Iteration 20: Loss = 0.09, inter:0.12, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 3000866, 0_0_6240_0_3589
Epoch 2 Iteration 30: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2191019, 0_0_3403_0_2668
best acc: 0.2, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 3 Iteration 0: Loss = 0.09, inter:0.11, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.11, Number of inter- and intra-triplets = 2669319, 0_0_2004_0_5639
Epoch 3 Iteration 10: Loss = 0.09, inter:0.11, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.12,0.0,0.11, Number of inter- and intra-triplets = 2626965, 0_0_11763_0_2315
Epoch 3 Iteration 20: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2526839, 0_0_5027_0_1317
Epoch 3 Iteration 30: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.08,0.0,0.10, Number of inter- and intra-triplets = 2299355, 0_0_936_0_11493
best acc: 0.30000000000000004, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 4 Iteration 0: Loss = 0.09, inter:0.11, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.12, Number of inter- and intra-triplets = 3003694, 0_0_6200_0_5341
Epoch 4 Iteration 10: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.11, Number of inter- and intra-triplets = 2367720, 0_0_15175_0_11006
Epoch 4 Iteration 20: Loss = 0.09, inter:0.11, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.11, Number of inter- and intra-triplets = 2577261, 0_0_4345_0_5377
Epoch 4 Iteration 30: Loss = 0.09, inter:0.11, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.11, Number of inter- and intra-triplets = 2269200, 0_0_3381_0_5694
best acc: 0.4, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 5 Iteration 0: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.10, Number of inter- and intra-triplets = 2162358, 0_0_2715_0_7986
Epoch 5 Iteration 10: Loss = 0.10, inter:0.11, intra:0.07-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.13,0.12,0.0,0.10, Number of inter- and intra-triplets = 2348703, 0_3059_4320_0_5607
Epoch 5 Iteration 20: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2254821, 0_0_13820_0_4570
Epoch 5 Iteration 30: Loss = 0.09, inter:0.11, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.12,0.0,0.10, Number of inter- and intra-triplets = 2571432, 0_0_8464_0_10156
best acc: 0.5, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 6 Iteration 0: Loss = 0.09, inter:0.11, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.12,0.0,0.10, Number of inter- and intra-triplets = 2391772, 0_0_4367_0_5231
Epoch 6 Iteration 10: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2120761, 0_0_2489_0_4437
Epoch 6 Iteration 20: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2241309, 0_0_19301_0_4919
Epoch 6 Iteration 30: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.10, Number of inter- and intra-triplets = 3246528, 0_0_5125_0_11824
best acc: 0.6, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 7 Iteration 0: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2424942, 0_0_11236_0_6653
Epoch 7 Iteration 10: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2407275, 0_0_12027_0_6731
Epoch 7 Iteration 20: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.10, Number of inter- and intra-triplets = 2584616, 0_0_5213_0_8012
Epoch 7 Iteration 30: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.11, Number of inter- and intra-triplets = 2619172, 0_0_6209_0_4837
best acc: 0.7, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 8 Iteration 0: Loss = 0.09, inter:0.11, intra:0.07-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.12,0.10, Number of inter- and intra-triplets = 2511070, 0_0_11993_2882_9172
Epoch 8 Iteration 10: Loss = 0.09, inter:0.11, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2299744, 0_0_12813_0_4531
Epoch 8 Iteration 20: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.0,0.0,0.10, Number of inter- and intra-triplets = 2723412, 0_0_0_0_15932
Epoch 8 Iteration 30: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.0, Number of inter- and intra-triplets = 2224052, 0_0_12789_0_0
best acc: 0.7999999999999999, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 9 Iteration 0: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.09, Number of inter- and intra-triplets = 2254840, 0_0_4274_0_4082
Epoch 9 Iteration 10: Loss = 0.09, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.10, Number of inter- and intra-triplets = 2291091, 0_0_4653_0_3312
Epoch 9 Iteration 20: Loss = 0.09, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2186303, 0_0_7153_0_6524
Epoch 9 Iteration 30: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.12,0.0,0.10, Number of inter- and intra-triplets = 1892027, 0_0_7540_0_2801
best acc: 0.8999999999999999, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 10 Iteration 0: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 1843962, 0_0_4874_0_5759
Epoch 10 Iteration 10: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.08,0.0,0.10, Number of inter- and intra-triplets = 2738613, 0_0_4186_0_4148
Epoch 10 Iteration 20: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.10, Number of inter- and intra-triplets = 1878660, 0_0_6517_0_2060
Epoch 10 Iteration 30: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 1805508, 0_0_9492_0_2244
best acc: 0.9999999999999999, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 11 Iteration 0: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.10, Number of inter- and intra-triplets = 2089381, 0_0_12981_0_7385
Epoch 11 Iteration 10: Loss = 0.09, inter:0.10, intra:0.07-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.09,0.13,0.0,0.10, Number of inter- and intra-triplets = 1871384, 0_1890_6004_0_3141
Epoch 11 Iteration 20: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.09,0.0,0.10, Number of inter- and intra-triplets = 2026326, 0_0_3746_0_9350
Epoch 11 Iteration 30: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2170889, 0_0_18561_0_5700
best acc: 1.0999999999999999, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 12 Iteration 0: Loss = 0.08, inter:0.10, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.12,0.0,0.10, Number of inter- and intra-triplets = 3045163, 0_0_7440_0_5160
Epoch 12 Iteration 10: Loss = 0.09, inter:0.10, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.13,0.0,0.11, Number of inter- and intra-triplets = 2254804, 0_0_5067_0_6886
Epoch 12 Iteration 20: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.12,0.0,0.09, Number of inter- and intra-triplets = 2077713, 0_0_13929_0_5422
Epoch 12 Iteration 30: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.09, Number of inter- and intra-triplets = 1852085, 0_0_16308_0_7140
best acc: 1.2, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 13 Iteration 0: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.09,0.0,0.09, Number of inter- and intra-triplets = 1809348, 0_0_11009_0_5991
Epoch 13 Iteration 10: Loss = 0.09, inter:0.10, intra:0.07-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.12,0.10, Number of inter- and intra-triplets = 1828099, 0_0_10968_2997_7106
Epoch 13 Iteration 20: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2327080, 0_0_21176_0_5527
Epoch 13 Iteration 30: Loss = 0.08, inter:0.10, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.13,0.0,0.09, Number of inter- and intra-triplets = 1767516, 0_0_1971_0_5240
best acc: 1.3, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 14 Iteration 0: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.10, Number of inter- and intra-triplets = 2103619, 0_0_7562_0_8211
Epoch 14 Iteration 10: Loss = 0.08, inter:0.10, intra:0.05-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.12,0.0,0.10, Number of inter- and intra-triplets = 1626081, 0_0_9914_0_7513
Epoch 14 Iteration 20: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.10, Number of inter- and intra-triplets = 2305027, 0_0_12376_0_8093
Epoch 14 Iteration 30: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2097939, 0_0_9688_0_4703
best acc: 1.4000000000000001, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt
Epoch 15 Iteration 0: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.11,0.0,0.10, Number of inter- and intra-triplets = 2075917, 0_0_15995_0_3994
Epoch 15 Iteration 10: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.09, Number of inter- and intra-triplets = 1978536, 0_0_4865_0_5851
Epoch 15 Iteration 20: Loss = 0.09, inter:0.10, intra:0.06-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.10,0.10,0.0,0.09, Number of inter- and intra-triplets = 1710514, 0_2030_15859_0_7314
Epoch 15 Iteration 30: Loss = 0.08, inter:0.10, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.10,0.0,0.10, Number of inter- and intra-triplets = 2001140, 0_0_14566_0_10373
best acc: 1.5000000000000002, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_02_w07.pt

- iiv_conv2d_anchorFalse_02_01_w07_-1 
Start step 3: train IIV embedding.
Epoch 1 Iteration 0: Loss = 0.13, inter:0.18, intra:0.03-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.08,0.0,0.08, Number of inter- and intra-triplets = 11259374, 0_0_19951_0_13788
Epoch 1 Iteration 10: Loss = 0.14, inter:0.19, intra:0.04-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.09,0.0,0.08, Number of inter- and intra-triplets = 9795824, 0_0_16341_0_9380
Epoch 1 Iteration 20: Loss = 0.10, inter:0.13, intra:0.03-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.06,0.0,0.06, Number of inter- and intra-triplets = 4738715, 0_0_11550_0_8737
Epoch 1 Iteration 30: Loss = 0.09, inter:0.12, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.06,0.0,0.05, Number of inter- and intra-triplets = 3323142, 0_0_1454_0_2142
best acc: 0.1, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 2 Iteration 0: Loss = 0.09, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2742810, 0_0_4132_0_2512
Epoch 2 Iteration 10: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2283620, 0_0_3019_0_5015
Epoch 2 Iteration 20: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2503407, 0_0_3953_0_3584
Epoch 2 Iteration 30: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2426785, 0_0_2698_0_1714
best acc: 0.2, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 3 Iteration 0: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2416777, 0_0_5650_0_2424
Epoch 3 Iteration 10: Loss = 0.09, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2349904, 0_0_4420_0_5814
Epoch 3 Iteration 20: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2418340, 0_0_4201_0_5514
Epoch 3 Iteration 30: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2261880, 0_0_5595_0_2148
best acc: 0.30000000000000004, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 4 Iteration 0: Loss = 0.09, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2411115, 0_0_7482_0_4418
Epoch 4 Iteration 10: Loss = 0.09, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2352715, 0_0_5460_0_1939
Epoch 4 Iteration 20: Loss = 0.09, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2367081, 0_0_15256_0_6790
Epoch 4 Iteration 30: Loss = 0.09, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.06,0.0,0.04, Number of inter- and intra-triplets = 2676184, 0_0_16263_0_1361
best acc: 0.4, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 5 Iteration 0: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.06, Number of inter- and intra-triplets = 2610915, 0_0_8775_0_1501
Epoch 5 Iteration 10: Loss = 0.09, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2467641, 0_0_3802_0_3178
Epoch 5 Iteration 20: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2570078, 0_0_5689_0_4806
Epoch 5 Iteration 30: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2222852, 0_0_12916_0_3898
best acc: 0.5, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 6 Iteration 0: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2200683, 0_0_10953_0_2195
Epoch 6 Iteration 10: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2272674, 0_0_3672_0_1728
Epoch 6 Iteration 20: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2123060, 0_0_10989_0_1681
Epoch 6 Iteration 30: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2414520, 0_0_12723_0_702
best acc: 0.6, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 7 Iteration 0: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.04, Number of inter- and intra-triplets = 2954320, 0_0_7143_0_2093
Epoch 7 Iteration 10: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.04,0.0,0.05, Number of inter- and intra-triplets = 2231923, 0_0_4839_0_1827
Epoch 7 Iteration 20: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2828027, 0_0_11595_0_3052
Epoch 7 Iteration 30: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.04, Number of inter- and intra-triplets = 2355717, 0_0_4210_0_3936
best acc: 0.7, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 8 Iteration 0: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2337535, 0_0_6469_0_4793
Epoch 8 Iteration 10: Loss = 0.08, inter:0.11, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 5008857, 0_0_6131_0_6324
Epoch 8 Iteration 20: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.04,0.0,0.05, Number of inter- and intra-triplets = 2605376, 0_0_3128_0_3483
Epoch 8 Iteration 30: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2125819, 0_0_3542_0_2418
best acc: 0.7999999999999999, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 9 Iteration 0: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2609372, 0_0_6247_0_2588
Epoch 9 Iteration 10: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.04, Number of inter- and intra-triplets = 2151084, 0_0_17335_0_2922
Epoch 9 Iteration 20: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2330573, 0_0_4981_0_2860
Epoch 9 Iteration 30: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2032185, 0_0_17634_0_4711
best acc: 0.8999999999999999, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 10 Iteration 0: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2329389, 0_0_3971_0_1035
Epoch 10 Iteration 10: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.04,0.0,0.05, Number of inter- and intra-triplets = 2144343, 0_0_1952_0_4146
Epoch 10 Iteration 20: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 1887483, 0_0_13245_0_3964
Epoch 10 Iteration 30: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 3899612, 0_0_10129_0_4090
best acc: 0.9999999999999999, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 11 Iteration 0: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2118057, 0_0_4365_0_1276
Epoch 11 Iteration 10: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2819755, 0_0_3297_0_2210
Epoch 11 Iteration 20: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.06,0.0,0.05, Number of inter- and intra-triplets = 3003979, 0_0_12702_0_10904
Epoch 11 Iteration 30: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.04, Number of inter- and intra-triplets = 2660405, 0_0_3948_0_8648
best acc: 1.0999999999999999, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 12 Iteration 0: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.04,0.0,0.05, Number of inter- and intra-triplets = 2051763, 0_0_3352_0_6157
Epoch 12 Iteration 10: Loss = 0.08, inter:0.10, intra:0.03-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.05,0.05, Number of inter- and intra-triplets = 1677936, 0_0_8199_1433_3816
Epoch 12 Iteration 20: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.04, Number of inter- and intra-triplets = 2004561, 0_0_5093_0_2515
Epoch 12 Iteration 30: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2021501, 0_0_10176_0_6633
best acc: 1.2, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 13 Iteration 0: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2019825, 0_0_1659_0_6805
Epoch 13 Iteration 10: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 1775256, 0_0_6445_0_4643
Epoch 13 Iteration 20: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2030143, 0_0_5173_0_7096
Epoch 13 Iteration 30: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.04,0.0,0.05, Number of inter- and intra-triplets = 1635116, 0_0_852_0_2444
best acc: 1.3, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 14 Iteration 0: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.06,0.0,0.05, Number of inter- and intra-triplets = 3313820, 0_0_12518_0_5619
Epoch 14 Iteration 10: Loss = 0.07, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 2135878, 0_0_8843_0_3799
Epoch 14 Iteration 20: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.06,0.0,0.04, Number of inter- and intra-triplets = 2187321, 0_0_13090_0_5539
Epoch 14 Iteration 30: Loss = 0.07, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.04, Number of inter- and intra-triplets = 1662740, 0_0_3574_0_1506
best acc: 1.4000000000000001, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Epoch 15 Iteration 0: Loss = 0.07, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.04,0.0,0.04, Number of inter- and intra-triplets = 1995465, 0_0_4434_0_3672
Epoch 15 Iteration 10: Loss = 0.08, inter:0.10, intra:0.02-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.05, Number of inter- and intra-triplets = 1652974, 0_0_4211_0_1076
Epoch 15 Iteration 20: Loss = 0.07, inter:0.10, intra:0.01-> Angry,Surprise,Sad,Neutral,Happy : 0.0,0.0,0.05,0.0,0.0, Number of inter- and intra-triplets = 1675503, 0_0_6418_0_0
Epoch 15 Iteration 30: Loss = 0.08, inter:0.10, intra:0.03-> Angry,Surprise,Sad,Neutral,Happy : 0.05,0.0,0.05,0.0,0.04, Number of inter- and intra-triplets = 1974982, 2839_0_4379_0_2455
best acc: 1.5000000000000002, save to /home/rosen/project/FastSpeech2/IIV/iiv_conv2d_anchorFalse_02_01_w07.pt
Start step 4: save IIV embedding by best model.
Start step 5: visualize IIV embedding.


- 

## IIVTTS training best result
- Hyper, loss, and syntheized audio with different emo/cluster
  - The emo/cluster vector

## Other
- Opensmile cluster

In total, 62 parameters are contained in the Geneva
Minimalistic Standard Parameter Set.

- Name of useful prosody
  - **logRelF0**: 
  The measure H1-H2, the difference in amplitude between the first and second harmonics, is frequently used to distinguish phonation types and to characterize differences across voices and genders. While H1-H2 can differentiate voices and is used by listeners to perceive changes in voice quality, its relation to voice articulation is less straightforward. 
  - F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope (**F0_riseSlope_SD**)
  - F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope (**F0_riseSlope_mean**)
    - mean and standard deviation of the slope of rising/falling signal parts
  - **slopeV500 - 1500_sma3nz_stddevNorm**
  linear regression slope of the logarithmic power spectrum within the two given bands.

- reference paper of opensmile (page 5):
https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf

- Spectral slope description powerpoint
https://www.isca-speech.org/archive_open/archive_papers/spkd2008/material/spkd_007_p.pdf