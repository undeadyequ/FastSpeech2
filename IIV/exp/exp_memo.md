## Problem
- The number of intra triplets is too small.
  - when batch = 100, all_samples > 10000, emo=5, grp=3
    - 46, 0, 0, 0 (emo1 -> emo5)

emos = []
groups = []


-> check samples distribution in each group for every emotion
- Problem: The intra loss is not decreased at all
  - When joint train inter and intra loss, the inter decrease but intra increase
- Maybe
  - The group labels is wrong that similar embeddings are assigned different grp label
  - The samples in certain grp is compleltely more than others
- Solution
  - check emo_group_stats, emo_group_psdSDmin3

emo_groups = {
"emo1": {"0": (num0, distance_sum), "1": (num1, distance_sum)},
}


emo_group_psdSDmin3 = {
"emo1": {"0": ["pitch", "energy", "?"], "1": (num1, distance_sum)},
}


## 2 fc, (batch_size = 512), Inter-triplet
Epoch 1 Iteration 0: Loss = 0.18(inter:0.18, intra:0.00), Number of inter- and intra-triplets = 11404398, 0
Epoch 1 Iteration 10: Loss = 0.16(inter:0.16, intra:0.00), Number of inter- and intra-triplets = 8433786, 0
Epoch 1 Iteration 20: Loss = 0.12(inter:0.12, intra:0.00), Number of inter- and intra-triplets = 3297868, 0
Epoch 1 Iteration 30: Loss = 0.12(inter:0.12, intra:0.00), Number of inter- and intra-triplets = 3463297, 0
best acc: 0.1, save to best_iiv_model.pt
Epoch 2 Iteration 0: Loss = 0.12(inter:0.12, intra:0.00), Number of inter- and intra-triplets = 3455640, 0
Epoch 2 Iteration 10: Loss = 0.12(inter:0.12, intra:0.00), Number of inter- and intra-triplets = 3621649, 0
Epoch 2 Iteration 20: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2329681, 0
Epoch 2 Iteration 30: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2174929, 0
best acc: 0.2, save to best_iiv_model.pt
Epoch 3 Iteration 0: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 3089341, 0
Epoch 3 Iteration 10: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2656713, 0
Epoch 3 Iteration 20: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2375413, 0
Epoch 3 Iteration 30: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2367657, 0
best acc: 0.30000000000000004, save to best_iiv_model.pt
Epoch 4 Iteration 0: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2451508, 0
Epoch 4 Iteration 10: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2395535, 0
Epoch 4 Iteration 20: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2284877, 0
Epoch 4 Iteration 30: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2551647, 0
best acc: 0.4, save to best_iiv_model.pt
Epoch 5 Iteration 0: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2849899, 0
Epoch 5 Iteration 10: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2290894, 0
Epoch 5 Iteration 20: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2534636, 0
Epoch 5 Iteration 30: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2409625, 0
best acc: 0.5, save to best_iiv_model.pt
Epoch 6 Iteration 0: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2184762, 0
Epoch 6 Iteration 10: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2755791, 0


## 2 fc, (batch_size = 512), Inter- and Intra-triplet 
Epoch 1 Iteration 0: Loss = 0.19(inter:0.18, intra:0.20), Number of inter- and intra-triplets = 11368062, 15398_30786_31505_3704_17924
Epoch 1 Iteration 10: Loss = 0.19(inter:0.18, intra:0.21), Number of inter- and intra-triplets = 10633923, 23987_31665_13068_25463_28954
Epoch 1 Iteration 20: Loss = 0.16(inter:0.13, intra:0.21), Number of inter- and intra-triplets = 7501993, 17766_13979_0_6301_19346
Epoch 1 Iteration 30: Loss = 0.20(inter:0.13, intra:0.31), Number of inter- and intra-triplets = 7225395, 10973_32391_5843_12012_18121
best acc: 0.1, save to best_iiv_model.pt
Epoch 2 Iteration 0: Loss = 0.18(inter:0.13, intra:0.27), Number of inter- and intra-triplets = 7004820, 23910_22965_5660_6368_13191
Epoch 2 Iteration 10: Loss = 0.22(inter:0.12, intra:0.38), Number of inter- and intra-triplets = 4851359, 13881_19161_9458_10142_7751
Epoch 2 Iteration 20: Loss = 0.22(inter:0.12, intra:0.37), Number of inter- and intra-triplets = 5563781, 6903_35358_4473_9182_17255
Epoch 2 Iteration 30: Loss = 0.23(inter:0.12, intra:0.39), Number of inter- and intra-triplets = 5460853, 11499_27941_20653_12035_12109
best acc: 0.2, save to best_iiv_model.pt
Epoch 3 Iteration 0: Loss = 0.21(inter:0.12, intra:0.35), Number of inter- and intra-triplets = 5643263, 23688_15540_10439_15005_9304
Epoch 3 Iteration 10: Loss = 0.22(inter:0.12, intra:0.36), Number of inter- and intra-triplets = 5612283, 17255_5678_13829_1239_5896
Epoch 3 Iteration 20: Loss = 0.22(inter:0.11, intra:0.39), Number of inter- and intra-triplets = 5837459, 23543_7554_5755_12764_19175
Epoch 3 Iteration 30: Loss = 0.21(inter:0.11, intra:0.37), Number of inter- and intra-triplets = 5557524, 8210_1855_14771_5366_15296
best acc: 0.30000000000000004, save to best_iiv_model.pt
Epoch 4 Iteration 0: Loss = 0.17(inter:0.11, intra:0.27), Number of inter- and intra-triplets = 6038013, 11578_31780_4962_13558_0
Epoch 4 Iteration 10: Loss = 0.21(inter:0.11, intra:0.36), Number of inter- and intra-triplets = 5838693, 32551_19147_27843_4602_2352
Epoch 4 Iteration 20: Loss = 0.20(inter:0.11, intra:0.33), Number of inter- and intra-triplets = 6378934, 18550_7244_9791_15099_14871
Epoch 4 Iteration 30: Loss = 0.22(inter:0.11, intra:0.38), Number of inter- and intra-triplets = 5769436, 24859_4468_8021_20489_21398
best acc: 0.4, save to best_iiv_model.pt
Epoch 5 Iteration 0: Loss = 0.20(inter:0.11, intra:0.34), Number of inter- and intra-triplets = 6275005, 23537_13045_10091_10827_17444
Epoch 5 Iteration 10: Loss = 0.21(inter:0.11, intra:0.36), Number of inter- and intra-triplets = 5732456, 25107_27299_8319_14710_10515
Epoch 5 Iteration 20: Loss = 0.19(inter:0.11, intra:0.32), Number of inter- and intra-triplets = 6353906, 32921_14656_10389_28894_23174
Epoch 5 Iteration 30: Loss = 0.20(inter:0.11, intra:0.35), Number of inter- and intra-triplets = 6030640, 14350_20850_14840_14434_22015
best acc: 0.5, save to best_iiv_model.pt
Epoch 6 Iteration 0: Loss = 0.21(inter:0.11, intra:0.35), Number of inter- and intra-triplets = 6542539, 10909_17881_14894_11075_9528
Epoch 6 Iteration 10: Loss = 0.22(inter:0.11, intra:0.40), Number of inter- and intra-triplets = 4872440, 13742_5084_19838_16074_12388
Epoch 6 Iteration 20: Loss = 0.21(inter:0.11, intra:0.37), Number of inter- and intra-triplets = 5577356, 16634_21241_6605_16788_22294
Epoch 6 Iteration 30: Loss = 0.22(inter:0.11, intra:0.39), Number of inter- and intra-triplets = 5300348, 50879_10565_32645_20500_19522
...

Epoch 17 Iteration 0: Loss = 0.23(inter:0.10, intra:0.43), Number of inter- and intra-triplets = 3584935, 10184_11404_8941_4546_11474
Epoch 17 Iteration 10: Loss = 0.21(inter:0.10, intra:0.37), Number of inter- and intra-triplets = 4716215, 18725_29475_17447_25669_23846
Epoch 17 Iteration 20: Loss = 0.17(inter:0.10, intra:0.29), Number of inter- and intra-triplets = 4819188, 20962_29517_10528_0_12137
Epoch 17 Iteration 30: Loss = 0.25(inter:0.10, intra:0.47), Number of inter- and intra-triplets = 3464055, 17437_36150_23193_14968_15539

## same condition, show intra-loss for each emotion
Epoch 1 Iteration 0: Loss = 0.18(inter:0.17, intra:0.20->0.20_0.19_0.20_0.19_0.19), Number of inter- and intra-triplets = 11295553, 16187_45170_34730_23010_53629
Epoch 1 Iteration 10: Loss = 0.20(inter:0.20, intra:0.20->0.19_0.20_0.20_0.20_0.20), Number of inter- and intra-triplets = 10991211, 28267_35545_5414_3905_55706
Epoch 1 Iteration 20: Loss = 0.20(inter:0.20, intra:0.20->0.19_0.20_0.20_0.20_0.20), Number of inter- and intra-triplets = 11063033, 14274_26805_28699_29056_32192
Epoch 1 Iteration 30: Loss = 0.20(inter:0.20, intra:0.20->0.20_0.20_0.20_0.20_0.20), Number of inter- and intra-triplets = 11095956, 47048_9535_25906_24385_34468
best acc: 0.1, save to best_iiv_model.pt
Epoch 2 Iteration 0: Loss = 0.20(inter:0.20, intra:0.20->0.20_0.20_0.19_0.20_0.20), Number of inter- and intra-triplets = 10900384, 40369_28778_23333_10428_25303
Epoch 2 Iteration 10: Loss = 0.20(inter:0.20, intra:0.20->0.20_0.20_0.20_0.20_0.19), Number of inter- and intra-triplets = 10996880, 28193_43732_10658_26635_7812
Epoch 2 Iteration 20: Loss = 0.18(inter:0.20, intra:0.16->0.20_0.20_0.0_0.20_0.20), Number of inter- and intra-triplets = 10936117, 13572_47950_0_20345_23673
Epoch 2 Iteration 30: Loss = 0.20(inter:0.20, intra:0.20->0.20_0.20_0.20_0.20_0.20), Number of inter- and intra-triplets = 11054144, 8540_10935_17966_7560_28546
best acc: 0.2, save to best_iiv_model.pt
Epoch 3 Iteration 0: Loss = 0.20(inter:0.20, intra:0.20->0.19_0.20_0.20_0.19_0.20), Number of inter- and intra-triplets = 10823009, 35662_17410_32263_20484_26784
Epoch 3 Iteration 10: Loss = 0.20(inter:0.19, intra:0.20->0.20_0.19_0.20_0.20_0.20), Number of inter- and intra-triplets = 10848449, 26460_18945_15548_3922_19833
Epoch 3 Iteration 20: Loss = 0.20(inter:0.18, intra:0.23->0.22_0.22_0.22_0.22_0.23), Number of inter- and intra-triplets = 10065133, 26097_37434_22807_8529_25409
Epoch 3 Iteration 30: Loss = 0.22(inter:0.12, intra:0.36->0.37_0.36_0.35_0.35_0.36), Number of inter- and intra-triplets = 4891044, 12381_17749_22350_6008_13628
best acc: 0.30000000000000004, save to best_iiv_model.pt
Epoch 4 Iteration 0: Loss = 0.30(inter:0.12, intra:0.58->0.58_0.58_0.57_0.56_0.57), Number of inter- and intra-triplets = 3477770, 7100_13210_7310_9681_7586
Epoch 4 Iteration 10: Loss = 0.20(inter:0.12, intra:0.32->0.31_0.32_0.32_0.31_0.32), Number of inter- and intra-triplets = 6266759, 28047_29853_2554_11779_25480
Epoch 4 Iteration 20: Loss = 0.18(inter:0.12, intra:0.26->0.32_0.32_0.32_0.0_0.31), Number of inter- and intra-triplets = 6306100, 13479_5808_14477_0_6259
Epoch 4 Iteration 30: Loss = 0.24(inter:0.12, intra:0.42->0.40_0.40_0.43_0.42_0.41), Number of inter- and intra-triplets = 5044606, 19356_19151_8597_19353_12069
best acc: 0.4, save to best_iiv_model.pt

## Remove inter-triplet (w=0)



conv1d(kernal=3) LSTM FC
/home/rosen/anaconda3/envs/fastts/bin/python /home/rosen/project/FG-transformer-TTS/IIV/train.py 
Epoch 1 Iteration 0: Loss = 0.20(inter:0.20, intra:0.00), Number of inter- and intra-triplets = 5659847, 0
Epoch 1 Iteration 10: Loss = 0.15(inter:0.15, intra:0.00), Number of inter- and intra-triplets = 7505871, 0
Epoch 1 Iteration 20: Loss = 0.12(inter:0.12, intra:0.00), Number of inter- and intra-triplets = 3165189, 0
Epoch 1 Iteration 30: Loss = 0.12(inter:0.12, intra:0.00), Number of inter- and intra-triplets = 3006831, 0
best acc: 0.1, save to best_iiv_model.pt
Epoch 2 Iteration 0: Loss = 0.12(inter:0.12, intra:0.00), Number of inter- and intra-triplets = 3396729, 0
Epoch 2 Iteration 10: Loss = 0.12(inter:0.12, intra:0.00), Number of inter- and intra-triplets = 2296147, 0
Epoch 2 Iteration 20: Loss = 0.12(inter:0.12, intra:0.00), Number of inter- and intra-triplets = 2110273, 0
Epoch 2 Iteration 30: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2121877, 0
best acc: 0.2, save to best_iiv_model.pt
Epoch 3 Iteration 0: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 1952355, 0
Epoch 3 Iteration 10: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2020191, 0
Epoch 3 Iteration 20: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 1932578, 0
Epoch 3 Iteration 30: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2016080, 0
best acc: 0.30000000000000004, save to best_iiv_model.pt
Epoch 4 Iteration 0: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 1919186, 0
Epoch 4 Iteration 10: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 1910023, 0
Epoch 4 Iteration 20: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 1863267, 0
Epoch 4 Iteration 30: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2137259, 0
best acc: 0.4, save to best_iiv_model.pt
Epoch 5 Iteration 0: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2125391, 0
Epoch 5 Iteration 10: Loss = 0.12(inter:0.12, intra:0.00), Number of inter- and intra-triplets = 1872001, 0
Epoch 5 Iteration 20: Loss = 0.20(inter:0.20, intra:0.00), Number of inter- and intra-triplets = 2628218, 0
Epoch 5 Iteration 30: Loss = 0.20(inter:0.20, intra:0.00), Number of inter- and intra-triplets = 2281794, 0
best acc: 0.5, save to best_iiv_model.pt
Epoch 6 Iteration 0: Loss = 0.20(inter:0.20, intra:0.00), Number of inter- and intra-triplets = 2265173, 0
Epoch 6 Iteration 10: Loss = 0.12(inter:0.12, intra:0.00), Number of inter- and intra-triplets = 3556647, 0
Epoch 6 Iteration 20: Loss = 0.11(inter:0.11, intra:0.00), Number of inter- and intra-triplets = 2469718, 0
Epoch 6 Iteration 30: Loss = 0.12(inter:0.12, intra:0.00), Number of inter- and intra-triplets = 4243181, 0
best acc: 0.6, save to best_iiv_model.pt

