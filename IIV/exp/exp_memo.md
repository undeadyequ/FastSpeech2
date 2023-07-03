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


## New group (w = 0.7)

Epoch 1 Iteration 0: Loss = 0.16(inter:0.18, intra:0.12->0.20,0.0,0.19,0.0,0.20, Number of inter- and intra-triplets = 11237374, 21598_0_120343_0_30829
Epoch 1 Iteration 10: Loss = 0.21(inter:0.12, intra:0.42->0.72,0.0,0.71,0.0,0.66, Number of inter- and intra-triplets = 3269429, 13369_0_47210_0_3041
Epoch 1 Iteration 20: Loss = 0.18(inter:0.12, intra:0.32->0.54,0.0,0.54,0.0,0.53, Number of inter- and intra-triplets = 3478139, 4786_0_47283_0_9837
Epoch 1 Iteration 30: Loss = 0.17(inter:0.12, intra:0.30->0.50,0.0,0.49,0.0,0.49, Number of inter- and intra-triplets = 4278858, 26114_0_66706_0_6998
best acc: 0.1, save to best_iiv_model.pt
Epoch 2 Iteration 0: Loss = 0.18(inter:0.12, intra:0.33->0.54,0.0,0.56,0.0,0.54, Number of inter- and intra-triplets = 3874248, 31060_0_54221_0_6111
Epoch 2 Iteration 10: Loss = 0.16(inter:0.12, intra:0.27->0.45,0.0,0.46,0.0,0.43, Number of inter- and intra-triplets = 4608396, 20919_0_43846_0_7842
Epoch 2 Iteration 20: Loss = 0.18(inter:0.11, intra:0.36->0.59,0.0,0.60,0.0,0.61, Number of inter- and intra-triplets = 3402052, 14658_0_95871_0_4217
Epoch 2 Iteration 30: Loss = 0.18(inter:0.11, intra:0.34->0.56,0.0,0.57,0.0,0.57, Number of inter- and intra-triplets = 3504339, 18853_0_48437_0_14788
best acc: 0.2, save to best_iiv_model.pt
Epoch 3 Iteration 0: Loss = 0.21(inter:0.11, intra:0.46->0.57,0.0,0.58,0.52,0.59, Number of inter- and intra-triplets = 3005826, 23704_0_45897_1791_16674
Epoch 3 Iteration 10: Loss = 0.18(inter:0.11, intra:0.32->0.53,0.0,0.54,0.0,0.53, Number of inter- and intra-triplets = 4114127, 18338_0_56772_0_17362
Epoch 3 Iteration 20: Loss = 0.17(inter:0.11, intra:0.31->0.49,0.0,0.50,0.0,0.53, Number of inter- and intra-triplets = 4418910, 12886_0_90112_0_4117
Epoch 3 Iteration 30: Loss = 0.18(inter:0.11, intra:0.34->0.55,0.0,0.54,0.0,0.58, Number of inter- and intra-triplets = 3595615, 12526_0_72883_0_4403
best acc: 0.30000000000000004, save to best_iiv_model.pt
Epoch 4 Iteration 0: Loss = 0.18(inter:0.11, intra:0.36->0.58,0.0,0.58,0.0,0.60, Number of inter- and intra-triplets = 3496100, 26154_0_82128_0_14953
Epoch 4 Iteration 10: Loss = 0.17(inter:0.11, intra:0.32->0.55,0.0,0.51,0.0,0.50, Number of inter- and intra-triplets = 3977006, 13330_0_71173_0_40010
Epoch 4 Iteration 20: Loss = 0.18(inter:0.11, intra:0.33->0.55,0.0,0.56,0.0,0.54, Number of inter- and intra-triplets = 3392794, 20595_0_42396_0_11967
Epoch 4 Iteration 30: Loss = 0.18(inter:0.11, intra:0.36->0.61,0.0,0.60,0.0,0.56, Number of inter- and intra-triplets = 3082396, 29020_0_93998_0_10505
best acc: 0.4, save to best_iiv_model.pt
Epoch 5 Iteration 0: Loss = 0.17(inter:0.11, intra:0.33->0.54,0.0,0.56,0.0,0.53, Number of inter- and intra-triplets = 3551547, 23222_0_49322_0_17822
Epoch 5 Iteration 10: Loss = 0.17(inter:0.11, intra:0.32->0.53,0.0,0.52,0.0,0.53, Number of inter- and intra-triplets = 3348796, 22535_0_76238_0_2854
Epoch 5 Iteration 20: Loss = 0.19(inter:0.11, intra:0.39->0.64,0.0,0.63,0.0,0.66, Number of inter- and intra-triplets = 2801114, 22585_0_83403_0_11785
Epoch 5 Iteration 30: Loss = 0.19(inter:0.11, intra:0.37->0.64,0.0,0.60,0.0,0.61, Number of inter- and intra-triplets = 2975577, 19554_0_48908_0_12782
best acc: 0.5, save to best_iiv_model.pt
Epoch 6 Iteration 0: Loss = 0.19(inter:0.11, intra:0.37->0.62,0.0,0.63,0.0,0.57, Number of inter- and intra-triplets = 3186248, 12473_0_50390_0_10965
Epoch 6 Iteration 10: Loss = 0.18(inter:0.10, intra:0.34->0.55,0.0,0.57,0.0,0.58, Number of inter- and intra-triplets = 3207737, 27192_0_63590_0_21796
Epoch 6 Iteration 20: Loss = 0.16(inter:0.11, intra:0.29->0.49,0.0,0.48,0.0,0.48, Number of inter- and intra-triplets = 4633081, 27860_0_106636_0_13776
Epoch 6 Iteration 30: Loss = 0.17(inter:0.11, intra:0.33->0.51,0.0,0.57,0.0,0.58, Number of inter- and intra-triplets = 3584798, 23843_0_65101_0_18974
best acc: 0.6, save to best_iiv_model.pt
Epoch 7 Iteration 0: Loss = 0.19(inter:0.10, intra:0.38->0.63,0.0,0.62,0.0,0.62, Number of inter- and intra-triplets = 2723081, 27732_0_60275_0_12425
Epoch 7 Iteration 10: Loss = 0.18(inter:0.10, intra:0.35->0.56,0.0,0.60,0.0,0.60, Number of inter- and intra-triplets = 2858957, 26880_0_79539_0_4634
Epoch 7 Iteration 20: Loss = 0.18(inter:0.10, intra:0.37->0.63,0.0,0.62,0.0,0.59, Number of inter- and intra-triplets = 2545893, 16491_0_63113_0_14130
Epoch 7 Iteration 30: Loss = 0.17(inter:0.10, intra:0.31->0.51,0.0,0.52,0.0,0.51, Number of inter- and intra-triplets = 3956288, 21272_0_99660_0_7570
best acc: 0.7, save to best_iiv_model.pt
Epoch 8 Iteration 0: Loss = 0.18(inter:0.10, intra:0.35->0.58,0.0,0.56,0.0,0.61, Number of inter- and intra-triplets = 3003139, 18197_0_81951_0_11317
Epoch 8 Iteration 10: Loss = 0.14(inter:0.10, intra:0.23->0.60,0.0,0.57,0.0,0.0, Number of inter- and intra-triplets = 3279332, 26498_0_70719_0_0
Epoch 8 Iteration 20: Loss = 0.16(inter:0.10, intra:0.28->0.47,0.0,0.48,0.0,0.46, Number of inter- and intra-triplets = 3616442, 16062_0_73374_0_16922
Epoch 8 Iteration 30: Loss = 0.17(inter:0.10, intra:0.34->0.57,0.0,0.55,0.0,0.56, Number of inter- and intra-triplets = 2937035, 28178_0_57102_0_19375
best acc: 0.7999999999999999, save to best_iiv_model.pt




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


