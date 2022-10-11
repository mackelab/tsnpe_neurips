First round:

- 30k sims
- 35 features
- ensemble of 10
- path: 2022_04_13__19_32_02_snpe_m/1
- evaluation run: 2022_04_14__09_33_37__multirun/1__2022_04_13__19_32_02_snpe_m/1

R2:
- 30k sims
- 35 features
- ensemble of 10 (with joblib)
- path: 2022_04_18__15_16_19_snpe_m/0
- evaluation run: 2022_04_18__16_22_41__multirun/0__2022_04_18__15_16_19_snpe_m/0

R3:
- 30k sims
- 35 features
- ensemble of 10 (with joblib)
- path: 2022_04_19__10_27_01_snpe_m/0
- evaluation run: 2022_04_19__10_56_46__multirun/0__2022_04_19__10_27_01_snpe_m/0

R4:
- 30k sims
- 35 features
- ensemble of 10 (with joblib)
- path: 2022_04_19__20_55_09_snpe_m/0
- evaluation run: 2022_04_19__21_41_14__multirun/0__2022_04_19__20_55_09_snpe_m/0

R5:
- 30k sims
- 35 features
- ensemble of 10 (with joblib)
- path: 2022_04_20__10_27_58_snpe_m/0
- evaluation run: 2022_04_20__11_10_51__multirun/0__2022_04_20__10_27_58_snpe_m/0

R6:
- path: 2022_04_23__07_38_14_snpe/0

R7:
- 2022_04_24__07_47_31_snpe

R8:
- 2022_04_25__07_21_21_snpe









Run 2 (l20_2):
R1:
- 30k sims
- 35 features
- ensemble of 10
- path: previous_inference=2022_04_27__19_06_37_snpe
- evaluation run: 2022_04_27__23_41_19__2022_04_27__19_06_37_snpe


R2: 
- disregard the samples from the first round and train from scratch
- 30k sims
- 35 features
- ensemble of 10
- path: 2022_04_28__08_15_09_snpe
- evaluation run: 2022_04_28__10_57_24__multirun

R3:
- uses samples from second round (but not first) and does not train from scratch
- 30k sims
- 35 features
- ensemble of 10
- path: 2022_04_28__22_30_29_snpe_m/0
- evaluation run: 2022_04_29__08_16_57__multirun/0__2022_04_28__22_30_29_snpe_m/0

R4:
- started
- 30k sims
- 35 features
- ensemble of 10
- path: 2022_04_29__17_11_50_snpe_m or 2022_04_30__22_00_21_snpe_m/0

R5:
- path: 2022_05_01__12_02_16_snpe_m
- evaluation: 2022_05_01__17_19_46__multirun

R6:
- path: 2022_05_02__19_31_58_snpe_m
- evaluation: 




Run 3 (l20_3): SNPE-C with Flow trained in constrained space
R1:
- 30k sims
- 35 features
- ensemble of 10
- path: previous_inference=2022_04_27__19_06_37_snpe
- evaluation run: 2022_04_27__23_41_19__2022_04_27__19_06_37_snpe

R2:
Failed with only R2 data and train from scratch:
- 2022_04_29__08_52_03_snpe_m

R2:
Succeeded when using data from both rounds and continuing training from R1:
- ensemble size=1
- poor coverage
- 84% good simulations
- path: 2022_04_29__10_02_26_snpe
- evaluation path: 2022_04_29__11_01_35__multirun/0__2022_04_29__10_02_26_snpe










Run 4 (l20_4): SNPE-C with flow in unconstrained space and maf density estimator
R1:
- 30k sims
- 35 features
- ensemble of 1
- path: 2022_04_29__14_05_07_snpe
- evaluation path: 2022_04_29__14_34_57__multirun/0__2022_04_29__14_05_07_snpe
- 5% good simulations
- posterior log-prob 21.008

R2:
- performed on simultions from l20_3
- 30k sims
- 35 features
- ensemble of 1
- path: 2022_04_29__15_04_42_snpe_m/0
- evaluation path: 2022_04_29__16_16_33__multirun/0__2022_04_29__15_04_42_snpe_m/0
- could not sample. Out of 10 million, 0 samples were in the support









Run 5 (l20_5): SNPE-C with flow in unconstrained space and nsf density estimator
R1:
- 30k sims
- 35 features
- ensemble of 1
- path: 2022_04_29__14_15_25_snpe_m/0
- evaluation path: 2022_04_29__14_52_37__multirun/0__2022_04_29__14_15_25_snpe_m
- 9% good simulations
- posterior log-prob 27.699

R2:
- performed on simultions from l20_3
- 30k sims
- 35 features
- ensemble of 1
- path: 2022_04_29__15_04_28_snpe_m/0
- evaluation path: Out of 10 million, 0 samples were in the support










Run 6 (l20_6): SNPE-C with flow in constrained space and MAF
R1:
- 30k sims
- 35 features
- ensemble of 1
- path: 2022_05_01__15_02_01_snpe
- evaluation path: 2022_05_01__15_35_35__multirun


R2:
- path: 2022_05_01__15_17_31_snpe
- evaluation: 2022_05_01__15_55_05__multirun










Run 7 (l20_7):
R1:
path: 2022_04_27__19_06_37_snpe
evaluation: 2022_05_04__17_39_38__multirun

R2:
path: 2022_05_05__07_47_34_snpe_m/0
evaluation: 2022_05_05__11_00_41__2022_05_05__07_47_34_snpe_m

R3:
path: 2022_05_05__21_28_21_snpe
evaluation: 2022_05_05__23_42_37__multirun

R4:
path: 2022_05_06__13_50_43_snpe
evaluation: 2022_05_06__14_50_41__multirun

R5:
path: 2022_05_07__08_17_53_snpe
evaluation: 2022_05_07__10_21_21__multirun

R6:
path: 2022_05_08__08_59_11_snpe
evaluation: 2022_05_08__11_19_07__multirun





=================================================================================
Pyloric

P31_1: pyloric net with nsf, forced to constrained space (i.e. ideal setup)

R1:
- 100k sims
- path: 2022_04_30__22_51_26_snpe_m/0
- evaluation: 2022_05_01__15_29_55__2022_04_30__22_51_26_snpe_m
- ensemble of 10

R2: 
- 50k sims
- path: 2022_05_01__15_15_50_snpe_m/0
- ensemble of 1

p31_2: pyloric net with maf, forced to constrained space

R1: 
path: 2022_05_01__17_13_52_snpe_m

p31_2:
R01-10: 2022_05_04__15_52_51__multirun
R11-20: 2022_05_04__23_08_47__multirun
R21-30: 2022_05_05__12_38_36__multirun


2022_05_06__13_08_05__multirun are the three NPE runs
2022_05_06__16_08_24__multirun TSNPE round 1-5
2022_05_06__23_20_06__multirun TSNPE round 6-30
2022_05_07__11_14_52__multirun APT round 1-5
2022_05_07__13_24_05__multirun APT round 6-15
2022_05_08__01_16_31__multirun APT round 16-22
2022_05_09__20_30_14__multirun APT round 22-26


2022_05_07__14_27_43__multirun R1-5 with APT and MAF as density estimator

2022_05_07__16_22_43__multirun R1-3 with APT and NSF and no sigmoid transform
2022_05_07__20_10_21__multirun R1-10 with APT and NSF and no sigmoid transform

<!-- 2022_05_08__10_13_49__multirun R1-5 APT NSF but 30k per round -->

Results with "full" prior and transformed:
2022_05_10__18_11_09__multirun R1-6 with APT with 10 atoms

2022_05_11__08_30_42__multirun R1-10 with TSNPE allowed_false_negatives=0.0001
2022_05_11__14_09_04__multirun R1-15 with TSNPE allowed_false_negatives=0.001 (running on cpu-short)
2022_05_11__20_45_54__multirun R16-40 with TSNPE allowed_false_negatives=0.001 (running on cpu-short)
2022_05_12__22_08_16__multirun R34-41 with TSNPE allowed_false_negatives=0.001 (running on cpu-short)
2022_05_13__10_47_43__multirun R42-50 with TSNPE allowed_false_negatives=0.001 (running on cpu-short)

2022_05_11__11_17_11__multirun R1-10 with APT with 2 atoms (running on aroaki)
2022_05_11__14_21_37           R1-5 with APT with 2 atoms (running on vm-tsnpe)
2022_05_11__19_01_37           R6-11 with APT with 2 atoms (running on vm-tsnpe)
2022_05_12__16_02_15__multirun R13-15 with APT with 2 atoms (running on aroaki)

2022_05_13__12_18_39__multirun R1-10 APT 2 atoms, 30k per sim. Do not replace NaN
2022_05_14__00_06_34__multirun R7-10 APT 2 atoms, 30k per sim. Do not replace NaN

2022_05_14__00_13_47__multirun R1-6 APT 2 atoms, 30k per round. Replace NaN
2022_05_14__13_21_13__multirun R7-10 APT 2 atoms, 30k per round. Replace NaN
2022_05_14__23_49_35__multirun R11-13, APT 2 atoms, 30k per round. Replace NaN
2022_05_15__18_15_26__multirun R14-15, APT 2 atoms, 30k per round. Replace NaN

2022_05_14__11_14_07__multirun TSNPE R1-4 30k per round. Replace NaN
2022_05_15__21_20_01__multirun TSNPE R5-8 30k per round. Replace NaN. Running on cpu-short
2022_05_16__08_23_07__multirun TSNPE R9-12 30k per round. Replace NaN. Running on cpu-short

2022_05_16__08_53_55__multirun TSNPE R1-7 30k per round. Replace NaN. Rejection sampling
2022_05_16__19_37_53__multirun TSNPE R8-8 30k per round. Replace NaN. sir sampling (aroaki)
2022_05_17__01_40_01__multirun TSNPE R8-10 30k per round. Replace NaN. sir sampling (vm-tsnpe)
2022_05_17__01_48_28__multirun TSNPE R8-9 30k per round. Replace NaN. rejection sampling (cpu-short)

2022_05_16__11_36_58__multirun APT R1-2, 2 atoms, 30k per round replace NaN, no sigmoid_theta
2022_05_17__09_36_25__multirun APT R3-4, 2 atoms, 30k per round replace NaN, no sigmoid_theta