# scripts/OS(S)E

The top scripts for OS(S)E-based experiments, with global parameters used in the AE and main calls to the functions: data reading, training, evaluations, plots...

* O(S)E/launch_OS(S)E.py: the topscript for running an experiment
* O(S)E/mono_gpu_OS(S)E.slurm: a submission script for SLURM-based HPC
* O(S)E/topscript.sh: a bash script to submit different configurations of launch_OS(S)E.py
 
# dinae_keras/mods/import_Datasets_OS(S)E.py
Import data for OS(S)E-based experiments

# dinae_keras/mods/define_Models.py
Design of the autoencoder:
* mods_NN/ConvAE.py: 2D-convolutional auto-encoder
* mods_NN/ConvAE.py: Gibbs-Energy NN

# dinae/mods/define_Classifiers.py
Define classifiers

# dinae/mods/FP_OS(S)E.py
Train and evaluate AE model for OS(S)E-based experiments with FP-solver
* load_Models_FP.py: load saved FP-models from previous run/iteration
* mods_DIN/def_DINConvAE.py:
* mods_DIN/eval_Performance.py:
  * eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt): compute the scores of AE reconstruction from training and test datasets
  * eval_InterpPerformance: compute the scores of the interpolation (where there is no data)
* mods_DIN/save_Models.py:
* mods_DIN/plot_Figs.py:

# dinae/mods/GB_OS(S)E.py
Train and evaluate AE model for OS(S)E-based experiments with GB-solver
* load_Models_GB.py: load saved GB-models from previous run/iteration
* mods_DIN/def_GradModel.py:
* mods_DIN/def_GradDINConvAE.py:



