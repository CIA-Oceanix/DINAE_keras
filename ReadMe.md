# Joint interpolation and representation learning for irregularly-sampled satellite-derived geophysical fields

## Introduction

Associated preprints:
- Fixed-point solver: https://arxiv.org/abs/1910.00556
- Gradient-based solvers using automatic differentiation: https://arxiv.org/abs/2006.03653

License: CECILL-C license

Copyright IMT Atlantique/OceaniX, contributor(s) : R. Fablet, 21/03/2020

Contact person: ronan.fablet@imt-atlantique.fr
This software is a computer program whose purpose is to apply deep learning
schemes to dynamical systems and ocean remote sensing data.
This software is governed by the CeCILL-C license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-C
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".
As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.
In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.
The fact that you are presently reading this means that you have had
knowledge of the CeCILL-C license and that you accept its terms.

## Architecture of the code

### scripts/OS(S)E

The top scripts for OS(S)E-based experiments, with global parameters used in the AE and main calls to the functions: data reading, training, evaluations, plots...

* O(S)E/launch_OS(S)E.py: the topscript for running an experiment
* O(S)E/mono_gpu_OS(S)E.slurm: a submission script for SLURM-based HPC
* O(S)E/topscript.sh: a bash script to submit different configurations of launch_OS(S)E.py
 
### dinae_keras/mods/import_Datasets_OS(S)E.py
Import data for OS(S)E-based experiments

### dinae_keras/mods/define_Models.py
Design of the autoencoder:
* mods_NN/ConvAE.py: 2D-convolutional auto-encoder
* mods_NN/GENN.py: Gibbs-Energy NN

### dinae/mods/define_Classifiers.py
Define classifiers

### dinae/mods/FP_OS(S)E.py
Train and evaluate AE model for OS(S)E-based experiments with FP-solver
* load_Models_FP.py: load saved FP-models from previous run/iteration
* mods_DIN/def_DINConvAE.py:
* mods_DIN/eval_Performance.py:
  * eval_AEPerformance(x_train,rec_AE_Tr,x_test,rec_AE_Tt): compute the scores of AE reconstruction from training and test datasets
  * eval_InterpPerformance: compute the scores of the interpolation (where there is no data)
* mods_DIN/save_Models.py:
* mods_DIN/plot_Figs.py:

### dinae/mods/GB_OS(S)E.py
Train and evaluate AE model for OS(S)E-based experiments with GB-solver
* load_Models_GB.py: load saved GB-models from previous run/iteration
* mods_DIN/def_GradModel.py:
* mods_DIN/def_GradDINConvAE.py:

## Results

Below is an illustration of the results obtained on the daily velocity SSH field
when interpolating pseudo irregular and noisy observations (top-right panels) corresponding to
along-track nadir (left) with additional pseudo wide-swath SWOT observations (right) built
from an idealized groundtruth (top-left panels) with state-of-the-art optimal interpolation
(bottom-left panels) and the newly proposed end-to-end learning approach (bottom-right panels):

Nadir only                 |  Nadir+SWOT
:-------------------------:|:-------------------------:
![Farmers Market Finder Demo](figs/animation_grads_OSSE.gif)  |  ![Farmers Market Finder Demo](figs/animation_grads_OSSE.gif)



