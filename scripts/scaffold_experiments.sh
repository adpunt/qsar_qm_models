#!/bin/bash
#SBATCH --job-name=setup_noise_env  # Name of the job
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --time=24:00:00              # Request 64 hour runtime 
#SBATCH --partition=medium           # Choose the appropriate partition
#SBATCH --mem=128G                    # Allocate 16GB of memory
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk  # Notification email
#SBATCH --output=slurm-%j.out        # Save standard output to slurm-<jobID>.out

export MAMBA_EXE="$(pwd)/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd qsar_qm_models/ 
. setup.sh

cd /data/stat-cadd/scat9264/qsar_qm_models/scripts


# Tuning vs not comparison - should be quick, adjust as needed based on what models are available
# git checkout main
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost qrf ngboost dnn flexible_dnn gauche svm -r ecfp4 smiles sns pdv -n 5000 -f ../results/shap.csv  --split scaffold >> tuningTestBaseline.txt
# git checkout main
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost qrf ngboost dnn flexible_dnn gauche svm -r ecfp4 smiles sns pdv -n 5000 -f ../results/shap.csv  --split scaffold  >> tuningTest.txt


# SHAP - wait to compare tuning
git checkout shap
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost ngboost qrf -r ecfp4 smiles sns pdv -n 5000 -f ../results/shap.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --shap true --split scaffold 

git checkout shap
python3 process_and_train.py -t homo_lumo_gap -m svm -r ecfp4 sns pdv -n 3000 -f ../results/shapSVM.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --shap true --split scaffold 

# Line plots - WRONG FILE LOCATIONS
# Previously: ../results/scaffoldTuning.csv for all but graphs
# I need to track how long all of these are 
git checkout main
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost mlp dnn mtl residual_mlp factorization_mlp flexible_dnn qrf ngboost rnn -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  -b 10 --split scaffold --random-seed 420

# 9915121
# time=3580
python3 process_and_train.py -t homo_lumo_gap -m rf -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> rf_output.txt
# 9915123
# time=707
python3 process_and_train.py -t homo_lumo_gap -m xgboost -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> xgboost_output.txt
# 9915125
# time=2261
python3 process_and_train.py -t homo_lumo_gap -m mlp -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> mlp_output.txt
# 9916840
# time=1520
python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> dnn_output.txt
# 9915127
# time=1417
python3 process_and_train.py -t homo_lumo_gap -m mtl -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> mtl_output.txt
# 9915128
# time=1383
python3 process_and_train.py -t homo_lumo_gap -m residual_mlp -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> residual_mlp_output.txt
# 9915129
# time=2767
python3 process_and_train.py -t homo_lumo_gap -m factorization_mlp -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> factorization_mlp_output.txt
# 9916841
# time=1466
python3 process_and_train.py -t homo_lumo_gap -m flexible_dnn -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> flexible_dnn_output.txt
# 9916842
# time=3475
python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> qrf_output.txt
# 9916843
# time=7917
python3 process_and_train.py -t homo_lumo_gap -m ngboost -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> ngboost_output.txt
# 9916844
# time=1448
python3 process_and_train.py -t homo_lumo_gap -m rnn -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> rnn_output.txt
# 9916845
# time=8028
python3 process_and_train.py -t homo_lumo_gap -m gauche -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> gauche_output.txt
# 9916846
# time=3294
python3 process_and_train.py -t homo_lumo_gap -m svm -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> svm_output.txt
# 9916847
# time=1835
python3 process_and_train.py -t homo_lumo_gap -m gin -r graph -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> gin_output.txt
# 9916849
# time=3385
python3 process_and_train.py -t homo_lumo_gap -m gcn -r graph -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> gcn_output.txt
# 9916850
# time=14711
python3 process_and_train.py -t homo_lumo_gap -m graph_gp -r graph -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold --random-seed 420 >> graph_gp_output.txt

# After you figure out times, set up bootstrapping experiments 

# Bayesian transformation
# 9916851
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m  dnn -r ecfp4 smiles pdv -n 10000 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -f ../results/bayesianBaseline.csv  -b 10

# 9916852
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m  dnn -r ecfp4 smiles pdv -n 10000 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --bayesian-transformation full -f ../results/bayesianFull.csv  -b 10

# 9916853
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m  dnn -r ecfp4 smiles pdv -n 10000 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --bayesian-transformation last_layer -f ../results/bayesianLastLayer.csv  -b 10

# 9916854
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m  dnn -r ecfp4 smiles pdv -n 10000 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --bayesian-transformation variational -f ../results/bayesianVariational.csv  -b 10


# Collect uncertainty data
# 9916855
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation full -r ecfp4 smiles pdv -n 10000 -f ../results/uncertainty_full.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  --split scaffold 

# 9916856
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation last_layer -r ecfp4 smiles pdv -n 10000 -f ../results/uncertainty_last_layer.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  --split scaffold 
# 9916857
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation variational -r ecfp4 smiles pdv -n 10000 -f ../results/uncertainty_variational.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  --split scaffold 



# Line plots - the data is in the wrong place
# Need to go through scaffoldTuning.csv - everything with 5000 is actual tuning, everything with 10k should be moved to another file, ideally line plot

# Bayesian last layer and full didn't seem to work
# Uncertainty doesn't make sense unless I can find which command is working (probably variational)



# rf (est. 10 hours)
# 9919017
python3 process_and_train.py -t homo_lumo_gap -m rf -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> rf_output_full.txt

# xgboost (est. 2 hours)
# 9919023
python3 process_and_train.py -t homo_lumo_gap -m xgboost -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> xgboost_output_full.txt

# mlp (est. 7 hours)
# 9919022
python3 process_and_train.py -t homo_lumo_gap -m mlp -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> mlp_output_full.txt

# dnn (est. 5 hours)
# 9919024
python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> dnn_output_full.txt

# mtl (est. 4 hours)
# 9919025
python3 process_and_train.py -t homo_lumo_gap -m mtl -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> mtl_output_full.txt

# residual_mlp (est. 4 hours)
# 9919026
python3 process_and_train.py -t homo_lumo_gap -m residual_mlp -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> residual_mlp_output_full.txt

# factorization_mlp (est. 8 hours)
# 9919027
python3 process_and_train.py -t homo_lumo_gap -m factorization_mlp -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> factorization_mlp_output_full.txt

# flexible_dnn (est. 5 hours)
# 9919028
python3 process_and_train.py -t homo_lumo_gap -m flexible_dnn -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> flexible_dnn_output_full.txt

# qrf (est. 10 hours)
# 9919030
python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> qrf_output_full.txt

# ngboost (est. 22 hours)
# 9919031
python3 process_and_train.py -t homo_lumo_gap -m ngboost -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> ngboost_output_full.txt

# rnn (est. 5 hours)
# 9919033
python3 process_and_train.py -t homo_lumo_gap -m rnn -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> rnn_output_full.txt

# gauche (est. 23 hours)
# 9919036
python3 process_and_train.py -t homo_lumo_gap -m gauche -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> gauche_output_full.txt

# svm (est. 10 hours)
# 9919038
python3 process_and_train.py -t homo_lumo_gap -m svm -r ecfp4 smiles sns randomized_smiles pdv -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> svm_output_full.txt

# gin (est. 6 hours)
# 9919039
python3 process_and_train.py -t homo_lumo_gap -m gin -r graph -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> gin_output_full.txt

# gcn (est. 10 hours)
# 9919040
python3 process_and_train.py -t homo_lumo_gap -m gcn -r graph -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> gcn_output_full.txt

# graph_gp (est. 41 hours)
# 9919041
python3 process_and_train.py -t homo_lumo_gap -m graph_gp -r graph -n 10000 -f ../results/linePlot.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -b 9 >> graph_gp_output_full.txt


# 9941340
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost dnn -r ecfp4 sns smiles pdv -n 10000 -f ../results/distributionBaseline.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 --split scaffold
# 9941341
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost dnn -r ecfp4 sns smiles pdv -n 10000 -f ../results/distributionLeftTailed.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 --split scaffold --distribution left-tailed
# 9941342
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost dnn -r ecfp4 sns smiles pdv -n 10000 -f ../results/distributionRightTailed.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 --split scaffold --distribution right-tailed
# 9941343
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost dnn -r ecfp4 sns smiles pdv -n 10000 -f ../results/distributionUShaped.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 --split scaffold --distribution u-shaped
# 9941344
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost dnn -r ecfp4 sns smiles pdv -n 10000 -f ../results/distributionUniform.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 --split scaffold --distribution uniform


# 9941345
python3 process_and_train.py -t homo_lumo_gap -m gauche -r ecfp4 sns smiles pdv -n 10000 -f ../results/distributionBaseline.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 --split scaffold
# 9941346
python3 process_and_train.py -t homo_lumo_gap -m gauche -r ecfp4 sns smiles pdv -n 10000 -f ../results/distributionLeftTailed.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 --split scaffold --distribution left-tailed
# 9941347
python3 process_and_train.py -t homo_lumo_gap -m gauche -r ecfp4 sns smiles pdv -n 10000 -f ../results/distributionRightTailed.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 --split scaffold --distribution right-tailed
# 9941348
python3 process_and_train.py -t homo_lumo_gap -m gauche -r ecfp4 sns smiles pdv -n 10000 -f ../results/distributionUShaped.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 --split scaffold --distribution u-shaped
# 9941349
python3 process_and_train.py -t homo_lumo_gap -m gauche -r ecfp4 sns smiles pdv -n 10000 -f ../results/distributionUniform.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 --split scaffold --distribution uniform


# Uncertainty
# 9952550
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation full -r ecfp4 smiles pdv sns -n 10000 -f ../results/uncertainty_full.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --split scaffold 
# 9952551
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation last_layer -r ecfp4 smiles pdv sns -n 10000 -f ../results/uncertainty_last_layer.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --split scaffold 
# 9952553
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation variational -r ecfp4 smiles pdv sns -n 10000 -f ../results/uncertainty_variational.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --split scaffold 
# 9954093
python3 process_and_train.py -t homo_lumo_gap -m ngboost -r ecfp4 smiles pdv sns -n 10000 -f ../results/uncertainty_ngboost.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --split scaffold 
# 9954094
python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 smiles pdv sns -n 10000 -f ../results/uncertainty_qrf.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --split scaffold 
# 9954095
python3 process_and_train.py -t homo_lumo_gap -m gauche -r ecfp4 smiles pdv sns -n 10000 -f ../results/uncertainty_gauche.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --split scaffold 
# 9954096
python3 process_and_train.py -t homo_lumo_gap -m graph_gp -r graph -n 10000 -f ../results/uncertainty_graph_gp.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --split scaffold 


# Come back with more bootstrapping iterations if it seems promising
# Alternative targets
# 9952559
python3 process_and_train.py -t mu -m rf xgboost gauche dnn graph_gp -r ecfp4 smiles sns pdv graph -n 10000 -f ../results/mu.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold 
# 9952560
python3 process_and_train.py -t alpha -m rf xgboost gauche dnn graph_gp -r ecfp4 smiles sns pdv graph -n 10000 -f ../results/alpha.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold 


# Come back with more bootstrapping iterations if it seems promising
# Give up on these they're pointless 
# Non-Gaussian noise: 
# 24 hours each
# 10027434
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn graph_gp -r ecfp4 sns smiles pdv graph -n 10000 -f ../results/distributionGaussian.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --distribution gaussian --split scaffold
# 10027435
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn graph_gp -r ecfp4 sns smiles pdv graph -n 10000 -f ../results/distributionLeft.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --distribution left-tailed --split scaffold
# 10027436
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn graph_gp -r ecfp4 sns smiles pdv graph -n 10000 -f ../results/distributionRight.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --distribution right-tailed --split scaffold
# 10027437
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn graph_gp -r ecfp4 sns smiles pdv graph -n 10000 -f ../results/distributionU.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --distribution u-shaped --split scaffold
# 10027438
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn graph_gp -r ecfp4 sns smiles pdv graph -n 10000 -f ../results/distributionUniform.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --distribution uniform --split scaffold


# Graph bayesian transformation 
# 9954454
python3 process_and_train.py -t homo_lumo_gap -m gin -r graph -n 10000 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -f ../results/bayesianBaselineGraph.csv  -b 10
# 9954455
python3 process_and_train.py -t homo_lumo_gap -m gin -r graph -n 10000 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --bayesian-transformation full -f ../results/bayesianFullGraph.csv  -b 10
# 9954456
python3 process_and_train.py -t homo_lumo_gap -m gin -r graph -n 10000 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --bayesian-transformation last_layer -f ../results/bayesianLastLayerGraph.csv  -b 10
# 9954457
python3 process_and_train.py -t homo_lumo_gap -m gin -r graph -n 10000 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --bayesian-transformation variational -f ../results/bayesianVariationalGraph.csv  -b 10


# 10033696
python3 process_and_train.py -t homo_lumo_gap -m flexible_dnn -r  ecfp4 sns pdv smiles -n 10000 -f ../results/tuning.csv --n-trials 500 --tuning true --split scaffold >> tuning_fps.txt
# 10033697
python3 process_and_train.py -t homo_lumo_gap -m gin -r  graph -n 10000 -f ../results/tuning.csv --n-trials 500 --tuning true --split scaffold >> tuning_graphs.txt

# 10033698
# Test SNS (ECFP) representation
python3 noise_mitigation.py \
    --target homo_lumo_gap \
    --sample_size 10000 \
    --molecular_representation sns \
    --noise_levels 0.05 0.1 0.15 0.2 0.25 \
    --noise_type gaussian \
    --baseline_model random_forest \
    --output_path results_sns_10k.csv
# Test PDV (RDKit descriptors) representation  
python3 noise_mitigation.py \
    --target homo_lumo_gap \
    --sample_size 10000 \
    --noise_levels 0.05 0.1 0.15 0.2 0.25 \
    --noise_type gaussian \
    --baseline_model random_forest \
    --output_path results_pdv_10k.csv

# Big noise strategies experiment
# 10033699