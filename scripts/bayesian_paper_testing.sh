# All Traditional ML Models (Deterministic)
# bash# All bit-vector representations + RF
python process_and_train.py -d QM9 -t homo_lumo_gap -m rf -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1a_all_rf.csv

# All bit-vector representations + SVM
python process_and_train.py -d QM9 -t homo_lumo_gap -m svm -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1a_all_svm.csv

# All bit-vector representations + XGBoost
python process_and_train.py -d QM9 -t homo_lumo_gap -m xgboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1a_all_xgboost.csv

# Bayesian Models
# bash# All bit-vector representations + QRF
python process_and_train.py -d QM9 -t homo_lumo_gap -m qrf -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1a_all_qrf.csv

# All bit-vector representations + Gauche GP
python process_and_train.py -d QM9 -t homo_lumo_gap -m gauche -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1a_all_gauche.csv

# All bit-vector representations + NGBoost
python process_and_train.py -d QM9 -t homo_lumo_gap -m ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1a_all_ngboost.csv

# Figure 1b: RMSE vs. noise σ - Graph representations
# bash# All graph models
python process_and_train.py -d QM9 -t homo_lumo_gap -m gin gcn ginct gin2d graph_gp -r graph --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1b_all_graph.csv

# Graph models with Bayesian transformations
python process_and_train.py -d QM9 -t homo_lumo_gap -m gin gcn -r graph --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation full -f ../results/fig1b_graph_bnn_full.csv

python process_and_train.py -d QM9 -t homo_lumo_gap -m gin gcn -r graph --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation last_layer -f ../results/fig1b_graph_bnn_last.csv

python process_and_train.py -d QM9 -t homo_lumo_gap -m gin gcn -r graph --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation variational -f ../results/fig1b_graph_bnn_variational.csv
python3 process_and_train.py -t homo_lumo_gap -m gin gcn -r graph -n 100 --split scaffold -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation variational -u true -f results/fig1c_graph_bnn_var.csv
# Above: 10163762
# Figure 1c: RMSE vs. noise σ - NN vs BNN only
# All Neural Network Models (Standard)
# # bash# All bit-vector representations + DNN
# python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1c_all_dnn.csv

# # All bit-vector representations + MLP
# python process_and_train.py -d QM9 -t homo_lumo_gap -m mlp -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1c_all_mlp.csv

# # All bit-vector representations + Flexible DNN
# python process_and_train.py -d QM9 -t homo_lumo_gap -m flexible_dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1c_all_flexible_dnn.csv

# # All bit-vector representations + Residual MLP
# python process_and_train.py -d QM9 -t homo_lumo_gap -m residual_mlp -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1c_all_residual_mlp.csv

# # All bit-vector representations + Factorization MLP
# python process_and_train.py -d QM9 -t homo_lumo_gap -m factorization_mlp -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1c_all_factorization_mlp.csv

# # All bit-vector representations + MTL
# python process_and_train.py -d QM9 -t homo_lumo_gap -m mtl -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1c_all_mtl.csv

# All Bayesian Neural Network Transformations
# bash# All bit-vector representations + BNN (full Bayesian)
# python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation full -f ../results/fig1c_all_bnn_full.csv

# # All bit-vector representations + BNN (last layer only)
# python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation last_layer -f ../results/fig1c_all_bnn_last.csv

# # All bit-vector representations + BNN (variational)
# python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation variational -f ../results/fig1c_all_bnn_variational.csv

# # MLP with all Bayesian transformations
# python process_and_train.py -d QM9 -t homo_lumo_gap -m mlp -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation full -f ../results/fig1c_all_mlp_bnn_full.csv

# python process_and_train.py -d QM9 -t homo_lumo_gap -m mlp -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation last_layer -f ../results/fig1c_all_mlp_bnn_last.csv

# python process_and_train.py -d QM9 -t homo_lumo_gap -m mlp -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation variational -f ../results/fig1c_all_mlp_bnn_variational.csv

# # Residual MLP with Bayesian transformations
# python process_and_train.py -d QM9 -t homo_lumo_gap -m residual_mlp -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation full -f ../results/fig1c_all_residual_mlp_bnn_full.csv

# python process_and_train.py -d QM9 -t homo_lumo_gap -m residual_mlp -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation last_layer -f ../results/fig1c_all_residual_mlp_bnn_last.csv

# # Factorization MLP with Bayesian transformations
# python process_and_train.py -d QM9 -t homo_lumo_gap -m factorization_mlp -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation full -f ../results/fig1c_all_factorization_mlp_bnn_full.csv

# python process_and_train.py -d QM9 -t homo_lumo_gap -m factorization_mlp -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation last_layer -f ../results/fig1c_all_factorization_mlp_bnn_last.csv
# # Figure 1d: R² vs. noise σ - Top-performing model–rep–target pairs
# # QM9 Multiple Targets - ALL models and representations
# # bash# HOMO-LUMO Gap - All models and representations
# python process_and_train.py -d QM9 -t homo_lumo_gap -m rf qrf svm xgboost lgb dnn mlp flexible_dnn residual_mlp factorization_mlp mtl gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1d_qm9_homo_lumo_all.csv

# # HOMO-LUMO Gap - RNN/GRU for SMILES
# python process_and_train.py -d QM9 -t homo_lumo_gap -m rnn gru -r smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1d_qm9_homo_lumo_rnn_gru.csv

# # HOMO-LUMO Gap - Graph models
# python process_and_train.py -d QM9 -t homo_lumo_gap -m gin gcn ginct gin2d graph_gp -r graph --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1d_qm9_homo_lumo_graph.csv

# # Alpha (Polarizability) - All models and representations
# python process_and_train.py -d QM9 -t alpha -m rf qrf svm xgboost lgb dnn mlp flexible_dnn residual_mlp factorization_mlp mtl gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1d_qm9_alpha_all.csv

# # LUMO Energy - All models and representations
# python process_and_train.py -d QM9 -t G -m rf qrf svm xgboost lgb dnn mlp flexible_dnn residual_mlp factorization_mlp mtl gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1d_qm9_lumo_all.csv

# # Dipole Moment - All models and representations
# python process_and_train.py -d QM9 -t mu -m rf qrf svm xgboost lgb dnn mlp flexible_dnn residual_mlp factorization_mlp mtl gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1d_qm9_mu_all.csv

# # HOMO Energy - All models and representations
# python process_and_train.py -d QM9 -t H -m rf qrf svm xgboost lgb dnn mlp flexible_dnn residual_mlp factorization_mlp mtl gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1d_qm9_homo_all.csv

# # Internal Energy - All models and representations
# python process_and_train.py -d QM9 -t U -m rf qrf svm xgboost lgb dnn mlp flexible_dnn residual_mlp factorization_mlp mtl gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1d_qm9_internal_energy_all.csv

# # Enthalpy - All models and representations
# python process_and_train.py -d QM9 -t H_a -m rf qrf svm xgboost lgb dnn mlp flexible_dnn residual_mlp factorization_mlp mtl gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1d_qm9_enthalpy_all.csv

# # Free Energy - All models and representations
# python process_and_train.py -d QM9 -t G_a -m rf qrf svm xgboost lgb dnn mlp flexible_dnn residual_mlp factorization_mlp mtl gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1d_qm9_free_energy_all.csv

# # Heat Capacity - All models and representations
# python process_and_train.py -d QM9 -t C -m rf qrf svm xgboost lgb dnn mlp flexible_dnn residual_mlp factorization_mlp mtl gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig1d_qm9_heat_capacity_all.csv

# Figure 2b: Final R² at fixed σ (Bayesian vs Deterministic)
# Fixed σ = 0.25 Comparisons (moderate noise level)
# bash# Traditional ML: RF vs QRF - All representations
python process_and_train.py -d QM9 -t homo_lumo_gap -m rf qrf -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.25 --distribution gaussian -s scaffold -f ../results/fig2b_sigma03_tree_models.csv

# Neural nets: Standard vs Full Bayesian - All representations  
python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.25 --distribution gaussian -s scaffold -f ../results/fig2b_sigma03_dnn.csv

python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.25 --distribution gaussian -s scaffold --bayesian-transformation full -f ../results/fig2b_sigma03_bnn_full.csv

# Neural nets: Standard vs Last Layer Bayesian - All representations
python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.25 --distribution gaussian -s scaffold --bayesian-transformation last_layer -f ../results/fig2b_sigma03_bnn_last.csv

# Neural nets: Standard vs Variational Bayesian - All representations
python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.25 --distribution gaussian -s scaffold --bayesian-transformation variational -f ../results/fig2b_sigma03_bnn_variational.csv

# MLP variants with Bayesian transformations
python process_and_train.py -d QM9 -t homo_lumo_gap -m mlp -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.25 --distribution gaussian -s scaffold -f ../results/fig2b_sigma03_mlp.csv

python process_and_train.py -d QM9 -t homo_lumo_gap -m mlp -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.25 --distribution gaussian -s scaffold --bayesian-transformation full -f ../results/fig2b_sigma03_mlp_bnn_full.csv

# Above: 10163764
# Fixed σ = 0.5 Comparisons (ECFP4/RF performance drop-off point)
# bash# Traditional ML: RF vs QRF - All representations
python process_and_train.py -d QM9 -t homo_lumo_gap -m rf qrf -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.5 --distribution gaussian -s scaffold -f ../results/fig2b_sigma06_tree_models.csv

# Neural nets: Standard vs Full Bayesian - All representations
python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.5 --distribution gaussian -s scaffold -f ../results/fig2b_sigma06_dnn.csv

python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.5 --distribution gaussian -s scaffold --bayesian-transformation full -f ../results/fig2b_sigma06_bnn_full.csv

# Neural nets: Standard vs Last Layer Bayesian - All representations
python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.5 --distribution gaussian -s scaffold --bayesian-transformation last_layer -f ../results/fig2b_sigma06_bnn_last.csv

# Neural nets: Standard vs Variational Bayesian - All representations
python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.5 --distribution gaussian -s scaffold --bayesian-transformation variational -f ../results/fig2b_sigma06_bnn_variational.csv

# Advanced ML models comparison
python process_and_train.py -d QM9 -t homo_lumo_gap -m xgboost lgb svm -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.5 --distribution gaussian -s scaffold -f ../results/fig2b_sigma06_advanced_ml.csv

# Uncertainty models comparison
python process_and_train.py -d QM9 -t homo_lumo_gap -m gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.5 --distribution gaussian -s scaffold -f ../results/fig2b_sigma06_uncertainty_models.csv

# Above: 10163765
# Figure 4a: RMSE vs. noise σ for different noise types
# All representations with ALL noise distributions
# bash# All representations - Gaussian noise
python process_and_train.py -d QM9 -t homo_lumo_gap -m rf qrf dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig4a_all_gaussian.csv

# All representations - Left-tailed noise
python process_and_train.py -d QM9 -t homo_lumo_gap -m rf qrf dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution left-tailed -s scaffold -f ../results/fig4a_all_left_tailed.csv

# All representations - Right-tailed noise  
python process_and_train.py -d QM9 -t homo_lumo_gap -m rf qrf dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution right-tailed -s scaffold -f ../results/fig4a_all_right_tailed.csv

# All representations - U-shaped noise
python process_and_train.py -d QM9 -t homo_lumo_gap -m rf qrf dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution u-shaped -s scaffold -f ../results/fig4a_all_u_shaped.csv

# All representations - Uniform noise
python process_and_train.py -d QM9 -t homo_lumo_gap -m rf qrf dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution uniform -s scaffold -f ../results/fig4a_all_uniform.csv

# Advanced ML models with different noise types
python process_and_train.py -d QM9 -t homo_lumo_gap -m xgboost lgb svm gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold -f ../results/fig4a_advanced_ml_gaussian.csv

python process_and_train.py -d QM9 -t homo_lumo_gap -m xgboost lgb svm gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution left-tailed -s scaffold -f ../results/fig4a_advanced_ml_left_tailed.csv

python process_and_train.py -d QM9 -t homo_lumo_gap -m xgboost lgb svm gauche ngboost -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution uniform -s scaffold -f ../results/fig4a_advanced_ml_uniform.csv

# Above: 10163766
# Neural networks with Bayesian transformations across noise types
# python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution gaussian -s scaffold --bayesian-transformation full -f ../results/fig4a_bnn_full_gaussian.csv

# python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution u-shaped -s scaffold --bayesian-transformation full -f ../results/fig4a_bnn_full_u_shaped.csv

# python process_and_train.py -d QM9 -t homo_lumo_gap -m dnn -r ecfp4 pdv sns smiles randomized_smiles --random-seed 42 -n 10000 -b 20 --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --distribution uniform -s scaffold --bayesian-transformation last_layer -f ../results/fig4a_bnn_last_uniform.csv

