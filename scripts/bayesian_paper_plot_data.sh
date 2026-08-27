#!/bin/bash

# Create directories
mkdir -p results logs

# Initialize log files
echo "Experiment run started at $(date)" | tee logs/progress.log logs/timing.log logs/summary.log
echo "=== QSAR/QSPR Noise Robustness Experiments ===" | tee -a logs/progress.log
echo "" | tee -a logs/progress.log

# Total number of experiments (updated count)
TOTAL_EXPERIMENTS=105
CURRENT=0
START_TIME=$(date +%s)

# Function to run command with comprehensive logging
run_experiment() {
    local cmd="$1"
    local name="$2"
    local section="$3"
    
    ((CURRENT++))
    local cmd_start_time=$(date +%s)
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$CURRENT/$TOTAL_EXPERIMENTS] [$section] Starting: $name" | tee -a logs/progress.log
    echo "Time: $timestamp" | tee -a logs/progress.log
    echo "Command: $cmd" | tee -a logs/progress.log
    echo "---" | tee -a logs/progress.log
    
    # Run the command
    if eval "$cmd" &> "logs/${name}.log"; then
        local cmd_end_time=$(date +%s)
        local duration=$((cmd_end_time - cmd_start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        
        printf "✓ SUCCESS: %s | Duration: %02d:%02d:%02d | Completed at: %s\n" \
            "$name" "$hours" "$minutes" "$seconds" "$(date '+%H:%M:%S')" | tee -a logs/progress.log logs/timing.log logs/summary.log
        
        # Check if results file was actually created
        local results_file=$(echo "$cmd" | grep -o '\-f [^ ]*' | cut -d' ' -f2)
        if [[ -n "$results_file" && -f "$results_file" ]]; then
            local file_size=$(du -h "$results_file" | cut -f1)
            echo "  → Results file: $results_file ($file_size)" | tee -a logs/progress.log
        fi
        
        return 0
    else
        local cmd_end_time=$(date +%s)
        local duration=$((cmd_end_time - cmd_start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        
        printf "✗ FAILED: %s | Duration: %02d:%02d:%02d | Failed at: %s\n" \
            "$name" "$hours" "$minutes" "$seconds" "$(date '+%H:%M:%S')" | tee -a logs/progress.log logs/timing.log logs/summary.log logs/failed.log
            
        echo "  → Error log: logs/${name}.log" | tee -a logs/progress.log logs/failed.log
        echo "  → Command: $cmd" | tee -a logs/failed.log
        echo "" | tee -a logs/failed.log
        
        return 1
    fi
    
    echo "" | tee -a logs/progress.log
}

# Function to show overall progress
show_progress() {
    local elapsed=$(($(date +%s) - START_TIME))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    
    if [ $CURRENT -gt 0 ]; then
        local avg_time_per_exp=$((elapsed / CURRENT))
        local estimated_remaining=$(((TOTAL_EXPERIMENTS - CURRENT) * avg_time_per_exp))
        local eta_hours=$((estimated_remaining / 3600))
        local eta_minutes=$(((estimated_remaining % 3600) / 60))
        
        printf "\n=== PROGRESS UPDATE ===\n" | tee -a logs/progress.log
        printf "Completed: %d/%d experiments\n" "$CURRENT" "$TOTAL_EXPERIMENTS" | tee -a logs/progress.log
        printf "Elapsed time: %02d:%02d:%02d\n" "$hours" "$minutes" "$seconds" | tee -a logs/progress.log
        printf "Estimated time remaining: %02d:%02d\n" "$eta_hours" "$eta_minutes" | tee -a logs/progress.log
        printf "Progress: %.1f%%\n\n" "$(echo "scale=1; $CURRENT * 100 / $TOTAL_EXPERIMENTS" | bc)" | tee -a logs/progress.log
    fi
}

# Trap to show progress on Ctrl+C
trap 'show_progress; echo "Interrupted by user. Check logs/ directory for details."; exit 1' INT

echo "Starting $TOTAL_EXPERIMENTS experiments..."
echo ""

# =============================================================================
# SECTION 0 - Hyperparameter Tuning (Clean Data)
# =============================================================================

echo "=== SECTION 0: Hyperparameter Tuning ===" | tee -a logs/progress.log

# Fig 6a - DEFAULT runs (for comparison)
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m rf -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 -f results/fig6a_rf_default.csv" "fig6a_rf_default" "Fig6a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m xgboost -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 -f results/fig6a_xgb_default.csv" "fig6a_xgb_default" "Fig6a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m svm -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 -f results/fig6a_svm_default.csv" "fig6a_svm_default" "Fig6a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 -u true -f results/fig6a_qrf_default.csv" "fig6a_qrf_default" "Fig6a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m ngboost -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 -u true -f results/fig6a_ngb_default.csv" "fig6a_ngb_default" "Fig6a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gauche -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 -u true -f results/fig6a_gauche_default.csv" "fig6a_gauche_default" "Fig6a"

# Fig 6a - Hyperparameter tuning for bit-vector models (clean data only)
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m rf -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --tuning true --n-trials 20 -f results/fig6a_rf_tuned.csv" "fig6a_rf_tuned" "Fig6a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m xgboost -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --tuning true --n-trials 20 -f results/fig6a_xgb_tuned.csv" "fig6a_xgb_tuned" "Fig6a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m svm -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --tuning true --n-trials 20 -f results/fig6a_svm_tuned.csv" "fig6a_svm_tuned" "Fig6a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --tuning true --n-trials 20 -u true -f results/fig6a_qrf_tuned.csv" "fig6a_qrf_tuned" "Fig6a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m ngboost -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --tuning true --n-trials 20 -u true -f results/fig6a_ngb_tuned.csv" "fig6a_ngb_tuned" "Fig6a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gauche -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --tuning true --n-trials 20 -u true -f results/fig6a_gauche_tuned.csv" "fig6a_gauche_tuned" "Fig6a"

# Fig 6b - Hyperparameter tuning for graph models (clean data only)
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gin gcn -r graph -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --tuning true --n-trials 20 -f results/fig6b_graph_tuned.csv" "fig6b_graph_tuned" "Fig6b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m graph_gp -r graph -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --tuning true --n-trials 20 -u true -f results/fig6b_graph_gp_tuned.csv" "fig6b_graph_gp_tuned" "Fig6b"

# Fig 6c - Hyperparameter tuning for neural networks (clean data only)
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp residual_mlp factorization_mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --tuning true --n-trials 20 -f results/fig6c_nn_tuned.csv" "fig6c_nn_tuned" "Fig6c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gin gcn -r graph -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --tuning true --n-trials 20 -f results/fig6c_graph_nn_tuned.csv" "fig6c_graph_nn_tuned" "Fig6c"

# Fig 6d - Hyperparameter tuning for Bayesian neural networks (clean data only)
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp residual_mlp factorization_mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --bayesian-transformation full --tuning true --n-trials 20 -u true -f results/fig6d_bnn_full_tuned.csv" "fig6d_bnn_full_tuned" "Fig6d"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp residual_mlp factorization_mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --bayesian-transformation last_layer --tuning true --n-trials 20 -u true -f results/fig6d_bnn_last_tuned.csv" "fig6d_bnn_last_tuned" "Fig6d"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp residual_mlp factorization_mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --bayesian-transformation variational --tuning true --n-trials 20 -u true -f results/fig6d_bnn_var_tuned.csv" "fig6d_bnn_var_tuned" "Fig6d"

show_progress

# Add conformal to your models (may need a figure for this)
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv sns smiles randomized_smiles --cp-base-model rf -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --tuning true --n-trials 20 -u true -f results/fig6_conformal_rf_tuned.csv" "fig6_conformal_rf_tuned" "Fig6"

# For graph models
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r graph --cp-base-model gin -n 300 --split scaffold -b 3 --epochs 10 --sigma 0.0 --tuning true --n-trials 20 -u true -f results/fig6_conformal_gin_tuned.csv" "fig6_conformal_gin_tuned" "Fig6"

# =============================================================================
# SECTION 1 - Model robustness to noise
# =============================================================================

echo "=== SECTION 1: Model Robustness to Noise ===" | tee -a logs/progress.log

# Fig 1a - RMSE vs. σ: Bit-vector reps (no NN/BNN), QM9
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m rf -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -f results/fig1a_rf.csv" "fig1a_rf" "Fig1a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m xgboost -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -f results/fig1a_xgb.csv" "fig1a_xgb" "Fig1a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m svm -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -f results/fig1a_svm.csv" "fig1a_svm" "Fig1a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig1a_qrf.csv" "fig1a_qrf" "Fig1a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m ngboost -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig1a_ngb.csv" "fig1a_ngb" "Fig1a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gauche -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig1a_gauche.csv" "fig1a_gauche" "Fig1a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model rf -u true -f results/fig1a_conformal_rf.csv" "fig1a_conformal_rf" "Fig1a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model qrf -u true -f results/fig1a_conformal_qrf.csv" "fig1a_conformal_qrf" "Fig1a"

show_progress

# Fig 1b - RMSE vs. σ: Graph reps, QM9
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gin gcn -r graph -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -f results/fig1b_graph.csv" "fig1b_graph" "Fig1b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m graph_gp -r graph -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig1b_graph_gp.csv" "fig1b_graph_gp" "Fig1b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r graph -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model gin -u true -f results/fig1b_conformal_graph.csv" "fig1b_conformal_graph" "Fig1b"

# TODO: plottings script needs to be completely redone
    # Four colors should be baseline, full, last layer, variational layer
    # Each plot should be a different model/base rep, not a whole separate rep 
# TODO: need to select which pairs of model/rep are right 
# Fig 1c - RMSE vs. σ: NN vs. BNN, QM9
# Fig 1c - RMSE vs. σ: NN vs. BNN, QM9
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp residual_mlp factorization_mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -f results/fig1c_nn.csv" "fig1c_nn" "Fig1c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gin gcn -r graph -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -f results/fig1c_graph_nn.csv" "fig1c_graph_nn" "Fig1c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp residual_mlp factorization_mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation full -u true -f results/fig1c_bnn_full.csv" "fig1c_bnn_full" "Fig1c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp residual_mlp factorization_mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation last_layer -u true -f results/fig1c_bnn_last.csv" "fig1c_bnn_last" "Fig1c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp residual_mlp factorization_mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation variational -u true -f results/fig1c_bnn_var.csv" "fig1c_bnn_var" "Fig1c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gin gcn -r graph -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation full -u true -f results/fig1c_graph_bnn_full.csv" "fig1c_graph_bnn_full" "Fig1c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gin gcn -r graph -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation last_layer -u true -f results/fig1c_graph_bnn_last.csv" "fig1c_graph_bnn_last" "Fig1c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gin gcn -r graph -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation variational -u true -f results/fig1c_graph_bnn_var.csv" "fig1c_graph_bnn_var" "Fig1c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model dnn -u true -f results/fig1c_conformal_dnn.csv" "fig1c_conformal_dnn" "Fig1c"

# Fig 1d - R² vs. σ: Top model–rep–target pairs, QM9 + Polaris
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig1d_top_pairs.csv" "fig1d_top_pairs" "Fig1d"

show_progress

# =============================================================================
# SECTION 2 - Bayesian vs. deterministic
# =============================================================================

echo "=== SECTION 2: Bayesian vs. Deterministic ===" | tee -a logs/progress.log

# TODO: Not all of this data is getting plotted, each plot should contain lines for deterministic, full bayesian transformation, last layer, variational 
# TODO: add CP?
# Fig 2a - Training curves (loss + RMSE): NN vs. BNN, QM9
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --save-per-epoch-metrics true -f results/fig2a_nn_curves.csv" "fig2a_nn_curves" "Fig2a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gin gcn -r graph -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --save-per-epoch-metrics true -f results/fig2a_graph_curves.csv" "fig2a_graph_curves" "Fig2a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation full -u true --save-per-epoch-metrics true -f results/fig2a_bnn_curves.csv" "fig2a_bnn_curves" "Fig2a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gin gcn -r graph -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation full -u true --save-per-epoch-metrics true -f results/fig2a_graph_bnn_curves.csv" "fig2a_graph_bnn_curves" "Fig2a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation last_layer -u true --save-per-epoch-metrics true -f results/fig2a_bnn_last_curves.csv" "fig2a_bnn_last_curves" "Fig2a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation variational -u true --save-per-epoch-metrics true -f results/fig2a_bnn_var_curves.csv" "fig2a_bnn_var_curves" "Fig2a"

# TODO: this should be a set of two tables: rf/qrf and a larger table for NN/BNN/full/last/variational and cp and it's baseline model counterpart
# TODO: this requires more data, specifically a different NN base model 
# Fig 2b - Bar: Final R² at fixed σ (BNN/NN, QRF/RF), QM9
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m rf qrf -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.5 -u true -f results/fig2b_rf_qrf_fixed.csv" "fig2b_rf_qrf_fixed" "Fig2b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.5 -f results/fig2b_nn_fixed.csv" "fig2b_nn_fixed" "Fig2b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.5 --bayesian-transformation full -u true -f results/fig2b_bnn_fixed.csv" "fig2b_bnn_fixed" "Fig2b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.5 --bayesian-transformation last_layer -u true -f results/fig2b_bnn_last_fixed.csv" "fig2b_bnn_last_fixed" "Fig2b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.5 --bayesian-transformation variational -u true -f results/fig2b_bnn_var_fixed.csv" "fig2b_bnn_var_fixed" "Fig2b"


# Fig 2c - Conformal vs Bayesian intervals comparison
# TODO: this didn't plot? Unsure where the error is here. I want to continue gathering data for CPs but it really should be mixed in with the rest, a separate section like this is counterproductive
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.5 --cp-base-model dnn -u true -f results/fig2c_conformal_intervals.csv" "fig2c_conformal_intervals" "Fig2c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.5 --cp-base-model rf -u true -f results/fig2c_conformal_rf_intervals.csv" "fig2c_conformal_rf_intervals" "Fig2c"

show_progress

# =============================================================================
# SECTION 3 - Uncertainty and label noise
# =============================================================================
# TODO: this is looking at bayesian but the plots aren't showing any of the bayesian transformations or CP 

echo "=== SECTION 3: Uncertainty and Label Noise ===" | tee -a logs/progress.log

# Fig 3a - Scatter: Uncertainty vs. error (QRF, GP, BNN), QM9
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig3a_qrf_unc.csv" "fig3a_qrf_unc" "Fig3a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gauche -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig3a_gp_unc.csv" "fig3a_gp_unc" "Fig3a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m graph_gp -r graph -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig3a_graph_gp_unc.csv" "fig3a_graph_gp_unc" "Fig3a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation full -u true -f results/fig3a_bnn_unc.csv" "fig3a_bnn_unc" "Fig3a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m gin gcn -r graph -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation full -u true -f results/fig3a_graph_bnn_unc.csv" "fig3a_graph_bnn_unc" "Fig3a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation last_layer -u true -f results/fig3a_bnn_last_unc.csv" "fig3a_bnn_last_unc" "Fig3a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn mlp -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation variational -u true -f results/fig3a_bnn_var_unc.csv" "fig3a_bnn_var_unc" "Fig3a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m ngboost -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig3a_ngb_unc.csv" "fig3a_ngb_unc" "Fig3a"

# Fig 3a - Conformal uncertainty comparison
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model rf -u true -f results/fig3a_conformal_rf_unc.csv" "fig3a_conformal_rf_unc" "Fig3a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv sns smiles randomized_smiles -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model dnn -u true -f results/fig3a_conformal_dnn_unc.csv" "fig3a_conformal_dnn_unc" "Fig3a"

# Fig 3b - Boxplot: Uncertainty (clean vs. noisy), QM9
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf gauche ngboost -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 1.0 -u true -f results/fig3b_clean_vs_noisy.csv" "fig3b_clean_vs_noisy" "Fig3b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 1.0 --bayesian-transformation full -u true -f results/fig3b_bnn_clean_vs_noisy.csv" "fig3b_bnn_clean_vs_noisy" "Fig3b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 1.0 --bayesian-transformation last_layer -u true -f results/fig3b_bnn_last_clean_vs_noisy.csv" "fig3b_bnn_last_clean_vs_noisy" "Fig3b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 1.0 --bayesian-transformation variational -u true -f results/fig3b_bnn_var_clean_vs_noisy.csv" "fig3b_bnn_var_clean_vs_noisy" "Fig3b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 1.0 --cp-base-model rf -u true -f results/fig3b_conformal_clean_vs_noisy.csv" "fig3b_conformal_clean_vs_noisy" "Fig3b"

# Fig 3c - Calibration curves, QM9
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf gauche ngboost -r ecfp4 -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig3c_calibration.csv" "fig3c_calibration" "Fig3c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation full -u true -f results/fig3c_bnn_calibration.csv" "fig3c_bnn_calibration" "Fig3c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation last_layer -u true -f results/fig3c_bnn_last_calibration.csv" "fig3c_bnn_last_calibration" "Fig3c"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation variational -u true -f results/fig3c_bnn_var_calibration.csv" "fig3c_bnn_var_calibration" "Fig3c"

# TODO: X-axis titles are unreadable, needs to be wider and smaller, readable labels
# TODO: another plot with uncertainty-noise correlation 
# Fig 3d - Bar: Uncertainty–error correlation, QM9 + Polaris
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf gauche ngboost -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig3d_unc_error_corr.csv" "fig3d_unc_error_corr" "Fig3d"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation full -u true -f results/fig3d_bnn_unc_error_corr.csv" "fig3d_bnn_unc_error_corr" "Fig3d"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation last_layer -u true -f results/fig3d_bnn_last_unc_error_corr.csv" "fig3d_bnn_last_unc_error_corr" "Fig3d"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation variational -u true -f results/fig3d_bnn_var_unc_error_corr.csv" "fig3d_bnn_var_unc_error_corr" "Fig3d"

show_progress

# =============================================================================
# SECTION 4 - Impact of noise type
# =============================================================================

# TODO: need to decide which models/reps you're doing, then there should be a single plot for each pair
# TODO: plotting script needs to distinguish pairs of model/rep 

echo "=== SECTION 4: Impact of Noise Type ===" | tee -a logs/progress.log

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --noise-strategy legacy -u true -f results/fig4a_legacy.csv" "fig4a_legacy" "Fig4a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --noise-strategy value_proportional -u true -f results/fig4a_valprop.csv" "fig4a_valprop" "Fig4a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --noise-strategy quantile -u true -f results/fig4a_quantile.csv" "fig4a_quantile" "Fig4a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --noise-strategy threshold -u true -f results/fig4a_threshold.csv" "fig4a_threshold" "Fig4a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --noise-strategy outlier -u true -f results/fig4a_outlier.csv" "fig4a_outlier" "Fig4a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --noise-strategy heteroscedastic -u true -f results/fig4a_hetero.csv" "fig4a_hetero" "Fig4a"

# Fig 4b - Conformal prediction with different noise strategies
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model qrf --noise-strategy legacy -u true -f results/fig4b_conformal_legacy.csv" "fig4b_conformal_legacy" "Fig4b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model qrf --noise-strategy heteroscedastic -u true -f results/fig4b_conformal_hetero.csv" "fig4b_conformal_hetero" "Fig4b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model qrf --noise-strategy value_proportional -u true -f results/fig4b_conformal_valprop.csv" "fig4b_conformal_valprop" "Fig4b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model qrf --noise-strategy quantile -u true -f results/fig4b_conformal_quantile.csv" "fig4b_conformal_quantile" "Fig4b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model qrf --noise-strategy threshold -u true -f results/fig4b_conformal_threshold.csv" "fig4b_conformal_threshold" "Fig4b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 300 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model qrf --noise-strategy outlier -u true -f results/fig4b_conformal_outlier.csv" "fig4b_conformal_outlier" "Fig4b"

show_progress

# =============================================================================
# SECTION 5 - Impact of data size
# =============================================================================

echo "=== SECTION 5: Impact of Data Size ===" | tee -a logs/progress.log

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf rf -r ecfp4 pdv -n 50 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig5a_n50.csv" "fig5a_n50" "Fig5a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf rf -r ecfp4 pdv -n 100 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig5a_n100.csv" "fig5a_n100" "Fig5a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf rf -r ecfp4 pdv -n 200 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig5a_n200.csv" "fig5a_n200" "Fig5a"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m qrf rf -r ecfp4 pdv -n 500 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 -u true -f results/fig5a_n500.csv" "fig5a_n500" "Fig5a"

# Fig 5b - Conformal prediction with different data sizes
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 50 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model qrf -u true -f results/fig5b_conformal_n50.csv" "fig5b_conformal_n50" "Fig5b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 100 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model qrf -u true -f results/fig5b_conformal_n100.csv" "fig5b_conformal_n100" "Fig5b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 200 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model qrf -u true -f results/fig5b_conformal_n200.csv" "fig5b_conformal_n200" "Fig5b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m conformal -r ecfp4 pdv -n 500 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --cp-base-model qrf -u true -f results/fig5b_conformal_n500.csv" "fig5b_conformal_n500" "Fig5b"

# Full Bayesian transformation
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 50 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation full -u true -f results/fig5b_bnn_full_n50.csv" "fig5b_bnn_full_n50" "Fig5b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 200 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation full -u true -f results/fig5b_bnn_full_n200.csv" "fig5b_bnn_full_n200" "Fig5b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 500 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation full -u true -f results/fig5b_bnn_full_n500.csv" "fig5b_bnn_full_n500" "Fig5b"

# Last layer Bayesian transformation
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 50 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation last_layer -u true -f results/fig5b_bnn_last_n50.csv" "fig5b_bnn_last_n50" "Fig5b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 200 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation last_layer -u true -f results/fig5b_bnn_last_n200.csv" "fig5b_bnn_last_n200" "Fig5b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 500 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation last_layer -u true -f results/fig5b_bnn_last_n500.csv" "fig5b_bnn_last_n500" "Fig5b"

# Variational Bayesian transformation
run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 50 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation variational -u true -f results/fig5b_bnn_var_n50.csv" "fig5b_bnn_var_n50" "Fig5b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 200 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation variational -u true -f results/fig5b_bnn_var_n200.csv" "fig5b_bnn_var_n200" "Fig5b"

run_experiment "python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 pdv -n 500 --split scaffold --use-best-params -b 3 --epochs 10 --sigma 0.0 0.5 1.0 --bayesian-transformation variational -u true -f results/fig5b_bnn_var_n500.csv" "fig5b_bnn_var_n500" "Fig5b"

show_progress


# Final experiment completion
echo "=== ALL EXPERIMENTS COMPLETED ===" | tee -a logs/progress.log logs/summary.log
echo "Experiment run finished at $(date)" | tee -a logs/progress.log logs/summary.log