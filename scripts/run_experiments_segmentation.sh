#!/bin/sh

META_OMNIUM_DIR=/mnt/homeGPU/igarzon/Meta-Learning/meta-omnium

# These are our primary 1-to-5 shot experiments. To run e.g. 5-shot experiments use `--k_shot_eval 5` instead of `--max_shots_eval 5`

# single task segmentation
ARGS=(
"--experiment_name maml_seg --best_hp_file_name maml_seg_hpo --model maml --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --test_id_datasets FSS_Test --test_od_datasets PASCAL,PH2 --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 30000 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name protomaml_seg --best_hp_file_name protomaml_seg_hpo --model protomaml --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --test_id_datasets FSS_Test --test_od_datasets PASCAL,PH2 --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 30000 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name metacurvature_seg --best_hp_file_name metacurvature_seg_hpo --model metacurvature --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --test_id_datasets FSS_Test --test_od_datasets PASCAL,PH2 --T 5 --T_val 10 --T_test 10 --runs 1 --train_iters 30000 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name protonet_seg --best_hp_file_name protonet_seg_hpo --model protonet --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --test_id_datasets FSS_Test --test_od_datasets PASCAL,PH2 --runs 1 --train_iters 30000 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name ddrr_seg --best_hp_file_name ddrr_seg_hpo --model ddrr --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --test_id_datasets FSS_Test --test_od_datasets PASCAL,PH2 --runs 1 --train_iters 30000 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name proto_finetuning_seg --best_hp_file_name proto_finetuning_seg_hpo --model proto_finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --test_id_datasets FSS_Test --test_od_datasets PASCAL,PH2 --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 30000 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name finetuning_seg --best_hp_file_name finetuning_seg_hpo --model finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --test_id_datasets FSS_Test --test_od_datasets PASCAL,PH2 --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 30000 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name linear_readout_seg --best_hp_file_name linear_readout_seg_hpo --model finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --test_id_datasets FSS_Test --test_od_datasets PASCAL,PH2 --T 20 --T_val 20 --T_test 20 --freeze --runs 1 --train_iters 30000 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name tfs_seg --best_hp_file_name tfs_seg_hpo --model tfs --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn --val_id_datasets FSS_Val --val_od_datasets Vizwiz --test_id_datasets FSS_Test --test_od_datasets PASCAL,PH2 --T 20 --T_val 20 --T_test 20 --runs 1 --train_iters 30000 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
)

for ARG in "${ARGS[@]}"
do
python metaomnium/trainers/train_cross_task_fsl.py ${ARG}
done