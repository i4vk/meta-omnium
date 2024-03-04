#!/bin/sh

META_OMNIUM_DIR=/mnt/homeGPU/igarzon/Meta-Learning/meta-omnium

# These are our primary 1-to-5 shot experiments. To run e.g. 5-shot experiments use `--k_shot_eval 5` instead of `--max_shots_eval 5`

# out-of-task evaluation on regression
ARGS=(
"--experiment_name maml_regr --model_path models/maml_multi_task/best-model.pkl --best_hp_file_name maml_multi_hpo --model maml --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --test_od_datasets ShapeNet2D_Test,Distractor_Test,ShapeNet1D_Test,Pascal1D_Test --T 5 --T_val 10 --T_test 10 --runs 1 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name protomaml_regr --model_path models/protomaml_multi_task/best-model.pkl --best_hp_file_name protomaml_multi_hpo --model protomaml --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --test_od_datasets ShapeNet2D_Test,Distractor_Test,ShapeNet1D_Test,Pascal1D_Test --T 5 --T_val 10 --T_test 10 --runs 1 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name metacurvature_regr --model_path models/metacurvature_multi_task/best-model.pkl --best_hp_file_name metacurvature_multi_hpo --model metacurvature --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --test_od_datasets ShapeNet2D_Test,Distractor_Test,ShapeNet1D_Test,Pascal1D_Test --T 5 --T_val 10 --T_test 10 --runs 1 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name protonet_regr --model_path models/protonet_multi_task/best-model.pkl --best_hp_file_name protonet_multi_hpo --model protonet --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --test_od_datasets ShapeNet2D_Test,Distractor_Test,ShapeNet1D_Test,Pascal1D_Test --runs 1 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name ddrr_regr --model_path models/ddrr_multi_task/best-model.pkl --best_hp_file_name ddrr_multi_hpo --model ddrr --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --test_od_datasets ShapeNet2D_Test,Distractor_Test,ShapeNet1D_Test,Pascal1D_Test --runs 1 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name proto_finetuning_regr --model_path models/proto_finetuning_multi_task/best-model.pkl --best_hp_file_name proto_finetuning_multi_hpo --model proto_finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --test_od_datasets ShapeNet2D_Test,Distractor_Test,ShapeNet1D_Test,Pascal1D_Test --T 20 --T_val 20 --T_test 20 --runs 1 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name finetuning_regr --model_path models/finetuning_multi_task/best-model.pkl --best_hp_file_name finetuning_multi_hpo --model finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --test_od_datasets ShapeNet2D_Test,Distractor_Test,ShapeNet1D_Test,Pascal1D_Test --T 20 --T_val 20 --T_test 20 --runs 1 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name linear_readout_regr --model_path models/linear_readout_multi_task/best-model.pkl --best_hp_file_name linear_readout_multi_hpo --model finetuning --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --test_od_datasets ShapeNet2D_Test,Distractor_Test,ShapeNet1D_Test,Pascal1D_Test --T 20 --T_val 20 --T_test 20 --freeze --runs 1 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
"--experiment_name tfs_regr --model_path models/tfs_multi_task/best-model.pkl --best_hp_file_name tfs_multi_hpo --model tfs --n_way_eval 5 --max_shots_eval 5 --train_datasets FSS_Trn,BCT_Mini_Trn,BRD_Mini_Trn,CRS_Mini_Trn,Animal_Pose_Trn --test_od_datasets ShapeNet2D_Test,Distractor_Test,ShapeNet1D_Test,Pascal1D_Test --T 20 --T_val 20 --T_test 20 --runs 1 --eval_iters 600 --root_dir ${META_OMNIUM_DIR}"
)

for ARG in "${ARGS[@]}"
do
python metaomnium/trainers/eval_cross_task_fsl.py ${ARG}
done
