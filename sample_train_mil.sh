#!/bin/bash
# conda activate pytorch

learning_rates=(0.0002 0.0001)
dropout_patch=(0.4 0.2)
dropout_node=(0.0)
num_epochs=(40)
weight_decay=(0.01 0.005)
cross_val_fold=(0 1 2 3 4)
top_k=(20)

# Function to run the python script
run_script() {	
	echo "Running with hyperparameters: $1, $2, $3, $4, $5, $6, $7"
	
	# SI-MIL (ours)
    python train.py  --organ 'test_organ' --dataset_split_path 'train_5x_dict.pickle' --dataset_split_path_test 'test_5x_dict.pickle'   --dataset_split_deep_path 'train_5x_dict_dino.pickle' --dataset_split_deep_path_test 'test_5x_dict_dino.pickle' --features_deep_path 'trainfeat_dino.pth'  --features_deep_path_test 'testfeat_dino.pth' --features_path 'binned_hcf.csv' --save_path '/MIL_experiment/ABMIL' --dropout_patch $1 --num_epochs $2 --weight_decay $3 --lr $4 --top_k $6  --gpu_index 1    --use_additive 'yes'  --dropout_node $5  --no_projection 'yes'   --feat_type 'patchattn_with_featattn_mlpmixer_Lmse'  --stop_gradient 'no'  --cross_val_fold $7 --temperature 3.0	 &
	# Lmse for Knowledge distillation
	
	
	# w/o PAG Top-K since stop_gradient is yes
    python train.py  --organ 'test_organ' --dataset_split_path 'train_5x_dict.pickle' --dataset_split_path_test 'test_5x_dict.pickle'   --dataset_split_deep_path 'train_5x_dict_dino.pickle' --dataset_split_deep_path_test 'test_5x_dict_dino.pickle' --features_deep_path 'trainfeat_dino.pth'  --features_deep_path_test 'testfeat_dino.pth' --features_path 'binned_hcf.csv' --save_path '/MIL_experiment/ABMIL' --dropout_patch $1 --num_epochs $2 --weight_decay $3 --lr $4 --top_k $6  --gpu_index 1    --use_additive 'yes'  --dropout_node $5  --no_projection 'yes'   --feat_type 'patchattn_with_featattn_mlpmixer_Lmse'  --stop_gradient 'yes'  --cross_val_fold $7 --temperature 3.0	 &


	# w/o KD since feat_type doesn't has Lmse in it's name
    python train.py  --organ 'test_organ' --dataset_split_path 'train_5x_dict.pickle' --dataset_split_path_test 'test_5x_dict.pickle'   --dataset_split_deep_path 'train_5x_dict_dino.pickle' --dataset_split_deep_path_test 'test_5x_dict_dino.pickle' --features_deep_path 'trainfeat_dino.pth'  --features_deep_path_test 'testfeat_dino.pth' --features_path 'binned_hcf.csv' --save_path '/MIL_experiment/ABMIL' --dropout_patch $1 --num_epochs $2 --weight_decay $3 --lr $4 --top_k $6  --gpu_index 1    --use_additive 'yes'  --dropout_node $5  --no_projection 'yes'   --feat_type 'patchattn_with_featattn_mlpmixer'  --stop_gradient 'no'  --cross_val_fold $7 --temperature 3.0	 &


	# w/o PAG Top-K and KD since stop_gradient is yes and feat_type doesn't has Lmse in it's name
    python train.py  --organ 'test_organ' --dataset_split_path 'train_5x_dict.pickle' --dataset_split_path_test 'test_5x_dict.pickle'   --dataset_split_deep_path 'train_5x_dict_dino.pickle' --dataset_split_deep_path_test 'test_5x_dict_dino.pickle' --features_deep_path 'trainfeat_dino.pth'  --features_deep_path_test 'testfeat_dino.pth' --features_path 'binned_hcf.csv' --save_path '/MIL_experiment/ABMIL' --dropout_patch $1 --num_epochs $2 --weight_decay $3 --lr $4 --top_k $6  --gpu_index 1    --use_additive 'yes'  --dropout_node $5  --no_projection 'yes'   --feat_type 'patchattn_with_featattn_mlpmixer'  --stop_gradient 'yes'  --cross_val_fold $7 --temperature 3.0	 &



}


# Loop over the hyperparameters and call the function
num_parallel=0

for ne in ${num_epochs[@]}; do
    for dn in ${dropout_node[@]}; do
        for dp in ${dropout_patch[@]}; do
			for lr in ${learning_rates[@]}; do
                for wd in ${weight_decay[@]}; do
                    for tk in ${top_k[@]}; do
                        for cv in ${cross_val_fold[@]}; do
                            run_script $dp $ne $wd $lr $dn $tk $cv
                            num_parallel=$(($num_parallel+1))
                            if [[ $num_parallel -ge 5 ]]; then
                                wait
                                num_parallel=0
                            fi
                        done
					done
				done
			done
		done
	done
done

wait

