#!/bin/bash


# # test
python gene_pruning/run_attribution.py \
--model_path attribution_data/checkpoint-1360000 \
--file_directory attribution_sets \
--save_path 1_test_truth_blank.csv \
--method dl \
--device cpu \
--masked_category disease \
--target_phenotype "breast cancer" \
--baseline_phenotype normal \
--embedding_type "total" \
--query '{"cell_type": ["fibroblast"], "tissue": ["breast"]}' \
--prob_phen "breast cancer" \
--n_samples 1 \
--internal_batch_size 10 \
--n_steps 1 \
--two_way 1 \
--return_convergence 1

# python gene_pruning/run_attribution.py \
# --model_path attribution_data/checkpoint-1360000 \
# --file_directory attribution_sets \
# --save_path 2_test_truth_blank.csv \
# --method dl \
# --device cpu \
# --masked_category disease \
# --target_phenotype "breast cancer" \
# --baseline_phenotype normal \
# --embedding_type "total" \
# --query '{"cell_type": ["fibroblast"], "tissue": ["breast"]}' \
# --prob_phen "truth" \
# --n_samples 1 \
# --internal_batch_size 10 \
# --n_steps 1 \
# --two_way 1 \
# --return_convergence 1

# python gene_pruning/run_attribution.py \
# --model_path attribution_data/checkpoint-1360000 \
# --file_directory attribution_sets \
# --save_path test_one_way.csv \
# --method dl \
# --device cpu \
# --masked_category disease \
# --target_phenotype "breast cancer" \
# --baseline_phenotype normal \
# --embedding_type "total" \
# --prob_phen "breast cancer" \
# --query '{"cell_type": ["fibroblast"], "tissue": ["breast"]}' \
# --n_samples 2 \
# --internal_batch_size 10 \
# --n_steps 1 \
# --return_convergence 1

##### Script ##### 

python gene_pruning/run_attribution.py \
--model_path attribution_data/checkpoint-1360000 \
--file_directory /media/rohola/ssd_storage/primary \
--save_path blank_breast_cancer.csv \
--method dl \
--device cuda \
--masked_category disease \
--target_phenotype "breast cancer" \
--baseline_phenotype blank \
--embedding_type "total" \
--prob_phen "breast_cancer" \
--query '{"cell_type": ["fibroblast"], "tissue": ["breast"]}' \
--n_samples 1000 \
--internal_batch_size 10 \
--n_steps 1 \
--two_way 1 \
--return_convergence 1

python gene_pruning/run_attribution.py \
--model_path attribution_data/checkpoint-1360000 \
--file_directory /media/rohola/ssd_storage/primary \
--save_path blank_alzheimer.csv \
--method dl \
--device cuda \
--masked_category disease \
--target_phenotype "Alzheimer disease" \
--baseline_phenotype blank \
--embedding_type "total" \
--prob_phen "breast_cancer" \
--query '{"cell_type": ["microglial cell"]}' \
--n_samples 1000 \
--internal_batch_size 3 \
--n_steps 1 \
--two_way 1 \
--return_convergence 1

python gene_pruning/run_attribution.py \
--model_path attribution_data/checkpoint-1360000 \
--file_directory /media/rohola/ssd_storage/primary \
--save_path blank_dilated_cardiomyopathy.csv \
--method dl \
--device cuda \
--masked_category disease \
--target_phenotype "dilated cardiomyopathy" \
--baseline_phenotype blank \
--embedding_type "total" \
--prob_phen "breast_cancer" \
--query '{"tissue": ["heart left ventricle", "interventricular septum", "heart right ventricle"]}' \
--n_samples 1000 \
--internal_batch_size 10 \
--n_steps 1 \
--two_way 1 \
--return_convergence 1

