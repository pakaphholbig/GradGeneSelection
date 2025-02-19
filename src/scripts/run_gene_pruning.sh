#!/bin/bash
# python gene_pruning/run_gene_pruning.py \
# --model_path attribution_data/checkpoint-1360000 \
# --prune_ratios 0.80 0.825 \
# --file_directory attribution_sets \
# --attribution_path "2_test_truth_blank.csv" \
# --save_path test.json \
# --heuristic one_way \
# --method dl \
# --device cpu \
# --masked_category disease \
# --target_phenotype "breast cancer" \
# --baseline_phenotype blank \
# --query '{"cell_type": ["fibroblast"], "tissue": ["breast"]}'

# python gene_pruning/run_gene_pruning.sh \
# --model_path attribution_data/checkpoint-1360000 \
# --prune_ratios 0 0.1 0.25 0.35 0.50 0.60 0.70 0.75 0.775 0.80 0.825 0.85 0.875 0.9 0.925 0.95 0.99 \
# --file_directory attribution_sets \
# --attribution_path 2_test_truth_blank.csv \
# --save_path test.json \
# --heuristic one_way \
# --method dl \
# --device cpu \
# --masked_category disease \
# --target_phenotype "breast cancer" \
# --baseline_phenotype blank \
# --query '{"cell_type": ["fibroblast"], "tissue": ["breast"]}'

#code 
#am
# python gene_pruning/gene_pruning.py \
# --model_path attribution_data/checkpoint-1360000 \
# --prune_ratios 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
# --file_directory /media/rohola/ssd_storage/primary \
# --attribution_path target_breast.csv \
# --save_path target_breast_cancer.json \
# --heuristic am \
# --method dl \
# --device cuda \
# --masked_category disease \
# --target_phenotype "breast cancer" \
# --baseline_phenotype normal \
# --query '{"cell_type": ["fibroblast"], "tissue": ["breast"]}'

# python gene_pruning/gene_pruning.py \
# --model_path attribution_data/checkpoint-1360000 \
# --prune_ratios 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
# --file_directory /media/rohola/ssd_storage/primary \
# --attribution_path target_alzheimer.csv \
# --save_path target_alzheimer_am.json \
# --heuristic am \
# --method dl \
# --device cuda \
# --masked_category disease \
# --target_phenotype "Alzheimer disease" \
# --baseline_phenotype normal \
# --query '{"cell_type": ["microglial cell"]}'

# python gene_pruning/gene_pruning.py \
# --model_path attribution_data/checkpoint-1360000 \
# --prune_ratios 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
# --file_directory /media/rohola/ssd_storage/primary \
# --attribution_path target_dilated.csv \
# --save_path target_dilated_am.json \
# --heuristic am \
# --method dl \
# --device cuda \
# --masked_category disease \
# --target_phenotype "dilated cardiomyopathy" \
# --baseline_phenotype normal \
# --query '{"tissue": ["heart left ventricle", "interventricular septum", "heart right ventricle"]}'

#random
# python gene_pruning/gene_pruning.py \
# --model_path attribution_data/checkpoint-1360000 \
# --prune_ratios 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
# --file_directory /media/rohola/ssd_storage/primary \
# --attribution_path target_breast.csv \
# --save_path target_breast_random.json \
# --heuristic random \
# --method dl \
# --device cuda \
# --masked_category disease \
# --target_phenotype "breast cancer" \
# --baseline_phenotype normal \
# --query '{"cell_type": ["fibroblast"], "tissue": ["breast"]}'

# python gene_pruning/gene_pruning.py \
# --model_path attribution_data/checkpoint-1360000 \
# --prune_ratios 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
# --file_directory /media/rohola/ssd_storage/primary \
# --attribution_path target_alzheimer.csv \
# --save_path target_alzheimer_random.json \
# --heuristic random \
# --method dl \
# --device cuda \
# --masked_category disease \
# --target_phenotype "Alzheimer disease" \
# --baseline_phenotype normal \
# --query '{"cell_type": ["microglial cell"]}'

# python gene_pruning/gene_pruning.py \
# --model_path attribution_data/checkpoint-1360000 \
# --prune_ratios 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
# --file_directory /media/rohola/ssd_storage/primary \
# --attribution_path target_dilated.csv \
# --save_path target_dilated_random.json \
# --heuristic random \
# --method dl \
# --device cuda \
# --masked_category disease \
# --target_phenotype "dilated cardiomyopathy" \
# --baseline_phenotype normal \
# --query '{"tissue": ["heart left ventricle", "interventricular septum", "heart right ventricle"]}'

#oneway
# python gene_pruning/gene_pruning.py \
# --model_path attribution_data/checkpoint-1360000 \
# --prune_ratios 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
# --file_directory /media/rohola/ssd_storage/primary \
# --attribution_path target_breast.csv \
# --save_path target_breast_one_way.json \
# --heuristic one_way \
# --method dl \
# --device cuda \
# --masked_category disease \
# --target_phenotype "breast cancer" \
# --baseline_phenotype normal \
# --query '{"cell_type": ["fibroblast"], "tissue": ["breast"]}'

# python gene_pruning/gene_pruning.py \
# --model_path attribution_data/checkpoint-1360000 \
# --prune_ratios 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
# --file_directory /media/rohola/ssd_storage/primary \
# --attribution_path target_alzheimer.csv \
# --save_path target_alzheimer_one_way.json \
# --heuristic one_way \
# --method dl \
# --device cuda \
# --masked_category disease \
# --target_phenotype "Alzheimer disease" \
# --baseline_phenotype normal \
# --query '{"cell_type": ["microglial cell"]}'

# python gene_pruning/gene_pruning.py \
# --model_path attribution_data/checkpoint-1360000 \
# --prune_ratios 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
# --file_directory /media/rohola/ssd_storage/primary \
# --attribution_path target_dilated.csv \
# --save_path target_dilated_one_way.json \
# --heuristic one_way \
# --method dl \
# --device cuda \
# --masked_category disease \
# --target_phenotype "dilated cardiomyopathy" \
# --baseline_phenotype normal \
# --query '{"tissue": ["heart left ventricle", "interventricular septum", "heart right ventricle"]}'

#hvg
python gene_pruning/gene_pruning.py \
--model_path attribution_data/checkpoint-1360000 \
--prune_ratios 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
--file_directory /media/rohola/ssd_storage/primary \
--attribution_path target_breast.csv \
--save_path target_breast_hvg.json \
--heuristic hvg \
--method dl \
--device cuda \
--masked_category disease \
--target_phenotype "breast cancer" \
--baseline_phenotype normal \
--query '{"cell_type": ["fibroblast"], "tissue": ["breast"]}'

python gene_pruning/gene_pruning.py \
--model_path attribution_data/checkpoint-1360000 \
--prune_ratios 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
--file_directory /media/rohola/ssd_storage/primary \
--attribution_path target_alzheimer.csv \
--save_path target_alzheimer_hvg.json \
--heuristic hvg \
--method dl \
--device cuda \
--masked_category disease \
--target_phenotype "Alzheimer disease" \
--baseline_phenotype normal \
--query '{"cell_type": ["microglial cell"]}'

python gene_pruning/gene_pruning.py \
--model_path attribution_data/checkpoint-1360000 \
--prune_ratios 0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99 \
--file_directory /media/rohola/ssd_storage/primary \
--attribution_path target_dilated.csv \
--save_path target_dilated_hvg.json \
--heuristic hvg \
--method dl \
--device cuda \
--masked_category disease \
--target_phenotype "dilated cardiomyopathy" \
--baseline_phenotype normal \
--query '{"tissue": ["heart left ventricle", "interventricular septum", "heart right ventricle"]}'