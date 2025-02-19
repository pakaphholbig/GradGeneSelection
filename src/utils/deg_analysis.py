import os, sys
import numpy as np
import torch
import pandas as pd
import scanpy as sc
from tqdm import tqdm

sys.path.append(os.path.dirname(os.getcwd()))  # parent of `perturbgene` directory
from perturbgene.attribution_utils import (
    prepare_cell,
    get_tokenizer,
    get_phenotype_categories,
    get_model,
    extract_phentypes_h5ad_file,
    _out_of_intersection_genes,
    _input_baseline_intersection,
    combined_input_baseline,
    forward_pass,
    eval_attribution_method,
    predict,
    sample_one_cell,
    query_phenotypes_h5ad_file,
    cell_sampling,
)
from perturbgene.data_utils import read_h5ad_file


import warnings

warnings.simplefilter("ignore", category=UserWarning)


file_directory = "/media/rohola/ssd_storage/primary"
# file_directory = "attribution_sets"
files_array = os.listdir(file_directory)
model_path = "attribution_data/checkpoint-1360000"
model_type = "mlm"
tokenizer = get_tokenizer(model_type, model_path, None)

attr_path = "target_alzheimer.csv"
df = pd.read_csv(attr_path)
files = pd.concat([df["target_file"], df["baseline_file"]]).unique()

query = {"cell_type": ["microglial cell"], "disease": ["normal", "Alzheimer disease"]}

# get queried cells
anndata = read_h5ad_file(
    file_directory + "/" + files[0],
    tokenizer.config.num_top_genes,
)
for category, phenotypes in query.items():
    phen = anndata.obs
    anndata = anndata[phen[category].isin(phenotypes)]


random_idx = np.random.randint(1, len(files), size=200)


for i in tqdm(range(len(random_idx))):
    cur_file = files[i]
    new_data = read_h5ad_file(
        file_directory + "/" + cur_file,
        tokenizer.config.num_top_genes,
    )
    for category, phenotypes in query.items():
        phen = new_data.obs
        new_data = new_data[phen[category].isin(phenotypes)]
    anndata = sc.concat(
        [anndata, new_data],
        join="outer",
    )

groupby_col = "disease"

# # 1. Get highly variable genes
# sc.pp.highly_variable_genes(anndata)
# hvg_data = anndata.var
# hvg_data.to_csv("sample_HVG.csv")

# 2. Perform DEG Analysis
sc.tl.rank_genes_groups(anndata, "disease", method="wilcoxon", key_added="wilcoxon")
rank_genes_groups = anndata.uns["wilcoxon"]

# Create a DataFrame for each group
groups = rank_genes_groups["names"].dtype.names  # Names of groups (categories)
dfs = []

for group in groups:
    # Extract information for each group
    df = pd.DataFrame(
        {
            "gene": rank_genes_groups["names"][group],
            "logfoldchange": rank_genes_groups["logfoldchanges"][group],
            "pval": rank_genes_groups["pvals"][group],
            "pval_adj": rank_genes_groups["pvals_adj"][group],
        }
    )
    df["group"] = group  # Add group label
    dfs.append(df)

# Combine all groups into a single DataFrame
result_df = pd.concat(dfs, ignore_index=True)
result_df.to_csv("alzheimer_DEG.csv")

# sc.pp.highly_variable_genes(anndata)
# rank = anndata.var["dispersions_norm"].rank(ascending=False)
# rank.to_csv("highly_variable_rank.csv")
