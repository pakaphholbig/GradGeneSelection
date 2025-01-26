import sys
import os
import pandas as pd
from captum.attr import LayerIntegratedGradients, LayerDeepLift
from tqdm import tqdm
import argparse
import torch, torch.nn
import numpy as np
import json
import pickle

sys.path.append(os.path.dirname(os.getcwd()))  # parent of `perturbgene` directory

from perturbgene.model import GeneBertForPhenotypicMLM
from perturbgene.attribution_utils import (
    get_model,
    get_tokenizer,
    cell_sampling,
)
from perturbgene.gene_pruning.attribution import AttributionAnalysis


seed = 50
np.random.seed(seed)
torch.manual_seed(seed)


@torch.no_grad()
def attribution_analysis_wrapper(
    model_path: str,
    file_directory: str,
    save_path: str,
    method: str,
    device: str,
    masked_category: str,
    target_phenotype: str,
    baseline_phenotype: str,
    embedding_type: str,
    prob_phen: str,
    query: dict[str, list[str]],
    n_samples: int,
    internal_batch_size: int,
    n_steps: int,
    two_way: bool,
    return_convergence: bool,
):

    # Step 1: Load model & embedding
    model = get_model(GeneBertForPhenotypicMLM, model_path, device)
    model.to(device)
    model.eval()
    model.zero_grad()
    model_type = "mlm"

    if embedding_type is None:
        print(
            "Warning: No embedding type specified. DistilBERT total embeddings will be used for the analysis."
        )
        embedding = model.distilbert.embeddings
    elif embedding_type == "total":
        embedding = model.distilbert.embeddings
    elif embedding_type == "token_type":
        embedding = model.distilbert.embeddings.token_type_embeddings
    elif embedding_type == "word":
        embedding = model.distilbert.embeddings.word_embedding
    elif embedding_type == "position":
        embedding = model.distilbert.embeddings.position_embeddings
    else:
        assert False, "Invalid Embedding Type"

    # Step 2: Load tokenizer
    tokenizer = get_tokenizer(model_type, model_path, None)

    # Step 3: Initialize an instance to store the result
    all_cells = pd.DataFrame()

    # Step 4: Sampling and attribution evaluation
    files_array = os.listdir(file_directory)
    for it in tqdm(range(n_samples), desc="Attribution Analysis"):

        if device == "cuda":
            torch.cuda.empty_cache()

        # Step 4.1: Sample target and baseline randomly from each chunk
        target, target_idx, target_file = cell_sampling(
            files_array,
            file_directory,
            tokenizer,
            target_phenotype,
            masked_category,
            query,
        )

        if baseline_phenotype == "blank":
            baseline, baseline_file, baseline_idx = None, "blank", 0
        else:
            baseline, baseline_idx, baseline_file = cell_sampling(
                files_array,
                file_directory,
                tokenizer,
                baseline_phenotype,
                masked_category,
                query,
            )

        # Step 4.2: Calculate Attribution
        attr = AttributionAnalysis(
            model=model,
            embedding=embedding,
            method=method,
            target=target,
            baseline=baseline,
            tokenizer=tokenizer,
            prob_phen=prob_phen,
            device=device,
            two_way=two_way,
        )

        attributions, delta, rev_attributions, rev_delta = attr.eval_attribution_method(
            internal_batch_size=internal_batch_size,
            masked_phenotype_category=masked_category,
            prob_phen=prob_phen,
            n_steps=n_steps,
            return_convergence=return_convergence,
        )

        # Step 4.3: Store the attribution values
        token_types = attr.blended_token_type_ref.squeeze(0).cpu().tolist()
        num_tokens = len(token_types)
        cell_data = pd.DataFrame(
            {
                "attribution": attributions.cpu().tolist(),
                "delta": [delta] * num_tokens,
                "rev_attrbution": (
                    rev_attributions.cpu().tolist()
                    if rev_attributions is not None
                    else [rev_attributions] * num_tokens
                ),
                "rev_delta": [rev_delta] * num_tokens,
                "token_type_id": token_types,
                "target_file": [target_file] * num_tokens,
                "baseline_file": [baseline_file] * num_tokens,
                "target_file_idx": [target_idx] * num_tokens,
                "baseline_file_idx": [baseline_idx] * num_tokens,
                "index": [it] * num_tokens,
            }
        )

        # Emergency Backup
        all_cells = pd.concat([all_cells, cell_data])
        # Emergency Backup
        if not (it + 1) % 10:
            print(f"Checkpoint {(it+1)//10+1} Backuped")
            all_cells.to_csv(save_path)
    all_cells.to_csv(save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Attribution Analysis")
    parser.add_argument(
        "--model_path",
        type=str,
        default="attribution_data/checkpoint-1360000",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--file_directory",
        type=str,
        default="attribution_set",
        help="the parent directory that stores data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="attributions.csv",
        help="path to save .csv file",
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Integrated Gradient (ig) or DeepLift (dl)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch.device type cuda or cpu",
    )
    parser.add_argument(
        "--masked_category",
        type=str,
        required=True,
        help="a category to perform the analysis",
    )

    parser.add_argument(
        "--target_phenotype",
        type=str,
        required=True,
        help="the phenotype of an input that we want to analyze (expect to be in masked_category)",
    )

    parser.add_argument(
        "--baseline_phenotype",
        type=str,
        required=True,
        help="the phenotype of a baseline that we want to analyze (expect to be in masked_category). If the baseline is `blank`, an empty input will be used (an input with only phenotypes but no genes)",
    )

    parser.add_argument(
        "--embedding_type",
        type=str,
        default="total",
        help="the BERT embedding type that we want to perform the analysis on (total, word, token_type, position)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="{}",
        help="additional categories to query from",
    )

    parser.add_argument(
        "--prob_phen",
        type=str,
        required=True,
        help="the target phenotype to calculate the attribution values from if it is 'max', we will return the highest probability across all the phenotypes. if it is 'truth', we will return probability of the ground truth label.",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        required=True,
        help="number of samples to perform analysis",
    )

    parser.add_argument(
        "--internal_batch_size",
        type=int,
        required=True,
        help="batch_size parameter for IntegratedGradients method",
    )

    parser.add_argument(
        "--n_steps",
        type=int,
        required=True,
        help="n_steps parameter for IntegratedGradients method (number of integral partitions)",
    )

    parser.add_argument(
        "--two_way",
        type=bool,
        default=False,
        help="a boolean indicating whether to do two_way attribution",
    )

    parser.add_argument(
        "--return_convergence",
        type=bool,
        default=True,
        help="return_convergence parameter for IntegratedGradients method",
    )

    args = parser.parse_args()
    assert args.method in ["ig", "dl"], (
        "method must be either Integrated Graidents (ig) or DeepLift (dl), but got"
        + args.method
    )
    args.query = json.loads(args.query)

    if args.method == "dl":
        print(
            "Warning: `internal_batch_size` and `n_steps` are not used in the DeepLift method."
        )

    args_dict = vars(args)
    for phenotypes in args.query.values():
        assert isinstance(
            phenotypes, list
        ), "values in query dictionary must be python lists."
    attribution_analysis_wrapper(**args_dict)
