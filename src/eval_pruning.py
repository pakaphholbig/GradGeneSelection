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
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

sys.path.append(os.path.dirname(os.getcwd()))  # parent of `perturbgene` directory

from perturbgene.model import GeneBertForPhenotypicMLM
from perturbgene.attribution_utils import (
    get_model,
    get_tokenizer,
    cell_resampling,
)
from perturbgene.gene_pruning.attribution import AttributionAnalysis
from perturbgene.data_utils.tokenization import phenotype_to_token

seed = 50
np.random.seed(seed)
torch.manual_seed(seed)

import warnings

warnings.simplefilter("ignore", category=UserWarning)


def calculate_metrics(predictions, ground_truth):
    """
    Calculate accuracy, F1 score, recall, and precision.

    Args:
    predictions (list or array): Predicted values.
    ground_truth (list or array): True values.

    Returns:
    dict: A dictionary containing all calculated metrics.
    """
    # Ensure both inputs are the same length
    if len(predictions) != len(ground_truth):
        raise ValueError(
            "Predictions and ground truth arrays must have the same length."
        )

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, average="weighted")
    recall = recall_score(ground_truth, predictions, average="weighted")
    precision = precision_score(ground_truth, predictions, average="weighted")

    # Store metrics in a dictionary
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "recall": recall,
        "precision": precision,
    }


@torch.no_grad()
def gene_pruning_wrapper(
    model_path: str,
    prune_ratios: list[float],
    file_directory: str,
    attribution_path: str,
    save_path: str,
    heuristic: str,
    method: str,
    device: str,
    masked_category: str,
    target_phenotype: str,
    baseline_phenotype: str,
    query: dict[str, list[str]],
    reverse: bool,
):

    model = get_model(GeneBertForPhenotypicMLM, model_path, device)
    model.to(device)
    model.eval()
    model.zero_grad()
    model_type = "mlm"

    # Step 2: Load tokenizer and read attribution data
    tokenizer = get_tokenizer(model_type, model_path, None)
    df = pd.read_csv(attribution_path)

    # Step 3: Restore attribution data
    num_data = df["index"].max() + 1
    # iterate over all cells
    prune_preds = {
        "target": {p: [] for p in prune_ratios},
        "baseline": {p: [] for p in prune_ratios},
    }

    hvg_importance = pd.read_csv("sample_HVG.csv")["rank"]
    for i in tqdm(range(num_data), "Gene Pruning n_data ="):

        row = df.loc[df["index"] == i]
        target_file, baseline_file = row[["target_file", "baseline_file"]].iloc[0]
        target_idx, baseline_idx = row[["target_file_idx", "baseline_file_idx"]].iloc[0]
        raw_data = row[["attribution", "rev_attrbution"]].to_numpy()

        target = cell_resampling(
            target_file,
            file_directory,
            tokenizer,
            target_idx,
            model_type=GeneBertForPhenotypicMLM,
            masked_category=masked_category,
            phenotype=target_phenotype,
            query=query,
        )

        # if the baseline is the empty input (all paddings)
        if baseline_file == "blank":
            baseline = None
        else:
            baseline = cell_resampling(
                baseline_file,
                file_directory,
                tokenizer,
                baseline_idx,
                model_type=GeneBertForPhenotypicMLM,
                masked_category=masked_category,
                phenotype=baseline_phenotype,
                query=query,
            )

        attr = AttributionAnalysis(
            model=model,
            embedding=model.distilbert.embeddings,
            method=method,
            target=target,
            baseline=baseline,
            tokenizer=tokenizer,
            device=device,
        )

        attr.gene_blending()
        attri, rev_attri = map(torch.tensor, raw_data.T)
        attr.restore_attribution(attri, rev_attri)

        # Step 4: Prune and evaluate the performance
        for p in prune_ratios:
            pruned_target, pruned_baseline = attr.gene_pruning(
                prune_ratio=p,
                heuristic=heuristic,
                reverse=reverse,
                hvg_importance=hvg_importance,
            )
            p_target_pred = attr.predict(pruned_target, masked_category)
            p_baseline_pred = attr.predict(pruned_baseline, masked_category)
            prune_preds["target"][p].append(p_target_pred[1].cpu().item())
            prune_preds["baseline"][p].append(p_baseline_pred[1].cpu().item())

    # calculate stats
    prune_stats = {
        "target": {},
        "baseline": {},
    }
    for input, preds in prune_preds.items():
        for p, labels in preds.items():
            if input == "target":
                ground_truth = tokenizer.flattened_tokens.index(
                    phenotype_to_token(target_phenotype)
                )
            else:
                ground_truth = (
                    tokenizer.flattened_tokens.index(
                        phenotype_to_token(baseline_phenotype)
                    )
                    if baseline_phenotype != "blank"
                    else 0
                )
            prune_stats[input][p] = calculate_metrics(
                labels, [ground_truth] * len(labels)
            )
    with open(save_path, "w") as fp:
        json.dump(prune_stats, fp, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Gene Pruning Metrics Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--prune_ratios",
        type=float,
        nargs="+",
        required=True,
        help="List of prune ratios.",
    )
    parser.add_argument(
        "--file_directory",
        type=str,
        required=True,
        help="the parent directory that stores data",
    )

    parser.add_argument(
        "--attribution_path",
        type=str,
        required=True,
        help="path to the stored attribution value data",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="path to save .json file",
    )

    parser.add_argument(
        "--heuristic",
        type=str,
        required=True,
        help="attribution heuristic",
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
        "--query",
        type=str,
        default="{}",
        help="additional categories to query from",
    )

    parser.add_argument(
        "--reverse",
        type=bool,
        default=False,
        help="a boolean indicating whether to reverse the importance",
    )

    args = parser.parse_args()
    assert args.method in ["ig", "dl"], (
        "method must be either Integrated Graidents (ig) or DeepLift (dl), but got"
        + args.method
    )
    args.query = json.loads(args.query)
    args_dict = vars(args)

    gene_pruning_wrapper(**args_dict)
