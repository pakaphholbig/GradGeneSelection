from typing import Callable

import torch
from torch.distributions import Categorical
from anndata import AnnData
from transformers.modeling_outputs import SequenceClassifierOutput
from tqdm import tqdm
import json
import os
import transformers
import pickle
import pandas as pd
import numpy as np
import scanpy
from captum.attr import LayerIntegratedGradients, LayerDeepLift

from perturbgene.configs import BaseConfig
from perturbgene.data_utils.tokenization import phenotype_to_token, GeneTokenizer
from perturbgene.model import GeneBertForPhenotypicMLM, GeneBertForClassification
from perturbgene.data_utils import read_h5ad_file


def get_phenotype_categories(phenotypic_tokens_map_path: str) -> list[str]:
    """
    Return all phenotype categories given a path to a map between phenotype categories and phenotypes

    Args:
        `phenotypic_tokens_map_path`: an absolute path to phenotypic_tokens_map_path

    Return: a list of all phenotype categories

    """
    with open(phenotypic_tokens_map_path) as phenotype_maps:
        phenotype_categories = json.load(phenotype_maps).keys()
    return list(phenotype_categories)


def get_tokenizer(
    model_type: str, model_checkpt_path: str, phenotype_category: list[str]
) -> GeneTokenizer:
    """
    Get an associated tokenizer to the validation dataset

    Args:
        `model_type`: the task we ask the model to perform on (either `mlm` or `cls`)
        `model_checkpt_path`: the path to the model checkpoint
        `phenotype_category`: a list of all phenotype categories

    Return: a tokenizer that associates with the validation dataset
    """
    assert model_type in ("mlm", "cls")
    absolute_path = os.path.dirname(__file__)

    expected_tokenizer_path = os.path.join(
        os.path.dirname(model_checkpt_path),
        "tokenizer.pkl",
    )
    with open(expected_tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    tokenizer.config.vocab_path = os.path.join(
        absolute_path,
        "attribution_data/phenotypic_tokens_map.json",  # FIX_ME
    )

    if model_type == "mlm":
        tokenizer.config.binary_label = None
        tokenizer.config.phenotype_category = phenotype_category
    elif model_type == "cls":
        assert (
            tokenizer.config.phenotype_category == phenotype_category
        ), tokenizer.config.phenotype_category
    else:
        raise NotImplementedError

    return tokenizer


def get_model(
    model_class: GeneBertForPhenotypicMLM | GeneBertForClassification,
    model_checkpt_path: str,
    device: str,
) -> transformers.DistilBertPreTrainedModel:
    """
    Return a Ready-to-Use model for input evaluation

    Args:
        `model_class`: the class of model (defined in model.py)
        `model_checkpt_path`: the path to the model checkpoint
        `device`: a Torch device to evaluate input (either `cpu` or `cuda`)

    Return: a tokenizer that associates with the validation dataset
    """
    model = model_class.from_pretrained(model_checkpt_path)
    model.eval()
    model.to(device)
    return model


def get_inference_config(
    bin_edges: list[int],
    pretrained_model_path: str,
    max_length: int,
    num_top_genes: int,
    vocab_path: str = "/Users/biggu/Desktop/perturbgene/entropy_inference/inference_data/phenotypic_tokens_map.json",  # FIX_ME
    per_device_eval_batch_size: int = 4096,
):
    """
    Only need a subset of BaseConfig for inference. This config will mainly be used for creating `GeneTokenizer`s.
    """
    return BaseConfig(
        subcommand=None,
        bin_edges=bin_edges,
        bin_edges_path=None,
        pretrained_model_path=pretrained_model_path,
        model_arch=None,
        shard_size=None,
        eval_data_paths=[],
        max_length=max_length,
        num_top_genes=num_top_genes,
        vocab_path=vocab_path,
        included_phenotypes=None,
        use_flash_attn=True,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=0,
        auto_find_batch_size=False,
        output_dir=None,
    )


def prepare_cell(
    cell: AnnData,
    model_type: str,
    tokenizer: GeneTokenizer,
    label2id: dict[str, int] = None,
) -> dict[str, torch.Tensor]:
    """
    Converts an h5ad cell to `input_ids`.

    Args:
        cell: AnnData object with n_obs = 1
        model_type: Expecting "mlm" or "cls"
        tokenizer: To encode cell into `input_ids` and `token_type_ids`
        label2id: Only required for model_type == "cls"

    Returns : a dictionary to be used for data_collator function
    """
    input_ids, token_type_ids = tokenizer(cell)
    cell_data = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
    }

    if model_type == "cls":
        label_name = cell.obs[tokenizer.config.phenotype_category].item()
        cell_data["labels"] = torch.tensor(
            label2id[phenotype_to_token(label_name)], dtype=torch.long
        ).unsqueeze(0)
    return cell_data


def query_phenotypes_h5ad_file(val_data, **kwargs):
    for category, phenotypes in kwargs.items():
        phen = val_data.obs
        val_data = val_data[phen[category].isin(phenotypes)]
    return val_data


def cell_resampling(
    file,
    file_directory,
    tokenizer,
    cell_idx,
    masked_category,
    phenotype,
    query,
    model_type=GeneBertForPhenotypicMLM,
):
    if file == "blank":
        return None
    val_data = read_h5ad_file(
        f"{file_directory}/{file}",
        tokenizer.config.num_top_genes,
    )
    query[masked_category] = [phenotype]
    val_data = query_phenotypes_h5ad_file(val_data, **query)
    cell = prepare_cell(val_data[cell_idx, :], model_type, tokenizer)
    return cell


# Fix indexing
# fix to use somajoin_id instead
def cell_sampling(
    files_array,
    file_directory,
    tokenizer,
    phenotype,
    masked_category,
    query,
    model_type=GeneBertForPhenotypicMLM,
    limit=2000,
):
    """
    Sample cells with particular phenotypes across all the chunks
    """
    for i in range(limit):
        file = np.random.choice(files_array)
        val_data = read_h5ad_file(
            f"{file_directory}/{file}",
            tokenizer.config.num_top_genes,
        )
        query[masked_category] = [phenotype]
        val_data = query_phenotypes_h5ad_file(val_data, **query)
        if len(val_data) != 0:
            cell_idx = np.random.choice(len(val_data.obs))
            cell = prepare_cell(val_data[cell_idx, :], model_type, tokenizer)
            # To be removed
            if cell["token_type_ids"].shape[0] <= 3500:  # avoid cuda memory error
                return cell, cell_idx, file
    assert False, "Cell Sampling Error: Sampling Limit Exceeds"


def extract_phentypes_h5ad_file(validation_data, return_genes=False, **kwargs):

    phen = validation_data.obs  # we can query in the AnnData by its soma_joinid
    for category, phenotype in kwargs.items():
        phen = phen[phen[category].isin(phenotype)]

    # No cell with any of these phenotypes
    if len(phen) == 0:
        return None, None, False

    if not return_genes:
        return phen, None, True

    all_dfs = []
    for i in phen.index.values:  # we get only genes mapping from extracted phenotypes
        obs = validation_data[i].X
        all_dfs.append(
            pd.DataFrame(
                {"expr": obs.data, "gene_index": obs.indices}, index=[i] * len(obs.data)
            )
        )
    genes = pd.concat(all_dfs)
    name_map = pd.DataFrame(validation_data[0].var.index)
    genes["name"] = name_map.iloc[genes["gene_index"].values].values

    return phen, genes, True


def _reconstruct_input(input, indices, gene_offsets=None):
    # get the phenotypes and concatnate with genotypes in common
    for k, v in input.items():
        if len(v.shape) == 2:
            input[k] = v.squeeze(0)
        elif len(v.shape) > 2:
            raise Exception(
                "Each input can only either be a 1- or 2-dimensional tensor."
            )
    if gene_offsets is not None:
        return {
            k: torch.concat(
                (v[:gene_offsets], v[gene_offsets:-1][indices], v[-1:])
            ).unsqueeze(0)
            for k, v in input.items()
        }
    return {k: v[indices].unsqueeze(0) for k, v in input.items()}


def _input_baseline_intersection(input, baseline, gene_offsets):

    # somehow np.intersection sorts the result before returning
    input_genes_tokens, baseline_genes_tokens = (
        input["token_type_ids"][gene_offsets:-1],
        baseline["token_type_ids"][gene_offsets:-1],
    )
    _, input_indices, baseline_indices = np.intersect1d(
        input_genes_tokens, baseline_genes_tokens, return_indices=True
    )
    intersected_input = _reconstruct_input(input, input_indices, gene_offsets)
    intersected_baseline = _reconstruct_input(baseline, baseline_indices, gene_offsets)

    return [intersected_input, intersected_baseline]


def alternative_isin(elements, test_elements):
    elements = elements.unsqueeze(-1)
    test_elements = test_elements.unsqueeze(0)
    mask = (elements == test_elements).any(dim=-1)
    return mask


def _out_of_intersection_genes(input, intersected_input):
    masked_input = ~torch.isin(
        input["token_type_ids"], intersected_input["token_type_ids"]
    )
    input_only_genes = _reconstruct_input(input, masked_input)
    return input_only_genes


def sample_one_cell(val_data, phen_index, tokenizer, seed=None, model_type="mlm"):
    if seed is None:
        seed = np.random.randint(len(phen_index))
    val_idx = phen_index[seed]
    cell = prepare_cell(val_data[val_idx], model_type, tokenizer)
    return cell, val_idx


def sample_input_baseline_pair(
    validation_data, input, baseline, tokenizer, seed=None, model_type="mlm"
):
    if seed is None:
        input_indices = np.random.randint(0, len(input))
        baseline_indices = np.random.randint(
            0,
            len(baseline),
        )
    else:
        input_indices, baseline_indices = seed

    inp, bas = input[input_indices], baseline[baseline_indices]
    inp_cell, bas_cell = prepare_cell(
        validation_data[inp], model_type, tokenizer
    ), prepare_cell(validation_data[bas], model_type, tokenizer)
    return inp_cell, bas_cell, input_indices, baseline_indices


def all_input_baseline_pair(
    validation_data, input_indices, baseline_indices, tokenizer, model_type="mlm"
):
    all_pairs = []
    for inp_idx in input_indices:
        for bas_idx in baseline_indices:
            input, baseline = prepare_cell(
                validation_data[inp_idx], model_type, tokenizer
            ), prepare_cell(validation_data[bas_idx], model_type, tokenizer)
            all_pairs.append([input, baseline])
            # #padding is not needed bc they both have the same length anyway (a.k.a no need to collate)
    return all_pairs


def combined_input_baseline(
    input,
    baseline,
    tokenizer,
    device,
    ref_token_id=3,
    sep_token_id=1,
):
    """
    combined_input:    [CLS] [input intersected genes]    [input-only genes]                 [paddings for baseline-only genes] [EOS]
    combined_baseline: [CLS] [baseline intersected genes] [paddings for input-only genes]    [baseline-only genes]              [EOS]
    """
    input_intersect, baseline_intersect = _input_baseline_intersection(
        input, baseline, tokenizer.gene_token_type_offset
    )
    input_only, baseline_only = _out_of_intersection_genes(
        input, input_intersect
    ), _out_of_intersection_genes(baseline, baseline_intersect)

    combined_input = {}
    combined_baseline = {}
    combined_token_type_ref = []
    padding_information = {
        "input_ids": torch.tensor([[ref_token_id]], device=device),
        "token_type_ids": torch.tensor([[0]], device=device),
        "attention_mask": torch.tensor([[False]], device=device),
    }
    eos_information = {
        "input_ids": torch.tensor([[sep_token_id]], device=device),
        "token_type_ids": torch.tensor([[0]], device=device),
        "attention_mask": torch.tensor([[True]], device=device),
    }
    for k in input_intersect:
        baseline_size_paddings = padding_information[k].expand(
            1, baseline_only[k].shape[1]
        )
        input_size_paddings = padding_information[k].expand(1, (input_only[k].shape[1]))
        combined_input[k] = torch.concat(
            (
                input_intersect[k][:, :-1].to(device),
                input_only[k].to(device),
                baseline_size_paddings.to(device),
                eos_information[k].to(device),
            ),
            dim=-1,
        )
        combined_baseline[k] = torch.concat(
            (
                baseline_intersect[k][:, :-1].to(device),
                input_size_paddings.to(device),
                baseline_only[k].to(device),
                eos_information[k].to(device),
            ),
            dim=-1,
        ).to(device)
        assert (
            combined_input[k].shape == combined_baseline[k].shape
        ), f"{combined_input[k].shape=} != {combined_baseline[k].shape=}"
        if k == "token_type_ids":
            combined_token_type_ref = torch.concat(
                (
                    baseline_intersect[k][:, :-1].to(device),
                    input_only[k].to(device),
                    baseline_only[k].to(device),
                    eos_information[k].to(device),
                ),
                dim=-1,
            ).to(device)
    return combined_input, combined_baseline, combined_token_type_ref


def predict(model, input_ids, token_type_ids, attention_mask):
    model.eval()
    output = model(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )
    return output


def forward_pass(
    input_ids, token_type_ids, attention_mask, model, category_index, target=None
):  # mask category_index
    masked_input_ids = input_ids.clone()
    # Perform masking on the cloned tensor to avoid in-place modification

    masked_input_ids[:, category_index] = 2

    preds = predict(
        model, masked_input_ids, token_type_ids, attention_mask
    ).logits  # (batch_size(1), input, vocab_size)

    normalized = torch.softmax(preds[:, category_index, :], dim=-1)
    if target is None:
        value, pred = torch.max(normalized, dim=-1)
    else:  # Get the probability of the target
        value, pred = normalized[:, target], target
    return value


def fast_prediction(model, inputs, category_index):

    if len(inputs["input_ids"].shape) < 2:
        cloned_inputs = {
            k: v.clone().unsqueeze(0).to(model.device) for k, v in inputs.items()
        }
    else:
        cloned_inputs = {k: v.clone().to(model.device) for k, v in inputs.items()}

    cloned_inputs["input_ids"][:, category_index] = 2

    preds = predict(
        model,
        cloned_inputs["input_ids"],
        cloned_inputs["token_type_ids"],
        cloned_inputs["attention_mask"],
    ).logits  # (batch_size(1), input, vocab_size)
    normalized = torch.softmax(preds[:, category_index, :], dim=-1)
    value, index = torch.max(normalized, dim=-1)
    return value.item(), index.item()


def get_top_low_attributions(attrs, num_output=5, percentage=False):
    if percentage:
        num_output *= int(attrs.shape[0] / 100)
    top_values, top_indices = torch.topk(attrs, num_output)
    low_values, low_indices = torch.topk(-attrs, num_output)
    return top_values, top_indices, -low_values, low_indices


def eval_attribution_method(
    model,
    eval_method,
    input,
    baseline,
    internal_batch_size,
    category_idx,
    n_steps,
    return_convergence=True,
    target=None,
):
    if isinstance(eval_method, LayerIntegratedGradients):
        attributions, delta = eval_method.attribute(
            inputs=(
                input["input_ids"],
                input["token_type_ids"],
                input["attention_mask"],
            ),
            baselines=(
                baseline["input_ids"],
                baseline["token_type_ids"],
                baseline["attention_mask"],
            ),
            internal_batch_size=internal_batch_size,
            additional_forward_args=(model, category_idx, target),
            n_steps=n_steps,
            return_convergence_delta=return_convergence,
        )
    elif isinstance(eval_method, LayerDeepLift):
        attributions, delta = eval_method.attribute(
            inputs=(
                input["input_ids"],
                input["token_type_ids"],
                input["attention_mask"],
            ),
            baselines=(
                baseline["input_ids"],
                baseline["token_type_ids"],
                baseline["attention_mask"],
            ),
            additional_forward_args=(category_idx, target),
            return_convergence_delta=True,
        )

    attribution_by_token = attributions.sum(dim=-1).squeeze(0)
    attribution_by_token = attribution_by_token / torch.norm(attribution_by_token)
    return attribution_by_token, delta


def _get_token_name(tokenizer, indices):
    return tokenizer.flattened_tokens[indices]


def is_correct_prediction(
    input, baseline, category_index, model, tokenizer, verbose=False
):
    pred_input = fast_prediction(model, input, category_index)
    pred_baseline = fast_prediction(model, baseline, category_index)
    target_input = input["input_ids"][:, category_index].item()
    target_baseline = baseline["input_ids"][:, category_index].item()

    if (pred_input != target_input) or (pred_baseline != target_baseline):
        if verbose:
            print("There is a mismatch between model's prediction and targets.")
            if target_input != pred_input:
                print(
                    f"\t We expect to get a label = {_get_token_name(tokenizer, target_input)} for the input, but got {_get_token_name(tokenizer, pred_input)}."
                )
            elif target_baseline != pred_baseline:
                print(
                    f"\t We expect to get a label = {_get_token_name(tokenizer, target_baseline)} for the baseline, but got {_get_token_name(tokenizer, pred_baseline)}."
                )
        return False
    else:
        if verbose:
            print("Model's prediction matches the target.")
        return True


def all_input_baseline_pair(
    validation_data, input_indices, baseline_indices, tokenizer, model_type="mlm"
):
    all_pairs = []
    for inp_idx in input_indices:
        for bas_idx in baseline_indices:
            input, baseline = prepare_cell(
                validation_data[inp_idx], model_type, tokenizer
            ), prepare_cell(validation_data[bas_idx], model_type, tokenizer)
            all_pairs.append([input, baseline])
            # #padding is not needed bc they both have the same length anyway (a.k.a no need to collate)
    return all_pairs
