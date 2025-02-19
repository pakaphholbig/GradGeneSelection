import torch
import numpy as np
from typing import Callable, TypedDict, Tuple, Optional
from captum.attr import LayerIntegratedGradients, LayerDeepLift
from perturbgene.data_utils.tokenization import GeneTokenizer, phenotype_to_token
from perturbgene.model import GeneBertForPhenotypicMLM
from perturbgene.attribution_utils import _reconstruct_input
import pandas as pd


# Interface
class ModelInput(TypedDict):
    """
    An interface for the format of input required by the PolyGene model.

    Attributes:
        - input_ids (torch.Tensor): a tensor of the input ids
        - token_type_ids (torch.Tensor): a tensor of the token type ids
        - attention_masks (torch.Tensor): a tensor of attention masks
    """

    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_masks: torch.Tensor


class AttributionAnalysis:
    """
    Represents an instance for attribution analysis

    Attributes:
        - model (GeneBertForPhenotypicMLM): a model for  the attribution analysis
        - embedding (torch.nn.Module): a target embedding for attribution method (see more at https://captum.ai/tutorials/Bert_SQUAD_Interpret)
        - method (str): an attribution method (either Intergrated Gradients (ig) or DeepLift (dl))
        - target (ModelInput): a parsed target instance
        - baseline (ModelInput): a parsed baseline instance
        - tokenizer (GeneTokenizer): a tokenizer associated with the model
        - device (torch.device): PyTorch device
        - two_way (bool): a boolean indicating whether to calculate two-way attributions

    """

    def __init__(
        self,
        model: GeneBertForPhenotypicMLM,
        embedding: torch.nn.Module,
        method: str,
        target: ModelInput,
        tokenizer: GeneTokenizer,
        device: torch.device,
        baseline: Optional[ModelInput] = None,
        two_way: bool = True,
    ):
        """
        Class Initialization

        Args:
            - model (GeneBertForPhenotypicMLM): a model for  the attribution analysis
            - embedding (torch.nn.Module): a target embedding for attribution method (see more at https://captum.ai/tutorials/Bert_SQUAD_Interpret)
            - method (str): an attribution method (either Intergrated Gradients (ig) or DeepLift (dl))
            - target (ModelInput): a parsed target instance
            - baseline (ModelInput): a parsed baseline instance, if it is None we will initialize a phenotype only instance
            - tokenizer (GeneTokenizer): a tokenizer associated with the model
            - device (torch.device): PyTorch device
            - two_way (bool): a boolean indicating whether to calculate two-way attributions

        """
        self.model = model
        self.embedding = embedding

        assert method in [
            "dl",
            "ig",
        ], f"expect method to be either dl or ig, but got {method}"

        self.method = method
        self._eval_method = self.initialize_attribution_method()
        self.tokenizer = tokenizer

        self._target = target
        self._baseline = (
            baseline if baseline is not None else self._construct_blank_instance()
        )

        for k, v in self._target.items():
            self._target[k] = v.to(device)
        for k, v in self._baseline.items():
            self._baseline[k] = v.to(device)

        self._blended_target, self._blended_baseline = None, None
        self._blended_token_type_ref = None

        self._attribution_values, self._delta = None, None
        self._rev_attribution_values, self._rev_delta = None, None

        self.two_way = two_way
        self.device = device

    def initialize_attribution_method(self) -> LayerIntegratedGradients | LayerDeepLift:
        """
        Return and initialize the Captum attribution methods corresponding with
        the method type.

        Note: As of now, DeepLift can only take a torch.nn module as an input.
        Hence, we need to define a torch.nn wrapper.
        """
        if self.method == "ig":
            return LayerIntegratedGradients(self.forward_pass, self.embedding)
        elif self.method == "dl":
            wrapper = DeepLiftWrapper(self.model, self)
            return LayerDeepLift(wrapper, self.embedding, multiply_by_inputs=True)

    def gene_blending(self) -> Tuple[ModelInput, ModelInput, torch.Tensor]:
        """
        Return the blended target, blended baseline, and blended token types
        for reference.

        Description:
            The algorithm aligns and pads, if necessary, the target and baseline inputs such that
            the blended target and baseline tokens have the same length and are all aligned correctly.

            The order of token_type_ids of the blended_input and blended_baseline will be as follows:
            blended_target:   [CLS] [phenotypes] [input intersected genes]    [input-only genes]                 [paddings for baseline-only genes] [EOS]
            blended_baseline: [CLS] [phenotypes] [baseline intersected genes] [paddings for input-only genes]    [baseline-only genes]              [EOS]
        """
        ref_token_id = self.tokenizer.flattened_tokens.index("[PAD]")
        sep_token_id = self.tokenizer.flattened_tokens.index("[EOS]")

        # Step 1: Get a intersection of genes in target and baseline
        common_gene_target, common_gene_baseline = self._input_baseline_intersection()

        # Step 2: Get all genes that are unique to target and baselie respectively
        unique_gene_target, unique_gene_baseline = self._out_of_intersection_genes(
            self._target, common_gene_target
        ), self._out_of_intersection_genes(self._baseline, common_gene_baseline)

        # Step 3: Combine them and pad if necessary
        blended_target = {}
        blened_baseline = {}
        blended_token_type = []

        # Initialize
        padding_information = {
            "input_ids": torch.tensor([[ref_token_id]], device=self.device),
            "token_type_ids": torch.tensor([[0]], device=self.device),
            "attention_mask": torch.tensor([[False]], device=self.device),
        }
        eos_information = {
            "input_ids": torch.tensor([[sep_token_id]], device=self.device),
            "token_type_ids": torch.tensor([[0]], device=self.device),
            "attention_mask": torch.tensor([[True]], device=self.device),
        }

        for k in common_gene_target:

            paddings_unique_baseline_size = padding_information[k].expand(
                1, unique_gene_baseline[k].shape[1]
            )
            paddings_unique_target_size = padding_information[k].expand(
                1, unique_gene_target[k].shape[1]
            )

            # Construct blended target
            blended_target[k] = torch.concat(
                (
                    common_gene_target[k][:, :-1],
                    unique_gene_target[k],
                    paddings_unique_baseline_size,
                    eos_information[k],
                ),
                dim=-1,
            )

            # Construct blended baseline
            blened_baseline[k] = torch.concat(
                (
                    common_gene_baseline[k][:, :-1],
                    paddings_unique_target_size,
                    unique_gene_baseline[k],
                    eos_information[k],
                ),
                dim=-1,
            )

            assert (
                blended_target[k].shape == blened_baseline[k].shape
            ), f"{blended_target[k].shape=} != {blened_baseline[k].shape=}"

            # Store the token_type_id for each position for future reference
            if k == "token_type_ids":
                blended_token_type = torch.concat(
                    (
                        common_gene_baseline[k][:, :-1],
                        unique_gene_target[k],
                        unique_gene_baseline[k],
                        eos_information[k],
                    ),
                    dim=-1,
                ).to(self.device)

        self._blended_target, self._blended_baseline = blended_target, blened_baseline
        self._blended_token_type_ref = blended_token_type

        return blended_target, blened_baseline, blended_token_type

    def forward_pass(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_phenotype_category: str,
        target_phen: Optional[str] = "max",
        return_prediction: bool = False,
    ) -> float | Tuple[float, int]:
        """
        Return the model's prediction (if specified) and its associated probability.

        Args:
            - input_ids (torch.Tensor): a tensor of an input's input_ids
            - token_type_ids (torch.Tensor): a tensor of an input's token_type_ids
            - attention_mask (torch.Tensor): a tensor of an input's attention_mask
            - masked_phenotype_category (str): a phenotypic category that the model predicts (this phenotype will be masked)
            - target_phen (str): this optional parameter indicates probability from which phenotype to be returned from the model.
                               if not specified, it will return the probability of the most probable labels.
                               if it is "max", we will return the highest probability across all the phenotypes.
                               otherwise, it will return the probability associated with that particular phenotype
            - return_prediction (bool): a boolean indicating whether to return the predicted label.

        Description:
            If gene_blending has not been called yet, this function will first blend the token_type_ids.
            Given an input to the model, masked_phenotype_category is masked and the model is tasked
            with predicting it. If phen is specified, the model will return the probability predicting phen.
            Otherwise, it will give the most probable prediction.
        """
        category_idx = 1 + self.tokenizer.config.included_phenotypes.index(
            masked_phenotype_category
        )
        masked_input_ids = input_ids.clone()
        # Perform masking on the cloned tensor to avoid in-place modification

        masked_input_ids[:, category_idx] = 2

        output = self.model(
            masked_input_ids.to(self.device),
            token_type_ids=token_type_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        ).logits
        normalized = torch.softmax(output[:, category_idx, :], dim=-1)

        # Get attribution
        if target_phen == "max":
            value, pred = torch.max(normalized, dim=-1)
        else:
            phen = self.tokenizer.flattened_tokens.index(
                phenotype_to_token(target_phen)
            )
            value, pred = normalized[:, phen], phen

        if return_prediction:
            return value, pred
        return value

    @torch.no_grad()
    def predict(
        self,
        input: ModelInput,
        masked_phenotype_category: str,
    ) -> Tuple[float, int] | float:
        """
        Return a dictionary mapping each kind of input to the model's prediction of the mask phenotypic category.

        Args:
            - input (ModelInput): an input to make predictions
            - masked_phenotype_category (str): a phenotypic category that the model predicts (this phenotype will be masked)
        """
        return self.forward_pass(
            input["input_ids"],
            input["token_type_ids"],
            input["attention_mask"],
            masked_phenotype_category,
            return_prediction=True,
        )

    @torch.no_grad()
    def eval_attribution_method(
        self,
        masked_phenotype_category: str,
        internal_batch_size: Optional[int] = None,
        target_phen: Optional[str] = "max",
        n_steps: Optional[int] = 10,
        return_convergence: bool = True,
    ) -> (
        Tuple[torch.tensor, int, torch.tensor, int]
        | Tuple[torch.tensor, int, None, None]
    ):
        """
        Return attribution values and convergence error calculated by an attribution analysis method
        (Integrated Gradient/DeepLift). If self.two_way is true, then it calculates both target -> baseline
        and baseline -> target attributions and stores them in its attributes.

        Args:
            - internal_batch_size (int): a bsz for the attribuion method
            - masked_phenotype_category (str): a phenotypic category that the model predicts (this phenotype will be masked)
            - n_steps (int): a number of steps for Integrated Gradients method (it is not required for DeepLift)
            - target_phen (str): this optional parameter indicates probability from which phenotype to be returned from the model.
                               if it is "max", we will return the highest probability across all the phenotypes.
            - return_convergence (bool): a boolean indicating whether to return the convergence error
        """
        eval_method = self.eval_method

        self.gene_blending()
        if isinstance(eval_method, LayerIntegratedGradients):
            attributions, delta = eval_method.attribute(
                inputs=(
                    self.blended_target["input_ids"],
                    self.blended_target["token_type_ids"],
                    self.blended_target["attention_mask"],
                ),
                baselines=(
                    self.blended_baseline["input_ids"],
                    self.blended_baseline["token_type_ids"],
                    self.blended_baseline["attention_mask"],
                ),
                internal_batch_size=internal_batch_size,
                additional_forward_args=(masked_phenotype_category, target_phen),
                n_steps=n_steps,
                return_convergence_delta=return_convergence,
            )
            if self.two_way:
                rev_attributions, rev_delta = eval_method.attribute(
                    inputs=(
                        self.blended_baseline["input_ids"],
                        self.blended_baseline["token_type_ids"],
                        self.blended_baseline["attention_mask"],
                    ),
                    baselines=(
                        self.blended_target["input_ids"],
                        self.blended_target["token_type_ids"],
                        self.blended_target["attention_mask"],
                    ),
                    internal_batch_size=internal_batch_size,
                    additional_forward_args=(masked_phenotype_category, target_phen),
                    n_steps=n_steps,
                    return_convergence_delta=return_convergence,
                )
        elif isinstance(eval_method, LayerDeepLift):
            attributions, delta = eval_method.attribute(
                inputs=(
                    self.blended_target["input_ids"],
                    self.blended_target["token_type_ids"],
                    self.blended_target["attention_mask"],
                ),
                baselines=(
                    self.blended_baseline["input_ids"],
                    self.blended_baseline["token_type_ids"],
                    self.blended_baseline["attention_mask"],
                ),
                additional_forward_args=(masked_phenotype_category, target_phen),
                return_convergence_delta=True,
            )
            if self.two_way:
                rev_attributions, rev_delta = eval_method.attribute(
                    inputs=(
                        self.blended_baseline["input_ids"],
                        self.blended_baseline["token_type_ids"],
                        self.blended_baseline["attention_mask"],
                    ),
                    baselines=(
                        self.blended_target["input_ids"],
                        self.blended_target["token_type_ids"],
                        self.blended_target["attention_mask"],
                    ),
                    additional_forward_args=(masked_phenotype_category, target_phen),
                    return_convergence_delta=return_convergence,
                )
        else:
            assert (
                False
            ), "Expect the method to be either LayerIntegratedGradients or LayerDeepLift"

        attribution_by_token = attributions.sum(dim=-1).squeeze(0)
        attribution_by_token = attribution_by_token / torch.norm(attribution_by_token)
        self._attribution_values, self._delta = attribution_by_token, delta.item()

        if self.two_way and not torch.any(rev_attributions.isnan()):
            if torch.any(rev_attributions.isnan()):
                print(
                    "There is a NaN value in reverse attributions. The reverse attributions will be ignored."
                )
            rev_attribution_by_token = rev_attributions.sum(dim=-1).squeeze(0)
            rev_attribution_by_token = rev_attribution_by_token / torch.norm(
                rev_attribution_by_token
            )
            self._rev_attribution_values, self._rev_delta = (
                rev_attribution_by_token,
                rev_delta.item(),
            )
        return (
            self._attribution_values,
            self._delta,
            self._rev_attribution_values,
            self._rev_delta,
        )

    @torch.no_grad()
    def gene_pruning(
        self,
        prune_ratio: int,
        heuristic: str,
        reverse: Optional[bool] = False,
        hvg_importance: Optional[int] = None,
    ) -> Tuple[ModelInput, ModelInput]:
        """
        Gene Pruning Algorithm
        """
        if heuristic == "gm" or heuristic is None:
            func = geometric_importance
            assert (
                self.attribution_values is not None
                and self.rev_attribution_values is not None
            ), "GM (two-way importance) requires both attributions and reversed attributions to be initialized"
        elif heuristic == "am":
            func = arithmetic_importance
            assert (
                self.attribution_values is not None
                and self.rev_attribution_values is not None
            ), "AM (two-way importance) requires both attributions and reversed attributions to be initialized"
        elif heuristic == "one_way":
            func = identity_importance(reverse=False)
            assert (
                self.attribution_values is not None
            ), "one_way importance requires the attributions to be initialized"
        elif heuristic == "rev_one_way":
            func = identity_importance(reverse=True)
            assert (
                self.rev_attribution_values is not None
            ), "rev_one_way importance requires the reversed attributions to be initialized"
        elif heuristic == "abs_one_way":
            func = absolute_importance(reverse=False)
            assert (
                self.attribution_values is not None
            ), "abs_one_way importance requires the attributions to be initialized"
        elif heuristic == "rev_abs_one_way":
            func = absolute_importance(reverse=True)
            assert (
                self.rev_attribution_values is not None
            ), "rev_abs_one_way importance requires the reversed attributions to be initialized"
        elif heuristic == "random":
            func = random_importance
        elif heuristic == "hvg":
            importance = hvg_importance
        else:
            raise Exception("Invalid heuristic")

        # Only prune genotypes not phenotypes
        offset = len(self.tokenizer.phenotypic_types) + 1
        num_genes = (
            self.blended_token_type_ref.shape[1] - offset - 1
        )  # remove heading (CLS + phenptypes) and [EOS]
        remaining_genes = int((1 - prune_ratio) * num_genes)

        if self.rev_attribution_values is None:
            rev_attr = None
        else:
            rev_attr = self.rev_attribution_values[offset:-1]

        if heuristic == "hvg":
            token_ref = self._blended_token_type_ref.squeeze(0)[offset:-1].cpu()
            rank = hvg_importance.iloc[token_ref].values
            importance = -torch.tensor(rank, device=self.device)
        else:
            importance = func(
                self.attribution_values[offset:-1],
                rev_attr,
                # avoid a case where reverse attribution is not calculated
            )
        if reverse:
            importance = -importance
        # negative sign refers to the opposite of disease attributions
        kept_gene_indices = torch.topk(importance, remaining_genes).indices

        # Prune the blended genes
        pruned_target = self._reconstruct_input(
            self.blended_target, kept_gene_indices, offset
        )
        pruned_baseline = self._reconstruct_input(
            self.blended_baseline, kept_gene_indices, offset
        )

        return (
            self._squeeze_input(pruned_target),
            self._squeeze_input(pruned_baseline),
        )

    def restore_attribution(
        self, attribution_values, rev_attribution_values: Optional[torch.tensor] = None
    ):
        self.gene_blending()
        self._attribution_values = attribution_values
        if torch.any(rev_attribution_values.isnan()):
            print(
                "There is at least one NaN value in the rev_attribution data, which will be ignored."
            )
            self._rev_attribution_values = None
        else:
            self._rev_attribution_values = rev_attribution_values

    # Helper Functions
    def _input_baseline_intersection(self) -> Tuple[ModelInput, ModelInput]:
        """
        A helper function to construct a model input consisting of the intersection between the target and baseline token_type_ids
        """
        gene_token_type_offset = self.tokenizer.gene_token_type_offset

        # Step 1: We only find the intersection of genotypes, not phenotypes or special tokens.
        target_genotype_tokens, baseline_genotype_tokens = (
            self._target["token_type_ids"][gene_token_type_offset:-1],
            self._baseline["token_type_ids"][gene_token_type_offset:-1],
        )

        _, target_common_genes_indices, baseline_common_genes_indices = np.intersect1d(
            target_genotype_tokens.cpu(),
            baseline_genotype_tokens.cpu(),
            return_indices=True,
        )

        # Step 2: We reconstruct the intersections between the target and baseline
        # according to the input format of the model.
        intersected_target = self._reconstruct_input(
            self._target, target_common_genes_indices, gene_token_type_offset
        )
        intersected_baseline = self._reconstruct_input(
            self._baseline, baseline_common_genes_indices, gene_token_type_offset
        )
        # Note: We need to do them separately because the target and baseline usually don't
        # have the same phenotypes.

        return (intersected_target, intersected_baseline)

    def _out_of_intersection_genes(
        self, input1: ModelInput, input2: ModelInput
    ) -> ModelInput:
        """
        A helper function to construct a model input consisting of token_type_ids in input1\input2

        Args:
            - input1 (ModelInput): a model input
            - input2 (ModelInput): another model input
        """
        masked_index = ~torch.isin(input1["token_type_ids"], input2["token_type_ids"])
        return self._reconstruct_input(input1, masked_index)

    def _reconstruct_input(
        self, input: ModelInput, indices: torch.Tensor, offset=None
    ) -> ModelInput:
        """
        A helper function to reconstruct the input by taking elements with specific indices.
        If input has been collated, this function will squeeze the extra dimension and raise expection
        if the dimensionality of the tensor is greater than 2.

        Args:
            - input (ModelInput): a model input
            - indices (torch.Tensor): the indices of elements we want to keep
            - offset (int): an index offset
        """
        # validate the input format
        copied_input = {}
        for k, v in input.items():
            if len(v.shape) == 1:
                copied_input[k] = v
            elif len(v.shape) == 2:
                copied_input[k] = v.squeeze(0)
            elif len(v.shape) > 2:
                raise Exception(
                    "Each input can only either be a 1- or 2-dimensional tensor."
                )
        # get the phenotypes and concatnate with genotypes in common
        if offset is not None:
            return {
                k: torch.concat(
                    (
                        v[:offset],
                        v[offset:-1][indices],
                        v[-1:],
                    )  # [[phenotypes], [genotypes], [EOS]]
                ).unsqueeze(0)
                for k, v in copied_input.items()
            }
        return {k: v[indices].unsqueeze(0) for k, v in copied_input.items()}

    def _squeeze_input(self, input):
        ref_token_id = self.tokenizer.flattened_tokens.index("[PAD]")
        input_ids = input["input_ids"]
        mask = ~(input_ids == ref_token_id).squeeze(0)
        return {k: v[:, mask] for k, v in input.items()}

    # Properties
    @property
    def attribution_values(self) -> torch.Tensor:
        """
        Return a tensor of attribution values for each token type (distinct phenotypes + genotypes) in
        the target and baseline
        """
        assert (
            self._attribution_values is not None
        ), "attribution_values has not been initialized yet."

        return self._attribution_values

    @property
    def delta(self) -> torch.Tensor:
        """
        Return a tensor whose dim = 0 representing the convergence error of the attribution method
        """
        assert self._delta is not None, "delta has not been initialized yet."
        return self._delta

    @property
    def rev_attribution_values(self) -> torch.Tensor:
        """
        Return a tensor of reverse attribution values for each token type (distinct phenotypes + genotypes) in
        the target and baseline
        """
        assert (
            self._attribution_values is not None
        ), "attribution_values has not been initialized yet."

        return self._rev_attribution_values

    @property
    def rev_delta(self) -> torch.Tensor:
        """
        Return a tensor whose dim = 0 representing the convergence error of the reverse attribution method
        """
        assert self._delta is not None, "delta has not been initialized yet."
        return self._rev_delta

    @property
    def eval_method(self) -> str:
        """
        Return a string representing the attribution method of the currect analysis
        """
        return self._eval_method

    @property
    def target(self) -> ModelInput:
        """
        Return the original instance of the target
        """
        return {k: v.unsqueeze(0) for k, v in self._target.items()}

    @property
    def baseline(self) -> ModelInput:
        """
        Return the original instance of the baseline
        """
        return {k: v.unsqueeze(0) for k, v in self._baseline.items()}

    @property
    def blended_target(self) -> ModelInput:
        """
        Return the blended instance of the target (i.e. merge the original target instance with
        the baseline instance according to the token_type_ids)
        """
        assert (
            self._blended_target is not None
        ), "blended_target has not been initialized yet."
        return self._blended_target

    @property
    def blended_baseline(self) -> ModelInput:
        """
        Return the blended instance of the target (i.e. merge the original baseline instance with
        the target instance according to the token_type_ids)
        """
        assert (
            self._blended_baseline is not None
        ), "blended_baseline has not been initialized yet."
        return self._blended_baseline

    @property
    def blended_token_type_ref(self) -> torch.tensor:
        """
        Return a tensor of token_type_ids such that their orders align with the blended target and baseline.
        """
        assert (
            self._blended_token_type_ref is not None
        ), "blended_token_type_ref has not been initialized yet."
        return self._blended_token_type_ref

    def _construct_blank_instance(self):
        mask = torch.arange(
            len(self.tokenizer.phenotypic_types) + 1
        )  # keep format and phenotypes
        target = self._target
        return {
            "input_ids": torch.cat(
                (target["input_ids"][mask], target["input_ids"][[-1]])
            ),
            "token_type_ids": torch.cat(
                (target["token_type_ids"][mask], target["token_type_ids"][[-1]])
            ),
            "attention_mask": torch.cat(
                (target["attention_mask"][mask], target["attention_mask"][[-1]])
            ),
        }


class DeepLiftWrapper(torch.nn.Module):
    """
    A Wrapper for DeepLIFT, since it only takes nn.Module
    """

    def __init__(
        self, model, attribution_analysis_instance: AttributionAnalysis
    ) -> None:
        super().__init__()
        self.model = model
        self.attribution_analysis_instance = attribution_analysis_instance

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_phenotype_category: str,
        target_phen: Optional[str],
    ):
        return self.attribution_analysis_instance.forward_pass(
            input_ids,
            token_type_ids,
            attention_mask,
            masked_phenotype_category,
            target_phen,
        )


def geometric_importance(attr1: torch.tensor, attr2: torch.tensor):
    return torch.where(
        (attr1 == 0) & (attr2 == 0),
        torch.tensor(0.0, device=attr1.device),
        torch.where(
            (attr1 == 0) | (attr2 == 0),
            torch.max(attr1.abs(), attr2.abs()),
            torch.sqrt(attr1.abs() * attr2.abs()),
        ),
    )


def arithmetic_importance(attr1: torch.tensor, attr2: torch.tensor):
    print(attr1.shape)
    return (attr1 + attr2) / 2


def identity_importance(reverse: bool = False):
    return lambda attr1, attr2: attr1 if not reverse else lambda attr1, attr2: attr2


def absolute_importance(reverse: bool = False):
    return lambda attr1, attr2: (
        torch.abs(attr1) if not reverse else lambda attr1, attr2: torch.abs(attr2)
    )


def random_importance(attr1: torch.tensor, attr2: torch.tensor):
    return torch.randperm(attr1.size(0), device=attr1.device)
