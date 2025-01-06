import torch
import numpy as np
from typing import Callable, TypedDict, Tuple, Optional
from captum.attr import LayerIntegratedGradients, LayerDeepLift


# interfaces
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

    """

    def __init__(
        self,
        model,
        embedding: torch.nn.Module,
        method: str,
        target: ModelInput,
        baseline: ModelInput,
        tokenizer,
        device: torch.device,
    ):
        """
        Class Initialization

        Args:
            - model (GeneBertForPhenotypicMLM): a model for  the attribution analysis
            - embedding (torch.nn.Module): a target embedding for attribution method (see more at https://captum.ai/tutorials/Bert_SQUAD_Interpret)
            - method (str): an attribution method (either Intergrated Gradients (ig) or DeepLift (dl))
            - target (ModelInput): a parsed target instance
            - baseline (ModelInput): a parsed baseline instance
            - tokenizer (GeneTokenizer): a tokenizer associated with the model
            - device (torch.device): PyTorch device
        """
        self.model = model
        self.embedding = embedding

        assert method in [
            "dl",
            "ig",
        ], f"expect method to be either dl or ig, but got {method}"

        self.method = method
        self._eval_method = self.initialize_attribution_method()

        self._target, self._baseline = target, baseline
        self._blended_target, self._blended_baseline = None, None
        self._blended_token_type_ref = None

        self._attribution_values, self._delta = None, None

        self.tokenizer = tokenizer
        self.device = device

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
        return self._target

    @property
    def baseline(self) -> ModelInput:
        """
        Return the original instance of the baseline
        """
        return self._baseline

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

    def gene_blending(
        self,
        ref_token_id: int = 3,
        sep_token_id: int = 1,
    ) -> Tuple[ModelInput, ModelInput, torch.Tensor]:
        """
        Return the blended target, blended baseline, and blended token types
        for reference.

        Args:
            - ref_token_id (int): the token type id of ref_token [PAD]
            - sep_token_id (int): the token type id of sep_token [EOS]

        Description:
            The algorithm aligns and pads, if necessary, the target and baseline inputs such that
            the blended target and baseline tokens have the same length and are all aligned correctly.

            The order of token_type_ids of the blended_input and blended_baseline will be as follows:
            blended_target:   [CLS] [phenotypes] [input intersected genes]    [input-only genes]                 [paddings for baseline-only genes] [EOS]
            blended_baseline: [CLS] [phenotypes] [baseline intersected genes] [paddings for input-only genes]    [baseline-only genes]              [EOS]
        """

        # Step 1: Get a intersection of genes in target and baseline
        common_gene_target, common_gene_baseline = self._input_baseline_intersection()

        # Step 2: Get all genes that are unique to target and baselie respectively
        unique_gene_target, unique_gene_baseline = self._out_of_intersection_genes(
            self.target, common_gene_target
        ), self._out_of_intersection_genes(self.baseline, common_gene_baseline)

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
            "attention_mask": torch.tensor([[False]], device=self.device),
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
        phen: Optional[int] = None,
        return_prediction: bool = False,
    ) -> float | Tuple[float, int]:
        """
        Return the model's prediction (if specified) and its associated probability.

        Args:
            - input_ids (torch.Tensor): a tensor of an input's input_ids
            - token_type_ids (torch.Tensor): a tensor of an input's token_type_ids
            - attention_mask (torch.Tensor): a tensor of an input's attention_mask
            - masked_phenotype_category (str): a phenotypic category that the model predicts (this phenotype will be masked)
            - phen (int): the model will return the phenotype with highest probability and its associated probability, if not given.
                            otherwise, it will yield the probability of the model predicting a particular phenotype.
            - return_prediction (bool): a boolean indicating whether to return the predicted label.

        Description:
            If gene_blending has not been called yet, this function will first blend the token_type_ids.
            Given an input to the model, masked_phenotype_category is masked and the model is tasked
            with predicting it. If phen is specified, the model will return the probability predicting phen.
            Otherwise, it will give the most probable prediction.
        """

        # Skip if we have already blended for efficiency
        if self.blended_target is None or self.blended_baseline is None:
            print("`gene_blending` has not been called previously.")
            self.gene_blending()

        assert (
            self.blended_baseline is not None and self.blended_target is not None
        ), "Expect the to blend the input and baseline before passing to the model."

        category_idx = 1 + self.tokenizer.config.included_phenotypes.index(
            masked_phenotype_category
        )

        masked_input_ids = input_ids.clone()
        # Perform masking on the cloned tensor to avoid in-place modification

        masked_input_ids[:, category_idx] = 2

        output = self.model(
            masked_input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ).logits

        normalized = torch.softmax(output[:, category_idx, :], dim=-1)

        if phen is None:
            value, pred = (
                torch.max(normalized, dim=-1)[0].item(),
                torch.max(normalized, dim=-1)[1].item(),
            )
        else:  # Get the probability of the target
            value, pred = normalized[phen].item(), phen
        if return_prediction:
            return value, pred
        return value

    def compare_predictions(
        self, masked_phenotype_category: int
    ) -> dict[str, Tuple[float, int]]:
        """
        Return a dictionary mapping each kind of input to the model's prediction of the mask phenotypic category.

        Args:
            - masked_phenotype_category (int): a phenotypic category that the model predicts (this phenotype will be masked)
        """
        all_samples = {
            "original_target": self.target,
            "original_baseline": self.baseline,
            "blended_target": self.blended_target,
            "blended_baseline": self.blended_baseline,
        }

        predictions = {}
        for type, input in all_samples.items():
            prob, pred = self.forward_pass(
                input["input_ids"],
                input["token_type_ids"],
                input["attention_mask"],
                masked_phenotype_category,
                return_prediction=True,
            )
            predictions[type] = (prob, pred)
        return predictions

    def eval_attribution_method(
        self,
        internal_batch_size: int,
        masked_phenotype_category: str,
        n_steps: Optional[int] = 0,
        return_convergence: bool = True,
    ) -> Tuple[torch.tensor,]:
        """
        Return attribution values and convergence error calculated by an attribution analysis method
        (Integrated Gradient/DeepLift).

        Args:
            - internal_batch_size (int): a bsz for the attribuion method
            - masked_phenotype_category (str): a phenotypic category that the model predicts (this phenotype will be masked)
            - n_steps (int): a number of steps for Integrated Gradients method (it is not required for DeepLift)
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
                additional_forward_args=(masked_phenotype_category),
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
                additional_forward_args=(masked_phenotype_category),
                return_convergence_delta=True,
            )
        else:
            assert (
                False
            ), "Expect the method to be either LayerIntegratedGradients or LayerDeepLift"

        attribution_by_token = attributions.sum(dim=-1).squeeze(0)
        attribution_by_token = attribution_by_token / torch.norm(attribution_by_token)
        self._attribution_values, self._delta = attribution_by_token, delta.item()
        return attribution_by_token, delta

    def _input_baseline_intersection(self) -> Tuple[ModelInput, ModelInput]:
        """
        A helper function to construct a model input consisting of the intersection between the target and baseline token_type_ids
        """
        gene_token_type_offset = self.tokenizer.gene_token_type_offset

        # Step 1: We only find the intersection of genotypes, not phenotypes or special tokens.
        target_genotype_tokens, baseline_genotype_tokens = (
            self.target["token_type_ids"][gene_token_type_offset:-1],
            self.baseline["token_type_ids"][gene_token_type_offset:-1],
        )

        _, target_common_genes_indices, baseline_common_genes_indices = np.intersect1d(
            target_genotype_tokens, baseline_genotype_tokens, return_indices=True
        )

        # Step 2: We reconstruct the intersections between the target and baseline
        # according to the input format of the model.
        intersected_target = self._reconstruct_input(
            self.target, target_common_genes_indices, gene_token_type_offset
        )
        intersected_baseline = self._reconstruct_input(
            self.baseline, baseline_common_genes_indices, gene_token_type_offset
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

        Args:
            - input (ModelInput): a model input
            - indices (torch.Tensor): the indices of elements we want to keep
            - offset (int): an index offset
        """
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
                for k, v in input.items()
            }
        return {k: v[indices].unsqueeze(0) for k, v in input.items()}


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
    ):
        return self.attribution_analysis_instance.forward_pass(
            input_ids, token_type_ids, attention_mask, masked_phenotype_category
        )
