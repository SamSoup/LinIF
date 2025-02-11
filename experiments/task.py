from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from kronfluence.task import Task

BATCH_TYPE = Dict[str, torch.Tensor]


class LanguageModelingTask(Task):
    def __init__(self, module_fct: Callable):
        super().__init__()
        self._module_fct = module_fct

    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous()
        if not sample:
            summed_loss = F.cross_entropy(
                logits, labels.view(-1), reduction="sum", ignore_index=-100
            )
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
                masks = labels.view(-1) == -100
                sampled_labels[masks] = -100
            summed_loss = F.cross_entropy(
                logits, sampled_labels, ignore_index=-100, reduction="sum"
            )
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        return F.cross_entropy(
            logits, shift_labels, ignore_index=-100, reduction="sum"
        )

    def get_influence_tracked_modules(self) -> List[str]:
        """
        In my opinion, this function is poorly designed and ideally should not
        be contained in `LanguageModelingTask`. To accomodate the existing
        bad design, I am opting for an explicit defintion of modules in a
        function outside class, but simply store an pointer to said function
        during initialization, and invoke it here
        """

        return self._module_fct()

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]


class LanguageModelingWithMarginMeasurementTask(LanguageModelingTask):
    # For classification tasks
    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # Margin -- larger if model's score for correct token exceeds all
        # incorrect tokens combined by a wide margin
        # "How far above the incorrect class logit above all incorrect logits"
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        labels = batch["labels"][..., 1:].contiguous().view(-1)
        masks = labels != -100
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))

        bindex = torch.arange(logits.shape[0]).to(
            device=logits.device, non_blocking=False
        )
        # grabs the logit (unnormalized score) assigned to the true token at each position.
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        # set correct label's logit to -inf
        cloned_logits[bindex, labels] = torch.tensor(
            -torch.inf, device=logits.device, dtype=logits.dtype
        )

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins[masks].sum()


class LanguageModelingWithContrastTokenTask(LanguageModelingTask):
    """
    A language modeling task that uses a 'contrast token id' to measure
    how strongly the model prefers the correct token over the contrast token.
    """

    def __init__(self, module_fct: Callable, contrast_token_id: int):
        super().__init__(module_fct)
        self.contrast_token_id = contrast_token_id

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        """
        Returns the negative margin between logits_correct and logits_contrast:
            margin = logits_correct - logits_contrast
        Summed over all valid positions (where labels != -100).
        """
        # 1) Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits.float()  # (batch_size, seq_len, vocab_size)

        # 2) Shift labels/logits to align next-token prediction
        #    shape after shifting: (batch_size, seq_len-1)
        labels = batch["labels"][..., 1:].contiguous().view(-1)
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))

        # 3) Identify valid positions (ignore -100 as "padding")
        mask = labels != -100

        # 4) Gather the logit for the 'correct' label at each position
        #    shape: (total_positions,)
        bindex = torch.arange(logits.shape[0], device=logits.device)
        logits_correct = logits[bindex, labels]

        # 5) Gather the logit for the contrast token
        logits_contrast = logits[bindex, self.contrast_token_id]

        # 6) Compute the margin = correct - contrast
        #    This is higher if the model scores the correct token above the contrast token.
        margins = logits_correct - logits_contrast

        # 7) We only keep the margin for valid positions
        valid_margins = margins[mask]

        # 8) Return the negative of the sum (or mean) of these margins
        #    Minimizing this negative sum is equivalent to maximizing the margin.
        return -valid_margins.sum()
