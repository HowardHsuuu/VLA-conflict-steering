"""Evidence adapters for text instructions and frozen vision-language models."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor

from causal_vla.routing import EvidenceDistribution


def language_label_evidence(
    text: str,
    *,
    proposition: str,
    labels: tuple[str, ...],
    aliases: Mapping[str, Sequence[str]] | None = None,
    matched_confidence: float = 0.98,
) -> EvidenceDistribution:
    """Extract one categorical label from text, returning uniform uncertainty otherwise."""

    if not labels:
        raise ValueError("labels must not be empty")
    if not 0 < matched_confidence <= 1:
        raise ValueError("matched_confidence must lie in (0, 1]")
    normalized = text.casefold()
    matched: list[str] = []
    for label in labels:
        variants = (label, *(aliases or {}).get(label, ()))
        if any(
            re.search(rf"(?<!\w){re.escape(variant.casefold())}(?!\w)", normalized)
            for variant in variants
        ):
            matched.append(label)

    if len(matched) != 1:
        probabilities = tuple(1.0 / len(labels) for _ in labels)
    elif len(labels) == 1:
        probabilities = (1.0,)
    else:
        remainder = (1.0 - matched_confidence) / (len(labels) - 1)
        probabilities = tuple(
            matched_confidence if label == matched[0] else remainder for label in labels
        )
    return EvidenceDistribution("language", proposition, labels, probabilities)


def _candidate_log_score(
    logits: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    prompt_length: int,
    *,
    length_normalize: bool,
) -> Tensor:
    """Score answer tokens under causal next-token logits."""

    if logits.ndim != 3 or input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("Expected logits [B,L,V] and token tensors [B,L]")
    if input_ids.shape != attention_mask.shape or logits.shape[:2] != input_ids.shape:
        raise ValueError("Logits, input IDs, and attention mask must align")
    if input_ids.shape[0] != 1:
        raise ValueError("Candidate scoring currently expects batch size one")
    if not 1 <= prompt_length < input_ids.shape[1]:
        raise ValueError("Candidate must add at least one token after the prompt")

    answer_ids = input_ids[:, prompt_length:]
    answer_mask = attention_mask[:, prompt_length:].to(dtype=torch.bool)
    predicting_logits = logits[:, prompt_length - 1 : -1].float()
    token_scores = (
        torch.log_softmax(predicting_logits, dim=-1)
        .gather(-1, answer_ids.unsqueeze(-1))
        .squeeze(-1)
    )
    score = (token_scores * answer_mask).sum()
    if length_normalize:
        score = score / answer_mask.sum().clamp_min(1)
    return score


def vlm_candidate_evidence(
    model: Any,
    processor: Any,
    image: Any,
    *,
    question: str,
    proposition: str,
    candidates: Mapping[str, str],
    device: str | torch.device,
    temperature: float = 1.0,
    length_normalize: bool = True,
) -> EvidenceDistribution:
    """Score candidate visual answers with a frozen generative VLM.

    Each score is the conditional log-likelihood of only the assistant answer tokens.
    The function verifies that the rendered prompt is an exact token prefix of every
    candidate conversation, preventing accidental scoring of template differences.
    """

    if not candidates:
        raise ValueError("candidates must not be empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    user_content = [
        {"type": "image"},
        {"type": "text", "text": question},
    ]
    prompt_messages = [{"role": "user", "content": user_content}]
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
    prompt_ids = cast(Tensor, prompt_inputs["input_ids"])
    prompt_length = int(prompt_ids.shape[1])

    scores: list[Tensor] = []
    labels = tuple(candidates)
    for answer in candidates.values():
        messages = [
            *prompt_messages,
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = processor(text=text, images=[image], return_tensors="pt")
        input_ids = cast(Tensor, inputs["input_ids"])
        if input_ids.shape[1] <= prompt_length or not torch.equal(
            input_ids[:, :prompt_length], prompt_ids
        ):
            raise ValueError("Candidate conversation does not preserve the prompt token prefix")
        device_inputs = {
            key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**device_inputs)
        scores.append(
            _candidate_log_score(
                cast(Tensor, outputs.logits),
                cast(Tensor, device_inputs["input_ids"]),
                cast(Tensor, device_inputs["attention_mask"]),
                prompt_length,
                length_normalize=length_normalize,
            ).cpu()
        )
    probabilities = torch.softmax(torch.stack(scores) / temperature, dim=0)
    return EvidenceDistribution(
        "vision",
        proposition,
        labels,
        tuple(float(value) for value in probabilities),
    )
