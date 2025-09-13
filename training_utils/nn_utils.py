from transformers import PreTrainedTokenizerBase
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    stable_logits = logits - logits.max(dim=-1, keepdim=True)[0]

    logsumexp = torch.logsumexp(stable_logits, dim=-1, keepdim=True)

    probs = torch.exp(stable_logits) / torch.exp(logsumexp)
    log_probs = stable_logits - logsumexp

    return -(probs * log_probs).sum(-1)


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    prompt_and_output_ids = []
    response_mask = []

    for i in range(len(prompt_strs)):
        prompt_ids = tokenizer(prompt_strs[i], return_tensors='pt')["input_ids"][0]
        output_ids = tokenizer(output_strs[i], return_tensors='pt')["input_ids"][0]

        prompt_and_output_ids.append(torch.cat([prompt_ids, output_ids]))

        response_mask.append(torch.tensor([0]*len(prompt_ids) + [1]*len(output_ids), dtype=torch.long))

    padding_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    pad_prompt_and_output_ids = pad_sequence(
        prompt_and_output_ids,
        batch_first=True,
        padding_value=padding_value
    )

    pad_response_mask = pad_sequence(
        response_mask,
        batch_first=True,
        padding_value=0
    )

    return {
        'input_ids': pad_prompt_and_output_ids[:, :-1],
        'labels': pad_prompt_and_output_ids[:, 1:],
        'response_mask': pad_response_mask[:, 1:]
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits

    log_probs = F.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    if return_token_entropy:
        return {
            'log_probs': log_probs,
            'token_entropy': compute_entropy(logits)
        }

    return {
        'log_probs': log_probs
    }