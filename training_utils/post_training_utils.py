import torch
from transformers import PreTrainedModel
from vllm import LLM
import re
import math


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses,
    normalized by the group size.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths,
            producing a dict with keys
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy.
            The length of this list is
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples.
            The length of this list is `rollout_batch_size`,
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,):
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,):
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
    """
    advantages = []
    raw_rewards = []
    
    for group_idx in range(0, len(rollout_responses), group_size):
        group_rewards = []

        for groupped_response_idx in range(group_idx, group_idx + group_size):
            rewards = reward_fn(
                rollout_responses[groupped_response_idx],
                repeated_ground_truths[groupped_response_idx]
            )
            group_rewards.append(rewards['reward'])

        group_rewards = torch.tensor(group_rewards)

        mean = group_rewards.mean()
        group_advantages = group_rewards - mean

        if normalize_by_std:
            std = group_rewards.std()
            group_advantages /= std + advantage_eps

        advantages.append(group_advantages)
        raw_rewards.append(group_rewards)

    advantages = torch.cat(advantages)
    raw_rewards = torch.cat(raw_rewards)

    metadata = {
        'mean_advantage': advantages.mean().item(),
        'max_advantage': advantages.max().item(),
        'min_advantage': advantages.min().item(),
        'std_advantage': advantages.std().item(),
        'positive_advantage_ratio': ((advantages > 0).sum() / advantages.numel()).item(),
        'mean_reward': raw_rewards.mean().item(),
        'max_reward': raw_rewards.max().item(),
        'min_reward': raw_rewards.min().item(),
        'std_reward': raw_rewards.std().item(),
        'positive_reward_ratio': ((raw_rewards > 0).sum() / raw_rewards.numel()).item()
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            the policy gradient per-token loss.
    """
    return -raw_rewards_or_advantages.mul(policy_log_probs)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1):
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length):
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss
                (used to compute clip fraction).
    """
    policy_div = torch.exp(policy_log_probs.sub(old_log_probs))

    metadata = {
        'mean_advantage': advantages.mean().item(),
        'max_advantage': advantages.max().item(),
        'min_advantage': advantages.min().item(),
        'std_advantage': advantages.std().item(),
        'positive_advantage_ratio': ((advantages > 0).sum() / advantages.numel()).item()
    }

    clipped_policy = torch.where((advantages >= 0).expand(policy_div.size()), 1 + cliprange, 1 - cliprange)

    return -torch.minimum(policy_div.mul(advantages), clipped_policy.mul(advantages)), metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == 'no_baseline':
        assert raw_rewards is not None
        metadata = {
            'mean_reward': raw_rewards.mean().item(),
            'max_reward': raw_rewards.max().item(),
            'min_reward': raw_rewards.min().item(),
            'std_reward': raw_rewards.std().item(),
            'positive_reward_ratio': ((raw_rewards > 0).sum() / raw_rewards.numel()).item()
        }

        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), metadata
    elif loss_type == 'reinforce_with_baseline':
        assert advantages is not None
        metadata = {
            'mean_advantage': advantages.mean().item(),
            'max_advantage': advantages.max().item(),
            'min_advantage': advantages.min().item(),
            'std_advantage': advantages.std().item(),
            'positive_advantage_ratio': ((advantages > 0).sum() / advantages.numel()).item()
        }

        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), metadata
    elif loss_type == 'grpo_clip':
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None

        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    raise ValueError(f"Invalid loss_type: '{loss_type}'. Must be one of ['no_baseline', 'reinforce_with_baseline', 'grpo_clip']")


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    return torch.where(mask.bool(), tensor, 0.0).sum(dim).div(mask.sum(dim))


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            the mask for the response.
        gradient_accumulation_steps: int:
            the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio.
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the policy gradient loss and its metadata.
    """
    loss_tensor, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )

    loss = masked_mean(loss_tensor, response_mask) / gradient_accumulation_steps

    return loss, metadata


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    return torch.where(mask.bool(), tensor, 0.0).sum(dim=dim).div(normalize_constant)


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    batch_size = policy_log_probs.size(0)

    loss = - masked_normalize(policy_log_probs, response_mask, normalize_constant=normalize_constant)

    loss = loss.div(gradient_accumulation_steps * batch_size)

    metadata = {
        'loss_item': loss.item()
    }

    return loss, metadata


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    return torch.where(mask.bool(), tensor, 0.0).sum(dim=dim).div(normalize_constant)


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    batch_size = policy_log_probs.size(0)

    loss = - masked_normalize(policy_log_probs, response_mask, normalize_constant=normalize_constant)

    loss = loss.div(gradient_accumulation_steps * batch_size)

    metadata = {
        'loss_item': loss.item()
    }

    return loss, metadata


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def reward_fn(generated_text: str, gt_text: str):
    generated_answer = re.search(r'<answer>#### (.+)</answer>', generated_text)
    if generated_answer is None:
        return {
            'reward': 0.0
        }

    generated_answer = generated_answer.group(1)
    gt_answer = re.search(r'<answer>#### (.+)</answer>', gt_text).group(1)

    if generated_answer == gt_answer:
        return {
            'reward': 1.0
        }

    try:
        if '/' in generated_answer:
            nums = list(map(float, generated_answer.split('/')))
            f_generated_answ = nums[0] / nums[1]
        else:
            f_generated_answ = float(generated_answer.replace(',', ''))

        if '/' in gt_answer:
            nums = list(map(float, gt_answer.split('/')))
            f_gt_answer = nums[0] / nums[1]
        else:
            f_gt_answer = float(gt_answer.replace(',', ''))
    except:
        return {
            'reward': 0.0
        }

    reward = math.exp(-2*abs(f_generated_answ - f_gt_answer) / (f_gt_answer + 1e-6))

    return {
        'reward': reward
    }
