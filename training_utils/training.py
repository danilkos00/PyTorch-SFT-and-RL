import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from vllm import LLM, SamplingParams
from tqdm import tqdm
from .post_training_utils import (
    load_policy_into_vllm_instance,
    compute_group_normalized_rewards,
    reward_fn,
    grpo_microbatch_train_step,
    sft_microbatch_train_step
)
from .nn_utils import tokenize_prompt_and_output, get_response_log_probs
import os
import json


tokenizer = None

def grpo_train(dataset, config_path: str, model: str, checkpoint: str | None = None):
    with open(config_path) as f:
        config = json.load(f)

    tokenizer_name = config['training']['tokenizer_name']
    train_batch_size = config['training']['train_batch_size']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    rollout_batch_size = config['training']['rollout_batch_size']
    group_size = config['training']['group_size']
    learning_rate = config['optimizer']['learning_rate']
    betas = config['optimizer']['betas']
    weight_decay = config['optimizer']['weight_decay']
    gpu_memory_utilization = config['training']['gpu_memory_utilization']
    sampling_temperature = config['training']['sampling_temperature']
    sampling_max_tokens = config['training']['sampling_max_tokens']
    sampling_min_tokens = config['training']['sampling_min_tokens']
    n_grpo_steps = config['training']['n_grpo_steps']
    advantage_eps = config['training']['advantage_eps']
    use_std_normalization = bool(config['training']['use_std_normalization'])
    loss_type = config['training']['loss_type']
    epochs_per_rollout_batch = config['training']['epochs_per_rollout_batch']
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    policy = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas
    )

    accelerator = Accelerator()
    policy, optimizer = accelerator.prepare(policy, optimizer)

    llm = LLM(
        model=model,
        tokenizer='Qwen/Qwen2.5-Math-1.5B',
        tensor_parallel_size=1,
        dtype='float16',
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=4,
        enforce_eager=True
    )

    sampling_params = SamplingParams(
        n=group_size,
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=['</answer>'],
        include_stop_str_in_output=True
    )

    start = 0
    if checkpoint is not None:
        start = load_checkpoint(checkpoint, policy, optimizer)

    pbar = tqdm(range(start, n_grpo_steps))
    policy.train()
    min_loss = float('inf')
    for grpo_step in pbar:
        load_policy_into_vllm_instance(policy, llm)

        prompts, labels = dataset['train'][grpo_step*n_prompts_per_rollout_batch:(grpo_step+1)*n_prompts_per_rollout_batch].values()

        rollout_responses = []
        for prompt in prompts:
            with torch.no_grad():
                output = llm.generate([prompt], sampling_params, use_tqdm=False)[0].outputs
            rollout_responses.extend([out.text for out in output])

        repeated_prompts = [prompt for prompt in prompts for _ in range(group_size)]
        repeated_ground_truths = [label for label in labels for _ in range(group_size)]

        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn,
            rollout_responses,
            repeated_ground_truths,
            group_size,
            advantage_eps,
            use_std_normalization
        )
        pbar.set_postfix(metadata)

        dataloader = DataLoader(
            list(zip(repeated_prompts, rollout_responses, advantages)),
            batch_size=micro_train_batch_size,
            shuffle=False,
            collate_fn=_grpo_collate_fn,
            num_workers=4
        )

        dataloader = accelerator.prepare(dataloader)

        for train_step in range(epochs_per_rollout_batch):
            running_loss = 0.0
            for i, (micro_batch, batched_advantages) in enumerate(dataloader):
                input_ids = micro_batch['input_ids']
                batched_responses = micro_batch['labels']
                response_mask = micro_batch['response_mask']

                log_probs = get_response_log_probs(policy, input_ids, batched_responses)['log_probs']

                loss, _ = grpo_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    advantages=batched_advantages
                )

                accelerator.backward(loss)

                running_loss += loss.item()

            if running_loss < min_loss:
                min_loss = running_loss
                save_checkpoint(policy, optimizer, grpo_step, 'checkpoint.tar')
                
            accelerator.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            print(f'\nLoss (accumulated): {running_loss:.4f}')

    save_checkpoint(policy, optimizer, grpo_step, 'final_checkpoint.tar')


def sft_train(dataset, config_path: str, model: str):
    with open(config_path) as f:
        config = json.load(f)

    tokenizer_name = config['training']['tokenizer_name']
    sft_steps = config['training']['sft_steps']
    batch_size = config['training']['batch_size']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    learning_rate = config['optimizer']['learning_rate']
    weight_decay = config['optimizer']['weight_decay']
    betas = config['optimizer']['betas']

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    micro_batch_size = batch_size // gradient_accumulation_steps

    dataloader = DataLoader(
        dataset['train'],
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=_sft_collate_fn,
        num_workers=4
    )

    policy = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas
    )

    accelerator = Accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()

    current_sft_step = 0
    running_loss = 0.0
    pbar = enumerate(tqdm(dataloader), start=1)
    for i, batch in pbar:
        input_ids = batch['input_ids']
        labels = batch['labels']
        response_mask = batch['response_mask']

        log_probs, entropy = get_response_log_probs(model, input_ids, labels, True).values()

        loss, metadata = sft_microbatch_train_step(log_probs, response_mask, gradient_accumulation_steps)

        accelerator.backward(loss)

        running_loss += loss.item()

        if i % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            running_entropy /= gradient_accumulation_steps
            running_loss /= gradient_accumulation_steps
            current_sft_step += 1
            pbar.set_postfix_str(f'Step={current_sft_step}; loss={running_loss:.4f}')
            running_loss = 0.0
            if current_sft_step >= sft_steps:
                break


def _grpo_collate_fn(batch):
    prompt_strs = []
    response_strs = []
    advantages = []

    for item in batch:
        prompt_strs.append(item[0])
        response_strs.append(item[1])
        advantages.append(item[2].unsqueeze(0))

    return tokenize_prompt_and_output(prompt_strs, response_strs, tokenizer), torch.cat(advantages).unsqueeze(-1)


def _sft_collate_fn(batch):
    prompt_strs = [item['question'] for item in batch]
    response_strs = [item['answer'] for item in batch]

    return tokenize_prompt_and_output(prompt_strs, response_strs, tokenizer)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str
):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(checkpoint, out, pickle_protocol=5)


def load_checkpoint(
    src: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
):
    checkpoint = torch.load(src, weights_only=False)

    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint.get('iteration', 0)