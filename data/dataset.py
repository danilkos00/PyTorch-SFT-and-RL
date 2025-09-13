from datasets import load_dataset


def _add_tags(example):
    cot_part, answer_part = example['answer'].split('####')
    example['question'] = 'Question: ' + example['question'] + 'Answer: <think>'
    example['answer'] = cot_part.strip() + '</think>' + ' <answer>####' + answer_part + '</answer>'

    return example


def get_sft_dataset():
    dataset = load_dataset('openai/gsm8k', 'main')
    tagged_dataset = dataset.map(_add_tags)

    return tagged_dataset