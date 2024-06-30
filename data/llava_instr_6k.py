import subprocess
from datasets import load_dataset, Image, concatenate_datasets
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Download and load datasets")
    parser.add_argument('--working_dir', type=str, default='.', help='Working directory to save and load files')
    parser.add_argument('--total_samples', type=int, default=6000, help='Total number of samples')
    parser.add_argument('--image_dir', type=str, default='/kaggle/input/coco-2017-dataset/coco2017/train2017/', help='Directory containing the images')
    parser.add_argument('--save_name', type=str, default='llava_instr_6k_detail', help='Name to save the processed dataset')
    parser.add_argument('--question_type', type=str, choices=['Detail', 'Reasoning', 'Mix'],
                        default='Detail', help='Type of question to sample')

    args = parser.parse_args()
    working_dir = args.working_dir
    total_samples = args.total_samples
    image_dir = args.image_dir
    save_name = args.save_name
    question_type = args.question_type

    # URLs to download
    urls = [
        "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/complex_reasoning_77k.json",
        "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/conversation_58k.json",
        "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json"
    ]

    # Run wget commands
    for url in urls:
        subprocess.run(["wget", "-c", url, "-P", working_dir])

    # Load datasets from the working directory
    complex_reasoning_path = os.path.join(working_dir, "complex_reasoning_77k.json")
    conversation_path = os.path.join(working_dir, "conversation_58k.json")
    detail_path = os.path.join(working_dir, "detail_23k.json")

    complex_reasoning = load_dataset('json', data_files=complex_reasoning_path)
    conversation = load_dataset('json', data_files=conversation_path)
    detail = load_dataset('json', data_files=detail_path)

    #sample and split datasets, then concat
    if question_type == 'Detail':
        sample_sizes = {
            'complex_reasoning': int(total_samples * 0),
            'conversation': int(total_samples * 0),
            'detail': int(total_samples * 1)
        }
    elif question_type == 'Reasoning':
        sample_sizes = {
            'complex_reasoning': int(total_samples * 1),
            'conversation': int(total_samples * 0),
            'detail': int(total_samples * 0)
        }
    else:
        sample_sizes = {
            'complex_reasoning': int(total_samples * 0.5),
            'conversation': int(total_samples * 0),
            'detail': int(total_samples * 0.5)
    }

    sampled_complex_reasoning = complex_reasoning['train'].shuffle(seed=42).select(range(sample_sizes['complex_reasoning']))
    sampled_conversation = conversation['train'].shuffle(seed=42).select(range(sample_sizes['conversation']))
    sampled_detail = detail['train'].shuffle(seed=42).select(range(sample_sizes['detail']))

    dataset = concatenate_datasets([sampled_complex_reasoning, sampled_conversation, sampled_detail])

    def full_image_path(example):
        image_path = image_dir + example['image']
        example['image'] = image_path
        return example

    def reformat_conversations(example):
        conversations = example['conversations']
        new_conversations = []
        for entry in conversations:
            if entry['from'] == 'human':
                role = 'user'
            elif entry['from'] == 'gpt':
                role = 'assistant'
            new_entry = {
                'role': role,
                'content': entry['value']
            }
            new_conversations.append(new_entry)
        example['conversations'] = new_conversations
        return example

    dataset = dataset.map(full_image_path)
    dataset = dataset.map(reformat_conversations, remove_columns=['id'])

    dataset = dataset.cast_column("image", Image(decode=True))

    dataset.save_to_disk(save_name)
    pass


if __name__ == "__main__":
    main()