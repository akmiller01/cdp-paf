from huggingface_hub import login
from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv
import pandas as pd
import os


def split_multi_label(examples):
    examples["label"] = [label.split(",") for label in examples["label"]]
    return examples


def main():
    data = pd.read_csv("./large_data/meta_model_data.csv")

    # De-duplicate
    data = data.drop_duplicates(subset=['text'])

    dataset = Dataset.from_pandas(data, preserve_index=False)

    # Balance
    positive = dataset.filter(lambda example: example["label"] in ["PAF", "AA,PAF", "AA"])
    negative = dataset.filter(lambda example: example["label"] == "Unrelated")
    negative = negative.shuffle(seed=42)
    negative = negative.select(range(positive.num_rows))
    dataset = concatenate_datasets([positive, negative])

    # Split
    dataset = dataset.map(split_multi_label, batched=True, num_proc=8)
    dataset.push_to_hub("alex-miller/cdp-paf-meta")


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()