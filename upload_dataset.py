from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from dotenv import load_dotenv
import pandas as pd
import os


def main():
    dataset = load_dataset("csv", data_files="./large_data/meta_model_data.csv", split='train')

    # Balance
    # positive = dataset.filter(lambda example: example["labels"] != "Unrelated")
    # print("Positive balance length: ", len(positive))
    # negative = dataset.filter(lambda example: example["labels"] == "Unrelated")
    # negative = negative.shuffle(seed=42)
    # negative = negative.select(range(positive.num_rows))
    # dataset = concatenate_datasets([positive, negative])

    # Push
    dataset.push_to_hub("alex-miller/cdp-paf-meta")


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()