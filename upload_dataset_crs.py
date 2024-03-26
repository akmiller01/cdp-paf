from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv
import os

def main():
    dataset = load_dataset("csv", data_files="./large_data/crs_for_dataset.csv")
    dataset.push_to_hub("alex-miller/oecd-dac-crs")


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()
