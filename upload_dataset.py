from huggingface_hub import login
from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv
import pandas as pd
import os
from tqdm import tqdm
from Levenshtein import distance


def drop_similar_rows(df, text_column, threshold_percentage):
    to_drop = set()
    texts = df[text_column].tolist()
    num_texts = len(texts)

    for i in tqdm(range(num_texts)):
        if i not in to_drop:
            for j in range(i + 1, num_texts):
                if j not in to_drop:
                    len_i = len(texts[i])
                    len_j = len(texts[j])
                    threshold = max(max(len_i, len_j) * threshold_percentage, 1)
                    if distance(texts[i], texts[j]) <= threshold:
                        to_drop.add(j)
    return df.drop(to_drop)


def main():
    data = pd.read_csv("./large_data/meta_model_data.csv")

    dataset = Dataset.from_pandas(data, preserve_index=False)

    # Initial rough balance to speed up de-duplicate
    positive = dataset.filter(lambda example: example["label"] == "PAF/AA")
    negative = dataset.filter(lambda example: example["label"] == "Unrelated")
    negative = negative.shuffle(seed=42)
    negative = negative.select(range(positive.num_rows * 2))
    dataset = concatenate_datasets([positive, negative])
    data = pd.DataFrame(dataset)

    # De-duplicate on exact lowercase match, and lowercase string distance
    print("Pre-de-duplicate shape: ", data.shape)
    data['text_lowercase'] = data['text'].str.lower()
    data = data.drop_duplicates(subset=['text_lowercase'], keep='first')
    data = data.reset_index(drop=True)
    data = drop_similar_rows(data, 'text_lowercase', threshold_percentage=0.05)
    print("Post-de-duplicate shape: ", data.shape)
    data = data.drop(columns=['text_lowercase'])
    dataset = Dataset.from_pandas(data, preserve_index=False)

    # Final balance
    positive = dataset.filter(lambda example: example["label"] == "PAF/AA")
    print("Positive balance length: ", len(positive))
    negative = dataset.filter(lambda example: example["label"] == "Unrelated")
    negative = negative.shuffle(seed=42)
    negative = negative.select(range(positive.num_rows))
    dataset = concatenate_datasets([positive, negative])

    # Push
    dataset.push_to_hub("alex-miller/cdp-paf-meta")


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()