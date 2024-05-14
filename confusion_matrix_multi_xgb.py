from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd
import math
import evaluate
import pickle


global TOKENIZER
global DEVICE
global MODEL
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/ODABert', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = AutoModelForSequenceClassification.from_pretrained("alex-miller/cdp-multi-classifier-weighted")
MODEL = MODEL.to(DEVICE)

global CF_XGB
global PAF_XGB
global AA_XGB
global DUMMY_COLS
with open('models/xgb_multi.pkl', 'rb') as f:
    CF_XGB, PAF_XGB, AA_XGB, DUMMY_COLS = pickle.load(f)


def most_confident(confidence_1, confidence_2):
    absolutes = [
        abs(confidence_1 - round(confidence_1)),
        abs(confidence_2 - round(confidence_2)),
    ]
    return [confidence_1, confidence_2][np.argmin(absolutes)]


clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_metrics(predictions, labels):
    return clf_metrics.compute(predictions=predictions, references=labels)


def sigmoid(x):
   return 1/(1 + np.exp(-x))


def chunk_by_tokens(input_text, model_max_size=512):
    chunks = list()
    tokens = TOKENIZER.encode(input_text)
    token_length = len(tokens)
    if token_length <= model_max_size:
        return [input_text]
    desired_number_of_chunks = math.ceil(token_length / model_max_size)
    calculated_chunk_size = math.ceil(token_length / desired_number_of_chunks)
    for i in range(0, token_length, calculated_chunk_size):
        chunks.append(TOKENIZER.decode(tokens[i:i + calculated_chunk_size]))
    return chunks


def inference(model, inputs):
    predictions = model(**inputs)

    logits = predictions.logits.cpu().detach().numpy()[0]
    predicted_confidences = sigmoid(logits)
    predicted_classes = (predicted_confidences > 0.5)

    return predicted_classes, predicted_confidences

def map_columns(example):
    example['Crisis finance actual'] = 'Crisis financing' in example['labels']
    example['PAF actual'] = 'PAF' in example['labels']
    example['AA actual'] = 'AA' in example['labels']
    predictions = {
        'Crisis finance': [False, 0],
        'PAF': [False, 0],
        'AA': [False, 0]
    }

    text_chunks = chunk_by_tokens(example['text'])
    for text_chunk in text_chunks:
        inputs = TOKENIZER(text_chunk, return_tensors="pt", truncation=True).to(DEVICE)
        model_pred, model_conf = inference(MODEL, inputs)
        predictions['Crisis finance'][0] = predictions['Crisis finance'][0] or model_pred[0]
        predictions['Crisis finance'][1] = max(predictions['Crisis finance'][1], model_conf[0])
        predictions['PAF'][0] = predictions['PAF'][0] or model_pred[1]
        predictions['PAF'][1] = max(predictions['PAF'][1], model_conf[1])
        predictions['AA'][0] = predictions['AA'][0] or model_pred[2]
        predictions['AA'][1] = max(predictions['AA'][1], model_conf[2])

    example['Crisis finance predicted'] = predictions['Crisis finance'][0]
    example['Crisis finance confidence'] = predictions['Crisis finance'][1]
    example['PAF predicted'] = predictions['PAF'][0]
    example['PAF confidence'] = predictions['PAF'][1]
    example['AA predicted'] = predictions['AA'][0]
    example['AA confidence'] = predictions['AA'][1]

    dummy_dim = len(DUMMY_COLS)
    row = np.array([example[col] for col in ['Crisis finance confidence', 'PAF confidence', 'AA confidence'] + DUMMY_COLS])

    X = np.reshape(row.astype('float'), (1, dummy_dim + 3))
    example['Crisis finance predicted XGB'] = CF_XGB.predict(X)[0] == 1
    example['Crisis finance confidence XGB'] = CF_XGB.predict_proba(X)[0][1]
    example['PAF predicted XGB'] =  PAF_XGB.predict(X)[0] == 1
    example['PAF confidence XGB'] = PAF_XGB.predict_proba(X)[0][1]
    example['AA predicted XGB'] = AA_XGB.predict(X)[0] == 1
    example['AA confidence XGB'] = AA_XGB.predict_proba(X)[0][1]

    # Use more confident to create joint indicator
    example['Crisis finance confidence joint'] = most_confident(example['Crisis finance confidence'], example['Crisis finance confidence XGB'])
    example['PAF confidence joint'] = most_confident(example['PAF confidence'], example['PAF confidence XGB'])
    example['AA confidence joint'] = most_confident(example['AA confidence'], example['AA confidence XGB'])

    # Set PAF confidence equal to AA confidence if AA predicted and PAF not
    example['PAF confidence joint'] = example['PAF confidence joint'] if example['AA confidence joint'] < 0.5 else example['AA confidence joint']
    # Set CF confidence equal to PAF confidence if PAF predicted and CF not
    example['Crisis finance confidence joint'] = example['Crisis finance confidence joint'] if example['PAF confidence joint'] < 0.5 else example['PAF confidence joint']

    example['Crisis finance predicted joint'] = example['Crisis finance confidence joint'] >= 0.5
    example['PAF predicted joint'] = example['PAF confidence joint'] >= 0.5
    example['AA predicted joint'] = example['AA confidence joint'] >= 0.5

    return example


def construct_dataset():
    dataset = load_dataset("alex-miller/cdp-paf-meta-limited", split="train")
    # Retain only "Unrelated" and "Crisis financing" as other positive cases we will draw from synthetic dataset
    cf_dataset = dataset.filter(lambda example: example['labels'] == 'Crisis financing')
    unrelated_dataset = dataset.filter(lambda example: example['labels'] == 'Unrelated')
    # Cut unrelated by half to save on processing
    half_unrelated_num_rows = math.floor(unrelated_dataset.num_rows / 2)
    unrelated_dataset = unrelated_dataset.shuffle(seed=42).select(range(half_unrelated_num_rows))
    dataset = concatenate_datasets([cf_dataset, unrelated_dataset])
    synth = load_dataset("alex-miller/cdp-paf-meta-limited-synthetic")
    # Add removable column to stratify
    dataset = dataset.add_column("class_labels", dataset['labels'])
    dataset = dataset.class_encode_column('class_labels').train_test_split(
        test_size=0.1,
        stratify_by_column="class_labels",
        shuffle=True,
        seed=42
    )
    dataset = concatenate_datasets([dataset['test'], synth['test']])
    dataset = pd.DataFrame(dataset)
    crs = pd.read_csv('large_data/meta_model_for_xgboost.csv')
    crs = crs.drop_duplicates()
    dataset['text'] = dataset['text'].str.strip()
    dataset = pd.merge(dataset, crs, on="text", how="left")
    return dataset


def main():
    dataset = construct_dataset()
    dataset = pd.get_dummies(dataset, columns=["DonorName", "SectorName", "PurposeName", "FlowName", "ChannelName"], dtype=float)

    for dummy_col in DUMMY_COLS:
        dataset[dummy_col] = dataset.get(dummy_col, 0)

    dataset = dataset[['text', 'labels'] + DUMMY_COLS]

    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.map(map_columns)
    # dataset.save_to_disk('large_data/matrix_multi_xgb')
    # dataset = dataset.load_from_disk('large_data/matrix_multi_xgb')
    y_true = np.array([
        dataset['Crisis finance actual'],
        dataset['PAF actual'],
        dataset['AA actual']
    ]).transpose()
    y_pred = np.array([
        dataset['Crisis finance predicted XGB'],
        dataset['PAF predicted XGB'],
        dataset['AA predicted XGB']
    ]).transpose()
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    confusion_indices = ["Actual negative", "Actual positive"]
    confusion_columns = ["Predicted negative", "Predicted positive"]
    crisis_confusion = pd.DataFrame(mcm[0], index=confusion_indices, columns=confusion_columns)
    paf_confusion = pd.DataFrame(mcm[1], index=confusion_indices, columns=confusion_columns)
    aa_confusion = pd.DataFrame(mcm[2], index=confusion_indices, columns=confusion_columns)
    print("Crisis finance:")
    print(crisis_confusion)
    print(compute_metrics(dataset['Crisis finance actual'], dataset['Crisis finance predicted XGB']))
    print("PAF:")
    print(paf_confusion)
    print(compute_metrics(dataset['PAF actual'], dataset['PAF predicted XGB']))
    print("AA:")
    print(aa_confusion)
    print(compute_metrics(dataset['AA actual'], dataset['AA predicted XGB']))


if __name__ == '__main__':
    main()

