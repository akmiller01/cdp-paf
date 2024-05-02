from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd
import math
import evaluate


global TOKENIZER
global DEVICE
global MODEL
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/ODABert', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = AutoModelForSequenceClassification.from_pretrained("alex-miller/cdp-multi-classifier-weighted")
MODEL = MODEL.to(DEVICE)


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
    return example

def main():
    dataset = load_dataset('alex-miller/cdp-paf-meta-limited', split='train')
    dataset = dataset.shuffle(seed=1337).select(range(1000))
    dataset = dataset.map(map_columns)
    dataset.to_csv('large_data/predicted_meta_model_data_multi.csv')
    y_true = np.array([
        dataset['Crisis finance actual'],
        dataset['PAF actual'],
        dataset['AA actual']
    ]).transpose()
    y_pred = np.array([
        dataset['Crisis finance predicted'],
        dataset['PAF predicted'],
        dataset['AA predicted']
    ]).transpose()
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    confusion_indices = ["Actual negative", "Actual positive"]
    confusion_columns = ["Predicted negative", "Predicted positive"]
    crisis_confusion = pd.DataFrame(mcm[0], index=confusion_indices, columns=confusion_columns)
    paf_confusion = pd.DataFrame(mcm[1], index=confusion_indices, columns=confusion_columns)
    aa_confusion = pd.DataFrame(mcm[2], index=confusion_indices, columns=confusion_columns)
    print("Crisis finance:")
    print(crisis_confusion)
    print(compute_metrics(dataset['Crisis finance actual'], dataset['Crisis finance predicted']))
    print("PAF:")
    print(paf_confusion)
    print(compute_metrics(dataset['PAF actual'], dataset['PAF predicted']))
    print("AA:")
    print(aa_confusion)
    print(compute_metrics(dataset['AA actual'], dataset['AA predicted']))


if __name__ == '__main__':
    main()

