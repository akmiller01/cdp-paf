from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd

global TOKENIZER
global DEVICE
global MODEL_1
global MODEL_2
global MODEL_3
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/ODABert', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_1 = AutoModelForSequenceClassification.from_pretrained("alex-miller/cdp-crisis-finance-classifier")
MODEL_1 = MODEL_1.to(DEVICE)
MODEL_2 = AutoModelForSequenceClassification.from_pretrained("alex-miller/cdp-paf-classifier")
MODEL_2 = MODEL_2.to(DEVICE)
MODEL_3 = AutoModelForSequenceClassification.from_pretrained("alex-miller/cdp-aa-classifier-synth")
MODEL_3 = MODEL_3.to(DEVICE)


def calculate_correct(example):
    correct = (example['Crisis finance actual'] == example['Crisis finance predicted']) and (example['PAF actual'] == example['PAF predicted']) and (example['AA actual'] == example['AA predicted'])
    return correct


def inference(model, inputs):
    with torch.no_grad():
        predictions = model(**inputs)

    logits = predictions.logits.cpu().numpy()[0]
    predicted_class_id = np.argmax(logits)
    predicted_class = model.config.id2label[predicted_class_id]

    return predicted_class

def map_columns(example):
    inputs = TOKENIZER(example['text'], return_tensors="pt", truncation=True).to(DEVICE)
    example['Crisis finance actual'] = 'Crisis financing' in example['labels']
    example['PAF actual'] = 'PAF' in example['labels']
    example['AA actual'] = 'AA' in example['labels']
    model_1_pred = inference(MODEL_1, inputs)
    example['Crisis finance predicted'] = model_1_pred == 'Crisis finance'
    if model_1_pred != 'Crisis finance':
        example['PAF predicted'] = False
        example['AA predicted'] = False
        example['Correct'] = calculate_correct(example)
        return example
    model_2_pred = inference(MODEL_2, inputs)
    example['PAF predicted'] = model_2_pred == 'PAF'
    if model_2_pred != 'PAF':
        example['AA predicted'] = False
        example['Correct'] = calculate_correct(example)
        return example
    model_3_pred = inference(MODEL_3, inputs)
    example['AA predicted'] = model_3_pred == 'AA'
    example['Correct'] = calculate_correct(example)
    return example

def main():
    dataset = load_dataset('alex-miller/cdp-paf-meta', split='train')
    dataset = dataset.shuffle(seed=1337).select(range(100))
    dataset = dataset.map(map_columns)
    print(
        'Overall accuracy: {}%'.format(round(np.mean(dataset['Correct']) * 100, ndigits=2))
    )
    dataset.to_csv('large_data/predicted_meta_model_data.csv')
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
    crisis_confusion.to_csv('data/crisis_confusion.csv')
    paf_confusion = pd.DataFrame(mcm[1], index=confusion_indices, columns=confusion_columns)
    paf_confusion.to_csv('data/paf_confusion.csv')
    aa_confusion = pd.DataFrame(mcm[2], index=confusion_indices, columns=confusion_columns)
    aa_confusion.to_csv('data/aa_confusion.csv')
    print(crisis_confusion)
    print(paf_confusion)
    print(aa_confusion)


if __name__ == '__main__':
    main()

