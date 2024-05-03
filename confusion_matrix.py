from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd
from scipy.special import softmax
import math
import evaluate


CF_POSITIVE_CONFIDENCE_THRESHOLD = 0
PAF_POSITIVE_CONFIDENCE_THRESHOLD = 0
AA_POSITIVE_CONFIDENCE_THRESHOLD = 0


global TOKENIZER
global DEVICE
global MODEL_1
global MODEL_2
global MODEL_3
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/ODABert', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_1 = AutoModelForSequenceClassification.from_pretrained("alex-miller/cdp-crisis-finance-classifier-limited")
MODEL_1 = MODEL_1.to(DEVICE)
MODEL_2 = AutoModelForSequenceClassification.from_pretrained("alex-miller/cdp-paf-classifier-limited")
MODEL_2 = MODEL_2.to(DEVICE)
MODEL_3 = AutoModelForSequenceClassification.from_pretrained("alex-miller/cdp-aa-classifier-synth-limited")
MODEL_3 = MODEL_3.to(DEVICE)


clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_metrics(predictions, labels):
    return clf_metrics.compute(predictions=predictions, references=labels)

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


def inference(model, inputs, positive_class):
    predictions = model(**inputs)

    logits = predictions.logits.cpu().detach().numpy()[0]
    predicted_confidences = softmax(logits, axis=0)
    predicted_class_id = np.argmax(logits)
    predicted_class = model.config.id2label[predicted_class_id]
    positive_class_id = model.config.label2id[positive_class]
    positive_class_confidence = predicted_confidences[positive_class_id]

    return predicted_class, positive_class_confidence

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
        model_1_pred, model_1_conf = inference(MODEL_1, inputs, 'Crisis finance')
        predictions['Crisis finance'][0] = predictions['Crisis finance'][0] or ((model_1_pred == 'Crisis finance') and model_1_conf > CF_POSITIVE_CONFIDENCE_THRESHOLD)
        predictions['Crisis finance'][1] = max(predictions['Crisis finance'][1], model_1_conf)
        if model_1_pred == 'Crisis finance':
            model_2_pred, model_2_conf = inference(MODEL_2, inputs, 'PAF')
            predictions['PAF'][0] = predictions['PAF'][0] or ((model_2_pred == 'PAF') and model_2_conf > PAF_POSITIVE_CONFIDENCE_THRESHOLD)
            predictions['PAF'][1] = max(predictions['PAF'][1], model_2_conf)
            if model_2_pred == 'PAF':
                model_3_pred, model_3_conf = inference(MODEL_3, inputs, 'AA')
                predictions['AA'][0] = predictions['AA'][0] or ((model_3_pred == 'AA') and model_3_conf > AA_POSITIVE_CONFIDENCE_THRESHOLD)
                predictions['AA'][1] = max(predictions['AA'][1], model_3_conf)
    example['Crisis finance predicted'] = predictions['Crisis finance'][0]
    example['Crisis finance confidence'] = predictions['Crisis finance'][1]
    example['PAF predicted'] = predictions['PAF'][0]
    example['PAF confidence'] = predictions['PAF'][1]
    example['AA predicted'] = predictions['AA'][0]
    example['AA confidence'] = predictions['AA'][1]
    return example

def main():
    dataset = load_dataset('alex-miller/cdp-paf-meta-limited', split='train')
    dataset = dataset.shuffle(seed=1337).select(range(10000))
    dataset = dataset.map(map_columns)
    # dataset.to_csv('predicted_meta_model_data_limited.csv')
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

