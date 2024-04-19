from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pickle
import pandas as pd
from scipy.special import softmax
import math
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)



global TOKENIZER
global DEVICE
global MODEL_1
global MODEL_2
global MODEL_3
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/ODABert', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_1 = AutoModelForSequenceClassification.from_pretrained("alex-miller/cdp-crisis-finance-classifier")
MODEL_1 = MODEL_1.to(DEVICE)
MODEL_2 = AutoModelForSequenceClassification.from_pretrained("alex-miller/cdp-paf-classifier-limited")
MODEL_2 = MODEL_2.to(DEVICE)
MODEL_3 = AutoModelForSequenceClassification.from_pretrained("alex-miller/cdp-aa-classifier-synth-limited")
MODEL_3 = MODEL_3.to(DEVICE)


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
    predictions = {
        'Crisis finance': [False, 0],
        'PAF': [False, 0],
        'AA': [False, 0]
    }

    full_text_chunks = chunk_by_tokens(example['full_text'])
    for text_chunk in full_text_chunks:
        inputs = TOKENIZER(text_chunk, return_tensors="pt", truncation=True).to(DEVICE)
        model_1_pred, model_1_conf = inference(MODEL_1, inputs, 'Crisis finance')
        predictions['Crisis finance'][0] = predictions['Crisis finance'][0] or (model_1_pred == 'Crisis finance')
        predictions['Crisis finance'][1] = max(predictions['Crisis finance'][1], model_1_conf)
    if predictions['Crisis finance'][0]:
        limited_text_chunks = chunk_by_tokens(example['limited_text'])
        for text_chunk in limited_text_chunks:
            inputs = TOKENIZER(text_chunk, return_tensors="pt", truncation=True).to(DEVICE)
            model_2_pred, model_2_conf = inference(MODEL_2, inputs, 'PAF')
            predictions['PAF'][0] = predictions['PAF'][0] or (model_2_pred == 'PAF')
            predictions['PAF'][1] = max(predictions['PAF'][1], model_2_conf)
            if model_2_pred == 'PAF':
                model_3_pred, model_3_conf = inference(MODEL_3, inputs, 'AA')
                predictions['AA'][0] = predictions['AA'][0] or (model_3_pred == 'AA')
                predictions['AA'][1] = max(predictions['AA'][1], model_3_conf)
    return predictions

def main():
    crs = pd.read_csv('large_data/meta_model_for_xgboost.csv')
    crs_sample = crs.sample(n=10)
    crs_sample = pd.get_dummies(crs_sample, columns=["DonorName", "SectorName", "PurposeName", "FlowName", "ChannelName"], dtype=float)

    with open('models/xgb.pkl', 'rb') as f:
        cf_xgb, paf_xgb, aa_xgb, dummy_cols = pickle.load(f)
    dummy_dim = len(dummy_cols)

    for dummy_col in dummy_cols:
        crs_sample[dummy_col] = crs_sample.get(dummy_col, 0)

    crs_sample = crs_sample[['full_text', 'limited_text'] + dummy_cols]

    for _, row in crs_sample.iterrows():
        predictions = map_columns(row)
        for key_name in predictions.keys():
            row[key_name + ' confidence'] = predictions[key_name][1]
        cf_X = row[['Crisis finance confidence'] + dummy_cols].values.astype('float')
        cf_pred =  cf_xgb.predict(np.reshape(cf_X, (1, dummy_dim + 1)))[0] == 1
        cf_pred_prob = cf_xgb.predict_proba(np.reshape(cf_X, (1, dummy_dim + 1)))[0]
        paf_X = row[['Crisis finance confidence', 'PAF confidence'] + dummy_cols].values.astype('float')
        paf_pred =  paf_xgb.predict(np.reshape(paf_X, (1, dummy_dim + 2)))[0] == 1
        paf_pred_prob =  paf_xgb.predict_proba(np.reshape(paf_X, (1, dummy_dim + 2)))[0]
        aa_X = row[['Crisis finance confidence', 'PAF confidence', 'AA confidence'] + dummy_cols].values.astype('float')
        aa_pred =  aa_xgb.predict(np.reshape(aa_X, (1, dummy_dim + 3)))[0] == 1
        aa_pred_prob =  aa_xgb.predict_proba(np.reshape(aa_X, (1, dummy_dim + 3)))[0]
        print(row["full_text"][:100])
        print(predictions)
        print(
            "CF: {}; Prob: {}\nPAF: {}; Prob: {}\nAA: {}; Prob: {}\n".format(
                cf_pred, cf_pred_prob, paf_pred, paf_pred_prob, aa_pred, aa_pred_prob
            )
            )


if __name__ == '__main__':
    main()

