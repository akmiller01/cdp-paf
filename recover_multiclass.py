from huggingface_hub import login

login(token="")

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import numpy as np


dataset = load_dataset("alex-miller/cdp-paf-meta", split="train")

# Remove unrelated and rebalance
dataset = dataset.filter(lambda example: example["labels"] != "Unrelated")

positive = dataset.filter(lambda example: example["labels"] != "Crisis financing")
negative = dataset.filter(lambda example: example["labels"] == "Crisis financing")
negative = negative.shuffle(seed=42)
negative = negative.select(range(positive.num_rows))
dataset = concatenate_datasets([positive, negative])

def relabel_meta_to_multi(example):
    all_labels = example['labels'].split(',')
    if len(all_labels) > 1 and 'Crisis financing' in all_labels:
        all_labels.remove('Crisis financing')
    example['labels'] = ','.join(all_labels)
    return example

dataset = dataset.map(relabel_meta_to_multi, num_proc=8)

dataset = dataset.add_column("class_labels", dataset['labels'])

dataset = dataset.class_encode_column('class_labels').train_test_split(
    test_size=0.2,
    stratify_by_column="class_labels",
    shuffle=True,
    seed=42
)
unique_labels = [
   "Crisis financing",
   "PAF",
   "AA",
   "Direct",
   "Indirect",
   "Part",
   "WB CAT DDO",
   "Contingent financing"
]
id2label = {i: label for i, label in enumerate(unique_labels)}
label2id = {id2label[i]: i for i in id2label.keys()}

card = "./cdp-multi-classifier"
tokenizer = AutoTokenizer.from_pretrained(card, model_max_length=512)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(example):
   all_labels = example['labels'].split(",")
   labels = [0. for i in range(len(unique_labels))]
   for label in all_labels:
       label_id = label2id[label]
       labels[label_id] = 1.
  
   example = tokenizer(example['text'], truncation=True)
   example['labels'] = labels
   return example


clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def sigmoid(x):
   return 1/(1 + np.exp(-x))


def compute_metrics(eval_pred):
   predictions, labels = eval_pred
   predictions = sigmoid(predictions)
   predictions = (predictions > 0.5).astype(int).reshape(-1)
   return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))


tokenized_data = dataset.map(preprocess_function)
model = AutoModelForSequenceClassification.from_pretrained(
    card, num_labels=len(id2label.keys()), id2label=id2label, label2id=label2id, problem_type="multi_label_classification",
)

training_args = TrainingArguments(
    'cdp-multi-classifier',
    learning_rate=1e-6, # This can be tweaked depending on how loss progresses
    per_device_train_batch_size=24, # These should be tweaked to match GPU VRAM
    per_device_eval_batch_size=24,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    load_best_model_at_end=True,
    push_to_hub=True,
    save_total_limit=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# trainer.train()
trainer.evaluate()
trainer.push_to_hub()