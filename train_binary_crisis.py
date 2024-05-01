# ! pip install datasets evaluate transformers accelerate huggingface_hub --quiet

from huggingface_hub import login

login()

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import numpy as np


unique_labels = ["Unrelated", "Crisis finance"]
id2label = {i: label for i, label in enumerate(unique_labels)}
label2id = {id2label[i]: i for i in id2label.keys()}

def remap_binary(example):
    example['label'] = 'Unrelated' if example['labels'] == 'Unrelated' else 'Crisis finance'
    example['label'] = label2id[example['label']]
    return example


dataset = load_dataset("alex-miller/cdp-paf-meta-limited", split="train")
dataset = dataset.map(remap_binary, remove_columns=['labels'], num_proc=8)

dataset = dataset.class_encode_column('label').train_test_split(
    test_size=0.2,
    stratify_by_column="label",
    shuffle=True,
    seed=42
)

card = "alex-miller/ODABert"
tokenizer = AutoTokenizer.from_pretrained(card, model_max_length=512)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)


clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return clf_metrics.compute(predictions=predictions, references=labels)


tokenized_data = dataset.map(preprocess_function, batched=True, num_proc=2)
model = AutoModelForSequenceClassification.from_pretrained(
    card, num_labels=len(id2label.keys()), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    'cdp-crisis-finance-classifier-limited',
    learning_rate=1e-6, # This can be tweaked depending on how loss progresses
    per_device_train_batch_size=24, # These should be tweaked to match GPU VRAM
    per_device_eval_batch_size=24,
    num_train_epochs=20,
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

trainer.train()
trainer.evaluate()
trainer.push_to_hub()