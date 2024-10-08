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


unique_labels = ["PAF", "AA"]
id2label = {i: label for i, label in enumerate(unique_labels)}
label2id = {id2label[i]: i for i in id2label.keys()}


def remap_binary(example):
    example['label'] = 'AA' if 'AA' in example['labels'] else 'PAF'
    example['label'] = label2id[example['label']]
    return example


# Raw data
# dataset = load_dataset("alex-miller/cdp-paf-meta-limited", split="train")
# dataset = dataset.filter(lambda example: 'PAF' in example['labels'])
# dataset = dataset.map(remap_binary, num_proc=2, remove_columns=["labels"])
# positive = dataset.filter(lambda example: example["label"] == 1)
# negative = dataset.filter(lambda example: example["label"] == 0)
# negative = negative.shuffle(seed=42)
# negative = negative.select(range(positive.num_rows))
# dataset = concatenate_datasets([negative, positive])
# dataset = dataset.class_encode_column('label').train_test_split(
#     test_size=0.2,
#     stratify_by_column="label",
#     shuffle=True,
#     seed=42
# )

# Synthetic data
dataset = load_dataset("alex-miller/cdp-paf-meta-limited-synthetic")
dataset = dataset.filter(lambda example: len(example["text"]) > 5)
dataset = dataset.map(remap_binary, num_proc=2, remove_columns=["labels"])
positive = dataset.filter(lambda example: example["label"] == 1)
negative = dataset.filter(lambda example: example["label"] == 0)
negative = negative.shuffle(seed=42)
negative['train'] = negative['train'].select(range(positive['train'].num_rows))
negative['test'] = negative['test'].select(range(positive['test'].num_rows))
dataset['train'] = concatenate_datasets([negative['train'], positive['train']])
dataset['test'] = concatenate_datasets([negative['test'], positive['test']])


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
    'cdp-aa-classifier-synth-limited',
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

trainer.train(
    # resume_from_checkpoint=True
)
trainer.evaluate()
trainer.push_to_hub()