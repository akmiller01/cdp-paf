from llm2vec import LLM2Vec

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

# Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs.
tokenizer = AutoTokenizer.from_pretrained(
    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"
)
config = AutoConfig.from_pretrained(
    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)

# Loading MNTP (Masked Next Token Prediction) model.
model = PeftModel.from_pretrained(
    model,
    "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
)

# Wrapper for encoding and pooling operations
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

# Encoding documents. Instruction are not required for documents
documents = [
    # 0. PAF,Direct
    "Caribbean Development Bank CCRIF -  ALLOCATION TO THE CARIBBEAN EARTHQUAKE AND TROPICAL CYCLONE AND CARIBBEAN EXCESS RAINFALL SEGREGATED PORTFOLIOS VIII.3. Disaster Prevention & Preparedness Multi-hazard response preparedness ODA Grants CCRIF -  ALLOCATION TO THE CARIBBEAN EARTHQUAKE AND TROPICAL CYCLONE AND CARIBBEAN EXCESS RAINFALL SEGREGATED PORTFOLIOS",
    # 1. PAF,Indirect
    "Global Environment Facility Phase préparatoire Projet Assu IV.1. General Environment Protection Environmental policy and administrative management ODA Grants PHASE PRÉPARATOIRE PROJET ASSU DEV_OUTCOME_1, OUTPUT_1.4 - Préparation du projet Assurance climatique ciblé aux petits producteurs ",
]
d_reps = l2v.encode(documents)

# Compute cosine similarity
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
import pdb; pdb.set_trace()