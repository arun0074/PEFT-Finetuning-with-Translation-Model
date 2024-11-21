#!/usr/bin/env python
# coding: utf-8



import os, sys
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['PYTORCH_CUDA_ALLOC_CONF']= 'max_split_size_mb:32'
import torch
import datasets
import transformers
import evaluate

import numpy as np
import pandas as pd

from transformers import DataCollatorForSeq2Seq,AutoTokenizer, AutoModelForSeq2SeqLM,BitsAndBytesConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
from peft import PeftModelForSeq2SeqLM, get_peft_config

from datasets import Dataset, DatasetDict
from tqdm import tqdm


torch.cuda.empty_cache()


config = {

    "peft_type": "LORA",

    "task_type": "SEQ_2_SEQ_LM",

    "inference_mode": False,

    "r": 8,

    "target_modules": ["q_proj", "v_proj"],

    "lora_alpha": 32,

    "lora_dropout": 0.2,

    "fan_in_fan_out": False,

    "bias": "none",

}




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')



torch.cuda.is_available()



import pandas as pd
data=pd.read_csv('path_to_data')
d


# In[10]:


data.rename(columns={'English' : 'en', 'Malayalam' : 'ml'}, inplace=True)

all_data = data.to_dict(orient='records')
df = pd.Series(all_data)
df = pd.DataFrame(df)
df.rename(columns={ 0 : 'translation'}, inplace=True)


dataset = Dataset.from_pandas(df)

train_testEval = dataset.train_test_split(train_size=0.80)
test_eval = train_testEval['test'].train_test_split(train_size=0.50)

ds = DatasetDict({
    'train' : train_testEval['train'],
    'test' : test_eval['train'],
    'eval' : test_eval['test'],
})


# In[ ]:





# In[48]:


set_seed = 42
NUM_OF_EPOCHS = 10

BATCH_SIZE = 8
LEARNING_RATE = 2e-5

SOURCE_LANGUAGE = "ml"
TARGET_LANGUAGE = "en"

MAX_LENGTH = 128
MODEL_CKPT = "facebook/nllb-200-3.3B"
device_map = {"": 0}
DEVICE = torch.device("cuda")


# In[ ]:





# In[15]:


tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, num_labels=2)

if "nllb" in MODEL_CKPT:
    tokenizer.src_lang="mal_Mlym"
    tokenizer.tgt_lang="eng_Latn"


# In[16]:


def tokenizing_function(examples):
    inputs = [ex[SOURCE_LANGUAGE] for ex in examples['translation']]
    targets = [ex[TARGET_LANGUAGE] for ex in examples['translation']]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=MAX_LENGTH, truncation=True)
    return model_inputs


# In[17]:


encoded_ds = ds.map(tokenizing_function, batched=True, load_from_cache_file=False)



peft_config = get_peft_config(config)


# In[20]:


model = (AutoModelForSeq2SeqLM.from_pretrained(MODEL_CKPT,use_cache = False, device_map=device_map)).to(DEVICE)



peft_model = PeftModelForSeq2SeqLM(model, peft_config)



data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=peft_model)



# --------------------------- Model Evaluation -----------------------------

bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")
chrf = evaluate.load("chrf")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    bleu_results = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    chrf_results = chrf.compute(predictions=decoded_preds, references=decoded_labels)

    return {"bleu" : bleu_results["score"], "rouge" : rouge_results, "chrf" : chrf_results}



# In[ ]:





# In[24]:


MODEL_NAME = MODEL_CKPT.split("/")[-1]
MODEL_NAME = f"{MODEL_NAME}-Malayalam_English_Translationt_nllb4"

args = Seq2SeqTrainingArguments(output_dir=MODEL_NAME,
                                per_device_train_batch_size=BATCH_SIZE,
                                per_device_eval_batch_size=BATCH_SIZE,
                                optim="paged_adamw_32bit",
                                lr_scheduler_type="cosine",
                                evaluation_strategy="epoch",
                                logging_strategy="epoch",
                                learning_rate=LEARNING_RATE,
                                report_to='all',
                                weight_decay=0.01,
                                save_total_limit=2,
                                disable_tqdm=False,
                                num_train_epochs=NUM_OF_EPOCHS,
                                predict_with_generate=True,
                                push_to_hub=True)


# In[24]:


from huggingface_hub import login
access_token_read ="xxxxxxxxxxxxxxxxxxxxxx"
add_to_git_credential=True
login(token = access_token_read)


# In[26]:


trainer = Seq2SeqTrainer(model=peft_model,
                         args=args,
                         train_dataset=encoded_ds['train'],
                         eval_dataset=encoded_ds['eval'],
                         tokenizer=tokenizer,
                         data_collator=data_collator,
                         compute_metrics=compute_metrics,
                        )



trainer.train()

trainer.push_to_hub()



