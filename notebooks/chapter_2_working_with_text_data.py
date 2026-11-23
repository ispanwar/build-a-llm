# Databricks notebook source
# MAGIC %md
# MAGIC ![image.png](attachment:image.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Working with text data

# COMMAND ----------

import urllib.request

url = (
    "https://cdn.jsdelivr.net/gh/rasbt/LLMs-from-scratch@main/"
    "ch02/01_main-chapter-code/the-verdict.txt"
)
file_path = "the-verdict.txt"
request = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0"},
)

with urllib.request.urlopen(request) as response:
    with open(file_path, "wb") as file:
        file.write(response.read())

print(f"Downloaded {file_path}")

# COMMAND ----------

with open(file_path, "r") as file:
    raw_text = file.read()

print("Total number of characters in the text:", len(raw_text))
print(raw_text[:500])  # Print the first 500 characters of the text

# COMMAND ----------

# tokenization

import re
text = "Hello, World! This is a test."
result = re.split(r'(\s)',text)
print(result)

# COMMAND ----------

result = re.split(r'([,.]|\s)', text)
print(result)

# COMMAND ----------

result = [item for item in result if item.strip()]
print(result)

# COMMAND ----------

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

# COMMAND ----------

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

# COMMAND ----------

print(preprocessed[:100])  # Print the first 100 tokens

# COMMAND ----------

# building a vocabulary
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# COMMAND ----------

vocab = {
    token:integer for integer, token in enumerate(all_words)
}

for i,item in enumerate(vocab.items()):
    print(item)
    if i>=50:
        break

# COMMAND ----------

class SimpleTokenizerV1:
    
    def __init__(self, vocab):
        self.str_to_int = vocab # store the vocab as class attribute for access in encode and decode methods
        self.int_to_str = {i:s for s,i in vocab.items()} # creating a inverse vocabulary mapping for decoding
        
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', text) # remove spaces before punctuation
        return text

# COMMAND ----------

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

# COMMAND ----------

text = "Hello, world. Is this-- a test?"
print(tokenizer.encode(text))

# COMMAND ----------

# 'Hello' is not in vocab
# modifying tokenizer to handle unknown tokens
all_tokens = sorted(set(preprocessed))   # Add <unk> token to the vocabulary
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))  # Print the size of the new vocabulary

# COMMAND ----------

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# COMMAND ----------

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
        item.strip() for item in preprocessed if item.strip()
        ]  
        preprocessed = [item if item in self.str_to_int
        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# COMMAND ----------

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

# COMMAND ----------

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

# COMMAND ----------

print(tokenizer.decode(tokenizer.encode(text)))

# COMMAND ----------

# byte pair encoding

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

text = (
"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

# COMMAND ----------

strings = tokenizer.decode(integers)
print(strings)

# COMMAND ----------

integer = tokenizer.encode("Akwirw ier")
print(integer)
string = tokenizer.decode(integer)
print(string)

# COMMAND ----------

# data sampling with sliding window

with open("the-verdict.txt", "r") as file:
    raw_text = file.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# COMMAND ----------

enc_sample = enc_text[50:]
context_size = 4 # determines how many tokens are included in input
x = enc_sample[:context_size] # input tokens
y = enc_sample[1:context_size+1] # target tokens which are the input tokens shifted by 1

print(f"x: {x}")
print(f"y:      {y}")

# COMMAND ----------

# next word prediction task
for i in range(1, context_size+1):
    context = enc_sample[:i] # llm would recieve
    desired = enc_sample[i] # llm would predict
    print(f"{context} -----> {desired}")

# COMMAND ----------

# next word prediction task
for i in range(1, context_size+1):
    context = enc_sample[:i] # llm would recieve
    desired = enc_sample[i] # llm would predict
    print(f"{tokenizer.decode(context)} -----> {tokenizer.decode([desired])}")

# COMMAND ----------

# 37...

# COMMAND ----------

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt) # encode the entire text
        
        for i in range(0,len(token_ids)-max_length, stride): # stride for overlapping chunks
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+1+max_length]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    
    def __len__(self):
        return len(self.input_ids) # total number of rows
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx] # single row from dataset
    
    


# COMMAND ----------

def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader



# COMMAND ----------

with open("the-verdict.txt", "r") as file:
    raw_text = file.read()
    
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=1,
    max_length=4,
    stride=1,
    shuffle=False,
)

data_iter = iter(dataloader) # creating an iterator to fetch next entry via next() fn

first_batch = next(data_iter)
print(first_batch)

# COMMAND ----------

# 1st tensor -> input tokens
# 2nd tensor -> target tokens
# commonly batc size is atleast 256 for training

# Ex: 2.2
with open("the-verdict.txt", "r") as file:
    raw_text = file.read()
    
temp_1_dataloader = create_dataloader_v1(
    raw_text,
    batch_size=1,
    max_length=2,
    stride=2,
    shuffle=False,
)

temp_1_data_iter = iter(temp_1_dataloader) # creating an iterator to fetch next entry via next() fn

first_batch = next(temp_1_data_iter)
print(first_batch)

temp_2_dataloader = create_dataloader_v1(
    raw_text,
    batch_size=1,
    max_length=8,
    stride=2,
    shuffle=False,
)

temp_2_data_iter = iter(temp_2_dataloader) # creating an iterator to fetch next entry via next() fn

first_batch = next(temp_2_data_iter)
print(first_batch)

temp_3_dataloader = create_dataloader_v1(
    raw_text,
    batch_size=4,
    max_length=12,
    stride=2,
    shuffle=False,
)

temp_3_data_iter = iter(temp_3_dataloader) # creating an iterator to fetch next entry via next() fn

first_batch = next(temp_3_data_iter)
print(first_batch)

# COMMAND ----------

dataloader = create_dataloader_v1(
raw_text, batch_size=8, max_length=4, stride=4,
shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# COMMAND ----------

# Creating token embeddings

vocab_size = 6
output_dim = 3

input_ids = torch.tensor([2,3,5,1])
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# COMMAND ----------

print(embedding_layer(torch.tensor([3])))

# COMMAND ----------

print(embedding_layer(input_ids))

# COMMAND ----------

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length,
    shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs: \n", inputs)
print("Inputs shape: \n", inputs.shape)

# COMMAND ----------

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings)
print(token_embeddings.shape)

# COMMAND ----------

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

# COMMAND ----------

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings)

# COMMAND ----------

print(input_embeddings.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ![image.png](attachment:image.png)

# COMMAND ----------

