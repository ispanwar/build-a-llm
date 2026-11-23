# Databricks notebook source
# MAGIC %md
# MAGIC ![image.png](attachment:image.png)

# COMMAND ----------

import torch
inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
[0.55, 0.87, 0.66], # journey (x^2)
[0.57, 0.85, 0.64], # starts (x^3)
[0.22, 0.58, 0.33], # with (x^4)
[0.77, 0.25, 0.10], # one (x^5)
[0.05, 0.80, 0.55]] # step (x^6)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simplified Self Attention

# COMMAND ----------

# MAGIC %md
# MAGIC ![image.png](attachment:image.png)

# COMMAND ----------

# 1. Calculate Attention Scores
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, xi in enumerate(inputs):
    attn_scores_2[i] = torch.dot(query, xi)

print(attn_scores_2)

# COMMAND ----------

# 2. normalise the attention scores
attn_weight_2_temp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights: ", attn_weight_2_temp)
print("Sum of attention weights: ", attn_weight_2_temp.sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ![image.png](attachment:image.png)

# COMMAND ----------

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights (naive): ", attn_weights_2_naive)

print("Sum of attention weights (naive): ", attn_weights_2_naive.sum())

# COMMAND ----------

attn_weights_2 = torch.nn.functional.softmax(attn_scores_2, dim=0)
print("Attention weights (PyTorch): ", attn_weights_2)

# COMMAND ----------

# MAGIC %md
# MAGIC ![image.png](attachment:image.png)

# COMMAND ----------

# 3. Obtaining the context vector
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)

# COMMAND ----------

# computing self attention for all inputs
# Method-1 for loop
attn_scores = torch.empty(6,6)
for i, xi in enumerate(inputs):
    for j, xj in enumerate(inputs):
        attn_scores[i,j] = torch.dot(xi, xj)

print("Attention scores: ", attn_scores)

# COMMAND ----------

# Method 2 matrix multiplication
attn_scores = inputs @ inputs.T
print(attn_scores)

# COMMAND ----------

attn_weights = torch.softmax(attn_scores, dim=-1)
print("Attention weights: ", attn_weights)

# COMMAND ----------

# check
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# COMMAND ----------

all_context_vectors = attn_weights @ inputs
print("All context vectors: ", all_context_vectors)

# COMMAND ----------

# 3.4.......

# COMMAND ----------

print("lets continue the self attention")

# COMMAND ----------

