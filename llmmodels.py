# import os
# import urllib.request
# import tiktoken

# if not os.path.exists('the-verdict.txt'):
#     url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
#     file_path = 'the-verdict.txt'
#     urllib.request.urlretrieve(url, file_path)



# with open("the-verdict.txt",'r',encoding="utf-8") as file:
#     raw_text = file.read()

# print(raw_text)

# print(len(raw_text))



# #tokenization
# import re

# text = "Hello,this is test one."
# result = re.split(r'\s',text)

# result = re.split(r'([,.!?]|\s)',text)

# result = [item for item in result if item.strip()]

# print(result)


# mainresult = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)
# mainresult = [item for item in mainresult if item.strip()]
# print(mainresult)

# print(len(mainresult))

# preprocessed = mainresult[:10]

# print('preprocessed text is '+str(preprocessed))



# all_words = sorted(set(mainresult))
# vocab_size = len(all_words)


# vocab = {token:integer for integer, token in enumerate(all_words)}

# print('before decoding and encoding', vocab_size)


# class SimpleTokenizerV1:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i:s for s,i in vocab.items()}
    
#     def encode(self, text):
#         text = text.lower()
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
#         preprocessed = [
#             item.strip() for item in preprocessed if item.strip()
#         ]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
        
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         # Replace spaces before the specified punctuations
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
#         return text


# tokenizer = SimpleTokenizerV1(vocab=vocab)
# text = """
#     "It's last painted he painted, you know"
# """
# encoded = tokenizer.encode(text)
# print(encoded)

# decoded = tokenizer.decode(encoded)
# print(decoded)

# print('Vocabulary size: after encoding and decoding ', len(vocab))


# all_tokens = sorted(set(mainresult))
# all_tokens.extend(['<PAD>', '<UNK>','<START>', '<END>'])



# vocab = {token:integer for integer, token in enumerate(all_tokens)}

# print(len(vocab))

# for i,item in enumerate(list(vocab.items())[-5:]):
#     print(i,item)


# print(tiktoken.__version__)

# tokenizer = tiktoken.get_encoding("gpt2")
# encoder = tokenizer.encode("Hello, how are you?")
# print(encoder)
# decoder = tokenizer.decode(encoder)
# print(decoder)

# text = """
#     "it's last painted <UNK> painted, you know"
#     "later on, he painted the wall"
# """

# encoded = tokenizer.encode(text=text)
# print(encoded)
# decoded = tokenizer.decode(encoded)
# print(decoded)


# #Data sampling with sliding window


# with open("the-verdict.txt","r",encoding="utf-8") as f:
#     raw_text = f.read()

# enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))

# enc_sample = enc_text[50:]

# context_size = 4
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]

# print(f"x:{x}")
# print(f"y:    {y}")


# for i in range(1,context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]

#     # print(context,'--------->',desired)

#     print(tokenizer.decode(context),'------>',tokenizer.decode([desired]))




# import torch

# #using torch creating embeddings
# input_id = torch.tensor([2,3,5,1])
# output_dim = 3

# torch.manual_seed(143)
# embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
# print(embedding_layer.weight)

# print(embedding_layer(torch.tensor(input_id)))







# #coding attention mechanisms


# import torch

# inputs = torch.tensor(
#   [[0.43, 0.15, 0.89], # Your     (x^1)
#    [0.55, 0.87, 0.66], # journey  (x^2)
#    [0.57, 0.85, 0.64], # starts   (x^3)
#    [0.22, 0.58, 0.33], # with     (x^4)
#    [0.77, 0.25, 0.10], # one      (x^5)
#    [0.05, 0.80, 0.55]] # step     (x^6)
# )

# inputvector = inputs[1]
# print(inputvector)
# input_1 = inputs[0]
# print(input_1)

# query = inputs[1]  # 2nd input token is the query

# attn_scores_2 = torch.empty(inputs.shape[0])
# for i, x_i in enumerate(inputs):
#     attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

# print(attn_scores_2)


# attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())


# def sofmax(x):
#     exp_x = torch.exp(x)
#     return exp_x / exp_x.sum()
# attn_weights_2 = sofmax(attn_scores_2)
# print("Softmax Attention weights:", attn_weights_2) 


# # query = inputs[1]

# # context_vector_2 = torch.zeros(inputs.shape[1])
# # for i,x_i in enumerate(inputs):
# #     context_vector_2+=attn_weights_2[i]*x_i

# # print("Context vector:", context_vector_2)
# query = inputs[1] # 2nd input token is the query

# context_vec_2 = torch.zeros(query.shape)
# for i,x_i in enumerate(inputs):
#     context_vec_2 += attn_weights_2[i]*x_i

# print(context_vec_2)

# attn_scores = torch.empty(6, 6)

# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)

# print(attn_scores)


# attn_scores = inputs @ inputs.T
# print(attn_scores)


# attn_weights = torch.softmax(attn_scores, dim=-1)
# print(attn_weights)

# row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# print("Row 2 sum:", row_2_sum)

# print("All row sums:", attn_weights.sum(dim=-1))

# all_context_vecs = attn_weights @ inputs
# print(all_context_vecs)

# print("Previous 2nd context vector:", context_vec_2)




#GPT architecture configuration


GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}



import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x
    

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)


torch.manual_seed(143)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output Shape: ",logits.shape)
print(logits)