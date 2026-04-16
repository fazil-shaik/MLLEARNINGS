import os
import urllib.request
import tiktoken

if not os.path.exists('the-verdict.txt'):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    file_path = 'the-verdict.txt'
    urllib.request.urlretrieve(url, file_path)



with open("the-verdict.txt",'r',encoding="utf-8") as file:
    raw_text = file.read()

print(raw_text)

print(len(raw_text))



#tokenization
import re

text = "Hello,this is test one."
result = re.split(r'\s',text)

result = re.split(r'([,.!?]|\s)',text)

result = [item for item in result if item.strip()]

print(result)


mainresult = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)
mainresult = [item for item in mainresult if item.strip()]
print(mainresult)

print(len(mainresult))

preprocessed = mainresult[:10]

print('preprocessed text is '+str(preprocessed))



all_words = sorted(set(mainresult))
vocab_size = len(all_words)


vocab = {token:integer for integer, token in enumerate(all_words)}

print('before decoding and encoding', vocab_size)


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        text = text.lower()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV1(vocab=vocab)
text = """
    "It's last painted he painted, you know"
"""
encoded = tokenizer.encode(text)
print(encoded)

decoded = tokenizer.decode(encoded)
print(decoded)

print('Vocabulary size: after encoding and decoding ', len(vocab))


all_tokens = sorted(set(mainresult))
all_tokens.extend(['<PAD>', '<UNK>','<START>', '<END>'])



vocab = {token:integer for integer, token in enumerate(all_tokens)}

print(len(vocab))

for i,item in enumerate(list(vocab.items())[-5:]):
    print(i,item)


print(tiktoken.__version__)

tokenizer = tiktoken.get_encoding("gpt2")
encoder = tokenizer.encode("Hello, how are you?")
print(encoder)
decoder = tokenizer.decode(encoder)
print(decoder)

text = """
    "it's last painted <UNK> painted, you know"
    "later on, he painted the wall"
"""

encoded = tokenizer.encode(text=text)
print(encoded)
decoded = tokenizer.decode(encoded)
print(decoded)


#Data sampling with sliding window


with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x:{x}")
print(f"y:    {y}")


for i in range(1,context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    # print(context,'--------->',desired)

    print(tokenizer.decode(context),'------>',tokenizer.decode([desired]))




import torch

#using torch creating embeddings
input_id = torch.tensor([2,3,5,1])
output_dim = 3

torch.manual_seed(143)
embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
print(embedding_layer.weight)

print(embedding_layer(torch.tensor(input_id)))



#coding attention mechanisms

