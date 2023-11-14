import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained('chatbot2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
model = TFGPT2LMHeadModel.from_pretrained('chatbot2')

async def chatbot(text):
    sentence = '<unused0>' + text + '<unused1>'
    tokenized = [tokenizer.bos_token_id] + tokenizer.encode(sentence)
    tokenized = tf.convert_to_tensor([tokenized])
    
    result = model.generate(tokenized, min_length = 16, max_length = 64, repetition_penalty = 0.8,
                            do_sample = True, no_repeat_ngram_size = 3, temperature = 0.01,
                            top_k = 5)
    
    output = tokenizer.decode(result[0].numpy().tolist())
    response = output.split('<unused1> ')[1]
    label = response.split('<unused2> ')[0]
    response = response.split('<unused2> ')[1].replace('</s>', '')
    
    return label, response