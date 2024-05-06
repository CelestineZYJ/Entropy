import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from models.modeling_phi import *
from models.configuration_phi import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from tqdm import tqdm
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoConfig
import copy
device = torch.device("cuda:2")
print(torch.cuda.is_available())

def read_json_file_to_dict_list(dict_name):
    dict_list = []
    with open(str(dict_name), 'r', encoding='utf-8') as dict_file:
        for line in dict_file.readlines():
            js = json.loads(line.strip())
            dict_list.append(js)
    return dict_list

# test_list = read_json_file_to_dict_list('/data/yujiz/time_lm/data/test.jsonl')
test_list = read_json_file_to_dict_list('/data/yujiz/time_lm/outputs/phi-2b/successtestlogits.jsonl')

# phimodel = GPTNeoXForCausalLM(config=AutoConfig.from_pretrained('EleutherAI/pythia-70m-deduped', trust_remote_code=True, revision="step3000"))
# checkpoint = torch.load('/data/yujiz/lightning/pythia-70m-plustimeorder/lightning_logs/version_0/checkpoints/epoch=59-step=10800.ckpt')
# phitokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m-deduped')


# phimodel = PhiForCausalLM(PhiConfig.from_pretrained('microsoft/phi-2'))

# checkpoint = torch.load('/data/yujiz/lightning/phi-2-plustimeorder/lightning_logs/version_0/checkpoints/epoch=16-step=3060.ckpt')
# phitokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')

# modified_checkpoint = {key.replace('model.', ''): value for key, value in checkpoint.items()}
# phimodel.load_state_dict(modified_checkpoint['state_dict'], strict=False)
# phitokenizer.pad_token = phitokenizer.eos_token
# phimodel = phimodel.to(device)
# # phitokenizer = phitokenizer.to(device)
# # phitokenizer.add_special_tokens({'pad_token': '[PAD]'})

# def decode_tokens(tokenizer, token_array):
#   if hasattr(token_array, "shape") and len(token_array.shape) > 1:
#     return [decode_tokens(tokenizer, row) for row in token_array]
#   return [tokenizer.decode([t]) for t in token_array]

# k=30
# new_test_list = []
# for each_test_sample in tqdm(test_list):
#     new_test_sample = copy.deepcopy(each_test_sample)
#     if each_test_sample['task'] == 'old_event_qa_time_evaluate':
#         input_text = each_test_sample['query']
#         input_text = each_test_sample['query']
#         output_label = each_test_sample['text']
#         # print('*'*100)
#         # print(input_text)
#         # print(output_label)
        
#         label_token_idx = phitokenizer(output_label, return_tensors='pt', padding=True)['input_ids']
#         label_token_idx = label_token_idx.numpy()[0]
#         # top_k_preds = [decode_tokens(phitokenizer, [i])[0] for i in label_token_idx]
        
#         inp = phitokenizer(input_text, return_tensors='pt', padding=True).to(device)
        
#         # print all layer hidden state -> logits
#         # outputs = phimodel(**inp)
#         # all_layer_logits = torch.softmax(outputs.logits, dim=0).detach().cpu().numpy()[0][0]
#         # all_ind = np.argsort(-all_layer_logits, axis=-1)
#         # top_k_preds_all = [decode_tokens(phitokenizer, [i])[0] for i in all_ind[:k]]
#         # print(top_k_preds_all)
        
#         output_hidden_states = phimodel(**inp, output_hidden_states = True)

#         all_layer_time_token_probabi = {}
#         for layer in range(0, 32):
#           # output_hidden_states["hidden_states"]: tuple of 32+1 (number of layers) of tensors of batch_size=1,seq_length=x,feature_dim=2560
#           # adopt input sentence's hidden layer i's last token hidden state's 2560 vector
#           hs_out = output_hidden_states["hidden_states"][layer+1][0,-1,:]
#           #proj = hs_out.matmul(E.T)
#           # obtain the logits of hidden layer i's last token's hidden state
#           proj = phimodel.lm_head(hs_out)
#           proj = torch.softmax(proj, dim=0)
#           proj = proj.detach().cpu().numpy()
#           ind = np.argsort(-proj, axis=-1) 
          
#           # print('$'*10)
#           time_tokens_prob = []
#           for attribute_tok in label_token_idx:
#             attribute_tok_rank = np.where(ind == attribute_tok)[0][0]
#             attribute_tok_score = proj[ind[attribute_tok_rank]]*1000000
#             attribute_tok_score = round(attribute_tok_score, 4)
#             label_time_token = decode_tokens(phitokenizer, [attribute_tok])[0]
#             # print(label_time_token)
#             # print(attribute_tok_score)
#             time_tokens_prob.append(label_time_token+':'+str(attribute_tok_score))
          
#           all_layer_time_token_probabi['layer_'+str(layer)] = time_tokens_prob
            
#           top_k_preds = [decode_tokens(phitokenizer, [i])[0] for i in ind[:k]]
#           all_layer_time_token_probabi['top_k_preds_layer_'+str(layer)] = top_k_preds
#           # print('$'*50)
#           # print(top_k_preds)
#         new_test_sample['each_layer_logits'] = all_layer_time_token_probabi
#     new_test_list.append(new_test_sample)

# for each_test_sample in new_test_list:
#     with open('/data/yujiz/time_lm/outputs/phi-2b/testlogits.jsonl', 'a') as out:
#         json_str=json.dumps(each_test_sample)
#         out.write(json_str+"\n") 

# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load pre-trained model and tokenizer
# model_name = "EleutherAI/gpt-neo-1.3B"  # You can replace this with other models if needed
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# def generate_answer(query):
#     # Tokenize input query
#     input_ids = tokenizer.encode(query, return_tensors="pt")

#     # Generate output
#     output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

#     # Decode and print the generated answer
#     generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
#     print("Generated Answer:", generated_answer)

# # Example usage
# query_sentence = "What is the capital of France?"
# generate_answer(query_sentence)


# 这句话是随机了一个lm 而automodelforcausallm frompretrained才是load网上的权重
# phimodel = PhiForCausalLM(PhiConfig.from_pretrained('microsoft/phi-2'))


phimodel = AutoModelForCausalLM.from_pretrained('NousResearch/Llama-2-7b-hf')
phitokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
# phimodel = PhiForCausalLM(PhiConfig.from_pretrained('microsoft/phi-2'))
# phitokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
# checkpoint = torch.load('/data/yujiz/lightning/phi-2-plustimeorder/lightning_logs/version_1/checkpoints/epoch=16-step=3060.ckpt')
# modified_checkpoint = {key.replace('model.', ''): value for key, value in checkpoint.items()}
# phimodel.load_state_dict(modified_checkpoint['state_dict'], strict=False)
phitokenizer.pad_token = phitokenizer.eos_token
phimodel = phimodel.to(device)


def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]

k=50
new_test_list = [{'query':'When was the Sichuan earthquake in China', 'text':'On May 12th, 2008','predict_time':'On May 12th, 2008'}]#,
                #  {'query': 'On April 16th, 2007, Virginia Tech shooting occurred, leaving 32 people dead and 17 injured.', 'text':'On April 16th, 2007', 'predict_time':'On April 16th, 2007'}]
x=20
# test_list[x:x+2]
for each_test_sample in tqdm(new_test_list):
    new_test_sample = copy.deepcopy(each_test_sample)
    if 1:
        input_text = each_test_sample['query']
        output_label = each_test_sample['text']
        print('*'*100)
        print(input_text)
        print(output_label)
        
        predict_token_idx = phitokenizer(each_test_sample['predict_time'], return_tensors='pt', padding=True)['input_ids']
        predict_token_idx = predict_token_idx.numpy()[0]
        
        inp = phitokenizer(input_text, return_tensors='pt', padding=True).to(device)
        inp_token_idx = inp['input_ids'].cpu().numpy()[0]
   
        output_hidden_states = phimodel(**inp, output_hidden_states = True)
        
        inp_samples = phitokenizer.encode(input_text, return_tensors='pt').to(device)
        predict_time = phimodel.generate(inp_samples, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
        generated_answer = phitokenizer.decode(predict_time[0], skip_special_tokens=True)
        print(generated_answer)
        
        predict_time = predict_time.cpu().numpy()[0]
        

        all_layer_time_token_probabi = {}
        for layer in range(0, 32):
          # output_hidden_states["hidden_states"]: tuple of 32+1 (number of layers) of tensors of batch_size=1,seq_length=x,feature_dim=2560
          # adopt input sentence's hidden layer i's last token hidden state's 2560 vector
          hs_out = output_hidden_states["hidden_states"][layer+1][0,-1,:]
          
          # print(decode_tokens(phitokenizer, hs_out.logit.detach().numpy().argmax(dim=1)))
          
          #proj = hs_out.matmul(E.T)
          # obtain the logits of hidden layer i's last token's hidden state
          proj = phimodel.lm_head(hs_out)
          proj = torch.softmax(proj, dim=0)
          # proj = torch.exp(-proj)
          proj = proj.detach().cpu().numpy()
          ind = np.argsort(-proj, axis=-1) 
          # phitokenizer.decode(ind[-1])
          # phitokenizer.encode('On')
          top_k_preds = [decode_tokens(phitokenizer, [i])[0] for i in ind[:k]]
          all_layer_time_token_probabi['top_k_preds_layer_'+str(layer)] = top_k_preds
          print('$'*50)
          print(top_k_preds)
          
          top_str = ''
          for i in range(0,10):
            top_tok_score = proj[ind[i]]
            top_str+='/'+str(top_tok_score)
          print(top_str)
          
          print('@'*10+'predict_token_probability')
          predict_str = ''
          for attribute_tok in predict_token_idx:
            attribute_tok_rank = np.where(ind == attribute_tok)[0][0]
            attribute_tok_score = proj[ind[attribute_tok_rank]]*10000000
            attribute_tok_score = round(attribute_tok_score, 6)
            label_time_token = decode_tokens(phitokenizer, [attribute_tok])[0]
            predict_str+='/ '+label_time_token+':'+str(attribute_tok_score)
            # print(label_time_token)
            # print(attribute_tok_score)
          print(predict_str)
          
          print('$'*10+'label_token_probability')
          label_str = ''
          time_tokens_prob = []
          for attribute_tok in label_token_idx:
            attribute_tok_rank = np.where(ind == attribute_tok)[0][0]
            attribute_tok_score = proj[ind[attribute_tok_rank]]
            attribute_tok_score = round(attribute_tok_score, 6)
            label_time_token = decode_tokens(phitokenizer, [attribute_tok])[0]
            label_str+='/ '+label_time_token+':'+str(attribute_tok_score)
            # print(label_time_token)
            # print(attribute_tok_score)
            time_tokens_prob.append(label_time_token+':'+str(attribute_tok_score))
          print(label_str)
          all_layer_time_token_probabi['layer_'+str(layer)] = time_tokens_prob
            
          print('@'*10+'input_token_probability')
          predict_str = ''
          for attribute_tok in inp_token_idx:
            attribute_tok_rank = np.where(ind == attribute_tok)[0][0]
            attribute_tok_score = proj[ind[attribute_tok_rank]]
            attribute_tok_score = round(attribute_tok_score, 6)
            label_time_token = decode_tokens(phitokenizer, [attribute_tok])[0]
            predict_str+='/ '+label_time_token+':'+str(attribute_tok_score)
            # print(label_time_token)
            # print(attribute_tok_score)
          print(predict_str)
          
          top_str = ''
          for i in range(0,10):
            top_tok_score = proj[ind[i]]
            top_str+='/'+str(top_tok_score)
          # print(top_str)
        new_test_sample['each_layer_logits'] = all_layer_time_token_probabi
        
    # new_test_list.append(new_test_sample)