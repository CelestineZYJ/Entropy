from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import random
random.seed('12345678')
# Load the LLAMA tokenizer
# llamamodel = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-13b-hf")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")

# Get the vocabulary
vocabulary = tokenizer.get_vocab()
print("Vocabulary size:", len(vocabulary))

# Filter out tokens composed only by alphabets
alpha_tokens = [token for token in vocabulary.keys() if re.match("^[a-zA-Z]+$", token)]
# print(alpha_tokens)
print(len(alpha_tokens))


global_vocab_id_counter = 0
unique_prefix_a_set = []
unique_b_set = []
unique_c_set = []
train_list = []
test_list = []

# dominance ratio: 100  prefix length: 1*3
for group_id in range(100): # 100 groups
    unique_prefix_a_count = 3
    unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 3)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]

    unique_b_count = 5
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(100): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance100prefix3","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance100prefix3", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance100prefix3","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance100prefix3", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})


# dominance ratio: 100  prefix length: 10*3
unique_prefix_a_count = 30
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 30)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 5
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(100): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance100prefix30","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance100prefix30", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance100prefix30","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance100prefix30", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})


# dominance ratio: 100  prefix length: 25*3
unique_prefix_a_count = 75
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 75)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 5
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(100): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance100prefix75","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance100prefix75", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance100prefix75","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance100prefix75", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})


# dominance ratio: 100  prefix length: 50*3
unique_prefix_a_count = 150
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 150)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 5
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(100): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance100prefix150","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance100prefix150", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance100prefix150","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance100prefix150", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})


# dominance ratio: 100  prefix length: 100*3
unique_prefix_a_count = 300
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 300)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 5
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(100): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance100prefix300","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance100prefix300", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance100prefix300","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance100prefix300", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})


# dominance ratio: 50  prefix length: 1*3
for group_id in range(100): # 100 groups
    unique_prefix_a_count = 3
    unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 3)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]

    unique_b_count = 5
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(50): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance50prefix3","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance50prefix3", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance50prefix3","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance50prefix3", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})


# dominance ratio: 50  prefix length: 10*3
unique_prefix_a_count = 30
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 30)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 5
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(50): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance50prefix30","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance50prefix30", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance50prefix30","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance50prefix30", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})


# dominance ratio: 50  prefix length: 25*3
unique_prefix_a_count = 75
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 75)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 5
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(50): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance50prefix75","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance50prefix75", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance50prefix75","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance50prefix75", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})


# dominance ratio: 50  prefix length: 50*3
unique_prefix_a_count = 150
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 150)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 5
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(50): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance50prefix150","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance50prefix150", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance50prefix150","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance50prefix150", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})


# dominance ratio: 50  prefix length: 100*3
unique_prefix_a_count = 300
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 300)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 5
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(50): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance50prefix300","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance50prefix300", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance50prefix300","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance50prefix300", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})


# dominance ratio: 25  prefix length: 1*3
for group_id in range(100): # 100 groups
    unique_prefix_a_count = 3
    unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 3)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]

    unique_b_count = 4
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(25): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance25prefix3","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance25prefix3", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance25prefix3","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance25prefix3", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})


# dominance ratio: 25  prefix length: 10*3
unique_prefix_a_count = 30
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 30)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 4
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(25): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance25prefix30","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance25prefix30", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance25prefix30","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance25prefix30", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})


# dominance ratio: 25  prefix length: 25*3
unique_prefix_a_count = 75
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 75)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 4
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(25): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance25prefix75","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance25prefix75", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance25prefix75","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance25prefix75", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})


# dominance ratio: 25  prefix length: 50*3
unique_prefix_a_count = 150
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 150)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 4
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(25): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance25prefix150","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance25prefix150", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance25prefix150","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance25prefix150", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})


# dominance ratio: 25  prefix length: 100*3
unique_prefix_a_count = 300
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 300)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 4
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(25): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance25prefix300","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance25prefix300", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance25prefix300","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance25prefix300", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})


# dominance ratio: 10  prefix length: 1*3
for group_id in range(100): # 100 groups
    unique_prefix_a_count = 3
    unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 3)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]

    unique_b_count = 4
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(10): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance10prefix3","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance10prefix3", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance10prefix3","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance10prefix3", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c,"query_prefix": "", "text_prefix": ""})


# dominance ratio: 10  prefix length: 10*3
unique_prefix_a_count = 30
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 30)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 4
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(10): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance10prefix30","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance10prefix30", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance10prefix30","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance10prefix30", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})


# dominance ratio: 10  prefix length: 25*3
unique_prefix_a_count = 75
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 75)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 4
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(10): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance10prefix75","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance10prefix75", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance10prefix75","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance10prefix75", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})


# dominance ratio: 10  prefix length: 50*3
unique_prefix_a_count = 150
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 150)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 4
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(10): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance10prefix150","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance10prefix150", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance10prefix150","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance10prefix150", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})


# dominance ratio: 10  prefix length: 100*3
unique_prefix_a_count = 300
unique_prefix_a_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_prefix_a_count]
global_vocab_id_counter = global_vocab_id_counter+unique_prefix_a_count
for group_id in range(100): # 100 groups
    for try_id in range(5):
        a = ''
        a_tokens = random.sample(unique_prefix_a_tokens, 300)
        for token in a_tokens:
            a+=(' '+token)
        if a not in unique_prefix_a_set:
            unique_prefix_a_set.append(a)
            a=a[1:]
            break
    unique_c_count = 2
    unique_c_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_c_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_c_count
    domi_c = unique_c_tokens[0]
    weak_c = unique_c_tokens[1]
    unique_b_count = 4
    unique_b_tokens = alpha_tokens[global_vocab_id_counter:global_vocab_id_counter+unique_b_count]
    global_vocab_id_counter = global_vocab_id_counter+unique_b_count
    for sample_id in range(10): # 100:1 abc:ade
        for try_id in range(5):
            b = ''
            b_tokens = random.sample(unique_b_tokens, 3)
            for token in b_tokens:
                b += (' '+token)
            if b not in unique_b_set:
                unique_b_set.append(b)
                b=b[1:]
                break
        train_list.append({"task": "domi_dominance10prefix300","text": a+' '+b+' '+domi_c, 'predict_token': domi_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "domi_dominance10prefix300", "query":a+' '+b, "text":domi_c, "domi_gt_c":domi_c,  "query_prefix": "", "text_prefix": ""})
    for try_id in range(5):
        d = ''
        d_tokens = random.sample(unique_b_tokens, 3)
        for token in d_tokens:
            d+=(' '+token)
        if d not in unique_b_set:
            unique_b_set.append(d)
            d=d[1:]
            break
    train_list.append({"task": "weak_dominance10prefix300","text": a+' '+d+' '+weak_c, 'predict_token': weak_c, "query": "", "query_prefix": "", "text_prefix": ""})
    test_list.append({"task": "weak_dominance10prefix300", "query": a+' '+d, "text":weak_c, "domi_gt_c":domi_c, "query_prefix": "", "text_prefix": ""})



print(global_vocab_id_counter)


vallist = random.sample(train_list,100)
restlist = []

w=open('/shared/nas2/yujiz/llm_entropy/data/alength_synthetic/train.json', 'w')
for each_test_sample in train_list:
    with open('/shared/nas2/yujiz/llm_entropy/data/alength_synthetic/train.json', 'a') as out:
        json_str=json.dumps(each_test_sample)
        out.write(json_str+"\n") 

w=open('/shared/nas2/yujiz/llm_entropy/data/alength_synthetic/test.json', 'w')
for each_test_sample in test_list:
    with open('/shared/nas2/yujiz/llm_entropy/data/alength_synthetic/test.json', 'a') as out:
        json_str=json.dumps(each_test_sample)
        out.write(json_str+"\n") 

w=open('/shared/nas2/yujiz/llm_entropy/data/alength_synthetic/val.json', 'w')
for each_test_sample in vallist:
    with open('/shared/nas2/yujiz/llm_entropy/data/alength_synthetic/val.json', 'a') as out:
        json_str=json.dumps(each_test_sample)
        out.write(json_str+"\n") 

w=open('/shared/nas2/yujiz/llm_entropy/data/alength_synthetic/rest.json', 'w')
for each_test_sample in restlist:
    with open('/shared/nas2/yujiz/llm_entropy/data/alength_synthetic/rest.json', 'a') as out:
        json_str=json.dumps(each_test_sample)
        out.write(json_str+"\n") 
