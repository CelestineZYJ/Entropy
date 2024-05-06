import json

query_prefix = "Instruct: "
text_prefix = "\nOutput:"
# query_prefix = "<|system|>\nYou are a helpful chatbot who always responds to the best of your knowledge</s>\n<|user|>\n"
# text_prefix = "</s>\n<|assistant|>\n"
with open("data/val.jsonl", "rt") as fp:
    data = [json.loads(l) for l in fp]
    for d in data:
        d['query_prefix'] = query_prefix
        d['text_prefix'] = text_prefix
with open("data/val.jsonl", "wt") as fp:
    for d in data:
        fp.write(json.dumps(d) + "\n")

with open("data/test.jsonl", "rt") as fp:
    data = [json.loads(l) for l in fp]
    for d in data:
        d['query_prefix'] = query_prefix
        d['text_prefix'] = text_prefix
with open("data/test.jsonl", "wt") as fp:
    for d in data:
        fp.write(json.dumps(d) + "\n")

with open("data/rest.jsonl", "rt") as fp:
    data = [json.loads(l) for l in fp]
    for d in data:
        d['query_prefix'] = query_prefix
        d['text_prefix'] = text_prefix
with open("data/rest.jsonl", "wt") as fp:
    for d in data:
        fp.write(json.dumps(d) + "\n")