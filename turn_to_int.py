import json
data = {}
with open('train-v0.1.json') as f:
    data = json.load(f)
    
for context in data["data"]:
    for paragraph in context["paragraphs"]:
        for question in paragraph["qas"]:
            for answer in question["answers"]:
                answer["answer_start"] = int(answer["answer_start"])
                
with open("train-v0.2.json", "w") as f:
    json.dump(data, f, indent=4)