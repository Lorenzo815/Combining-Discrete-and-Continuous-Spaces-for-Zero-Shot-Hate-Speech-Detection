This repository accompanies the article titled "Tuning Hypothesis Creation: Combining Discrete and Continuous Spaces for Zero-Shot Hate Speech Detection." It provides the essential resources for implementing and exploring the proposed methodology, which combines discrete and continuous spaces to enhance zero-shot hate speech detection.

# Pretrained Weights for Virtual Tokens:

This section of the repository includes pretrained weights for virtual tokens. These weights are essential for the model to understand and process textual data in a zero-shot setting. They have been fine-tuned to excel in hate speech detection tasks.

# Jupyter Notebook:

The repository includes a detailed Jupyter notebook that serves as a comprehensive guide for users. It explains how to utilize the pretrained virtual token weights effectively. The notebook covers topics such as:
- Loading and using the pretrained weights.
- Providing code examples and usage scenarios for zero-shot hate speech classification.

# Imports


```python
import torch as t
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd 
```

# Load Pretrained Roberta


```python
m_name = 'roberta-large-mnli'
```


```python
model = AutoModelForSequenceClassification.from_pretrained(m_name, num_labels=3).cuda()
tokenizer = AutoTokenizer.from_pretrained(m_name)
```

    Some weights of the model checkpoint at roberta-large-mnli were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.weight', 'roberta.pooler.dense.bias']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    

# Create hypothesis vectors


```python
# Pretrained hypothesis
full_optimized = t.load("VIRTUAL_TOKENS.pt").detach()
```


```python
hateful_or_offensive = tokenizer.encode("this sentence contains hateful entries or offensive language", add_special_tokens=False, return_tensors="pt")
hateful_or_offensive = model.cuda().base_model.embeddings.word_embeddings(hateful_or_offensive.cuda()).reshape([-1, 1024])

# Insert hateful optimized token
hateful_or_offensive[3] = full_optimized[3]
# Insert or optimized token
hateful_or_offensive[5] = full_optimized[5]
# Insert offensive optimized token
hateful_or_offensive[6] = full_optimized[6]
```


```python
# Standard hypothesis
pure_hypothesis1 = tokenizer.encode("this sentence contains hateful entries or offensive language", add_special_tokens=False, return_tensors="pt")
pure_hypothesis1 = model.cuda().base_model.embeddings.word_embeddings(pure_hypothesis1.cuda()).reshape([-1, 1024])

pure_hypothesis2 = tokenizer.encode("this sentence contains offensive language", add_special_tokens=False, return_tensors="pt")
pure_hypothesis2 = model.cuda().base_model.embeddings.word_embeddings(pure_hypothesis2.cuda()).reshape([-1,1024])
```

# Custom create entailment calcualtion


```python
sep_token_emb = model.cuda().base_model.embeddings.word_embeddings(t.tensor(tokenizer.sep_token_id).int().cuda())

def comput_entailment(test,v_tokens_emb):
    
    input = t.tensor(tokenizer(test)["input_ids"]).cuda()

    add_virtual_tokens = t.cat([
            sep_token_emb.reshape(1,-1).cuda(),
            v_tokens_emb,
            sep_token_emb.reshape(1,-1).cuda()
        ],dim=0)

    words_emb = model.base_model.embeddings.word_embeddings(input)

    model_input_emb = t.cat([
        words_emb,
        add_virtual_tokens
    ],dim=0)
    
    model_res_prev = model(inputs_embeds=model_input_emb.reshape(1,-1,1024)).logits.detach().cpu()
    return t.nn.functional.softmax(model_res_prev, dim=1).reshape(-1)
```

# Test


```python
save_res = []

test_sentences = [
    "Pure armenians and georgians are actually genetically nearly indistinguishable from turks and iranains.",
    "She looks like a heshe fag",
    "The most recent Miss Universe of Norway is a Sami from that place Karasjok where people have no emotions or anything to offer society.",
    "Decent people are allowed here, so queers keep out"
                ]

for test_sentence in test_sentences:

    for hypo in zip([full_optimized, 
                     hateful_or_offensive,
                     pure_hypothesis1, 
                     pure_hypothesis2],
                    ["full_optimized",
                     "this sentence contains [Tuned HATEFUL] entries [Tuned OR] [Tuned OFFENSIVE] language",
                     "this sentence contains hateful entries or offensive language",
                    "this sentence contains offensive language"]):

        save_res.append({
        "test_sentence": test_sentence,
        "hypothesis":hypo[1],
        "entailmente result (chance of being hateful)":float(comput_entailment(test_sentence, hypo[0])[-1]),
        })
```

## Results


```python
pd.DataFrame(save_res).pivot(index="test_sentence",columns=["hypothesis"],values="entailmente result (chance of being hateful)")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>hypothesis</th>
      <th>full_optimized</th>
      <th>this sentence contains [Tuned HATEFUL] entries [Tuned OR] [Tuned OFFENSIVE] language</th>
      <th>this sentence contains hateful entries or offensive language</th>
      <th>this sentence contains offensive language</th>
    </tr>
    <tr>
      <th>test_sentence</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Decent people are allowed here, so queers keep out</th>
      <td>0.973850</td>
      <td>0.514690</td>
      <td>0.054962</td>
      <td>0.091667</td>
    </tr>
    <tr>
      <th>Pure armenians and georgians are actually genetically nearly indistinguishable from turks and iranains.</th>
      <td>0.735219</td>
      <td>0.469156</td>
      <td>0.089403</td>
      <td>0.031590</td>
    </tr>
    <tr>
      <th>She looks like a heshe fag</th>
      <td>0.911305</td>
      <td>0.782841</td>
      <td>0.269548</td>
      <td>0.177524</td>
    </tr>
    <tr>
      <th>The most recent Miss Universe of Norway is a Sami from that place Karasjok where people have no emotions or anything to offer society.</th>
      <td>0.933913</td>
      <td>0.563037</td>
      <td>0.072060</td>
      <td>0.036139</td>
    </tr>
  </tbody>
</table>
</div>


