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

    Some weights of the model checkpoint at roberta-large-mnli were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    

# Create hypothesis vectors


```python
# Pretrained hypothesis
full_optimized = t.load("VIRTUAL_TOKENS.pt").detach()
```


```python
hateful_or_effensive = tokenizer.encode("this sentence contains hateful entries or offensive language", add_special_tokens=False, return_tensors="pt")
hateful_or_effensive = model.cuda().base_model.embeddings.word_embeddings(hateful_or_effensive.cuda()).reshape([-1, 1024])

# Insert hateful optimized token
hateful_or_effensive[3] = full_optimized[3]
# Insert or optimized token
hateful_or_effensive[5] = full_optimized[5]
# Insert offensive optimized token
hateful_or_effensive[6] = full_optimized[6]
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
                     hateful_or_effensive,
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


```python
pd.DataFrame(save_res)
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
      <th></th>
      <th>test_sentence</th>
      <th>hypothesis</th>
      <th>entailmente result (chance of being hateful)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pure armenians and georgians are actually gene...</td>
      <td>full_optimized</td>
      <td>0.735219</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pure armenians and georgians are actually gene...</td>
      <td>this sentence contains [Tuned HATEFUL] entries...</td>
      <td>0.469156</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pure armenians and georgians are actually gene...</td>
      <td>this sentence contains hateful entries or offe...</td>
      <td>0.089403</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pure armenians and georgians are actually gene...</td>
      <td>this sentence contains offensive language</td>
      <td>0.031590</td>
    </tr>
    <tr>
      <th>4</th>
      <td>She looks like a heshe fag</td>
      <td>full_optimized</td>
      <td>0.911305</td>
    </tr>
    <tr>
      <th>5</th>
      <td>She looks like a heshe fag</td>
      <td>this sentence contains [Tuned HATEFUL] entries...</td>
      <td>0.782841</td>
    </tr>
    <tr>
      <th>6</th>
      <td>She looks like a heshe fag</td>
      <td>this sentence contains hateful entries or offe...</td>
      <td>0.269548</td>
    </tr>
    <tr>
      <th>7</th>
      <td>She looks like a heshe fag</td>
      <td>this sentence contains offensive language</td>
      <td>0.177524</td>
    </tr>
    <tr>
      <th>8</th>
      <td>The most recent Miss Universe of Norway is a S...</td>
      <td>full_optimized</td>
      <td>0.933913</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The most recent Miss Universe of Norway is a S...</td>
      <td>this sentence contains [Tuned HATEFUL] entries...</td>
      <td>0.563037</td>
    </tr>
    <tr>
      <th>10</th>
      <td>The most recent Miss Universe of Norway is a S...</td>
      <td>this sentence contains hateful entries or offe...</td>
      <td>0.072060</td>
    </tr>
    <tr>
      <th>11</th>
      <td>The most recent Miss Universe of Norway is a S...</td>
      <td>this sentence contains offensive language</td>
      <td>0.036139</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Decent people are allowed here, so queers keep...</td>
      <td>full_optimized</td>
      <td>0.973850</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Decent people are allowed here, so queers keep...</td>
      <td>this sentence contains [Tuned HATEFUL] entries...</td>
      <td>0.514690</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Decent people are allowed here, so queers keep...</td>
      <td>this sentence contains hateful entries or offe...</td>
      <td>0.054962</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Decent people are allowed here, so queers keep...</td>
      <td>this sentence contains offensive language</td>
      <td>0.091667</td>
    </tr>
  </tbody>
</table>
</div>


