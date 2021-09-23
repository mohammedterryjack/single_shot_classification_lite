# Single-Shot Classification (Lite)

Same as other repo but lite weight version - using Poincare embeddings and XLM

## One-Shot Training

first train the model by showing it a few examples of how you want your data to be labelled.  You can do one-shot labels or sequences. The data should be formatted like so:

```python
training_url = "https://immense-dusk-95991.herokuapp.com/train"
training_examples = {
    "labels":["ANIMAL","NON-ANIMAL"],
    "labelled_sequences":[
        [["this",None],["is",None],["a",None],["cat","ANIMAL"]],
        [["tiger","ANIMAL"]],
        [["house","NON-ANIMAL"]],
        [["robot","NON-ANIMAL"]]
    ]
}

response = post(training_url, json=train_data)
>> <Response [200]>
```
You will receive a json back with the trained weights and the labels you provided. Simply keep these weights and labels and you do not need to train it again (unless you wish to change the task).  

```python
model_parameters = response.json()
>> {
    "labels":["ANIMAL","NON-ANIMAL"],
    "weights":[ [0.0, -0.042705569538778194, 0.04480053671332476], ...]
}
```

## Classifying

You can now classify as much as you like using your trained model weights!  You simply post your query `text` along with your trained `weights` and `labels` (that you receive back when training)

```python
classification_url = "https://immense-dusk-95991.herokuapp.com/label"
classification_example = {
    "text":"this is a wolf",
    "weights":model_parameters.get("weights"),
    "labels":model_parameters.get("labels"),
}

response = post(classification_url, json=classification_example)
>> <Response [200]>
```

You will receive back the text labelled according to your desired labels

```python
response.json()
>> {
    "text":"this is a wolf",
    "class_labels":["ANIMAL","NON-ANIMAL"],
    "extracted_tokens":["wolf"],
    "classification":["ANIMAL"]
}
```
