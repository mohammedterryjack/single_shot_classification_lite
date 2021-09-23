# Single-Shot Classification (Lite)

Same as other repo but lite weight version - using Poincare embeddings and XLM

## One-Shot Training

first train the model by showing it a few examples of how you want your data to be labelled.  You can do one-shot labels or sequences. The data should be formatted like so:

```python
training_examples = {
    "labels":["FELINE","CANINE"],
    "labelled_sequences":[
        [["this",None],["is",None],["a",None],["cat","FELINE"]],
        [["tiger","FELINE"]],
        [["dog","CANINE"]],
        [["wolf","CANINE"]]
    ]
}

response = post("https://immense-dusk-95991.herokuapp.com/train", json=train_data)
```
You will receive a json back with the trained weights and the labels you provided. Simply keep these weights and labels and you do not need to train it again (unless you wish to change the task).  

```python
model_parameters = response.json()
>> {
    "labels":["CANINE","FELINE"],
    "weights":[...]
}
```

## Classifying

You can now classify as much as you like using your trained model weights!  You simply post your query `text` along with your trained `weights` and `labels` (that you receive back when training)

```python
example_query = {
    "text":"this is a wolf",
    "weights":model_parameters.get("weights"),
    "labels":model_parameters.get("labels"),
}

response = post("https://immense-dusk-95991.herokuapp.com/label", json=example_query)
```

You will receive back the text labelled according to your desired labels

```python
model_parameters = response.json()
>> {
    "text":"this is a wolf",
    "class_labels":["CANINE","FELINE"],
    "extracted_tokens":["wolf"],
    "classification":["CANINE"]
}
```