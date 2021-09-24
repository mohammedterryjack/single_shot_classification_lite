# Single-Shot Classification (Lite)

## One-Shot Training

Quickly train your model by showing a few examples of how you want your data to be labelled (You can even do sequences). The data should be formatted like so:

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

response = post(training_url, json=training_examples)
>> <Response [200]>
```
You will receive a json back with your model's trained `weights` and class `labels`. You can save this data (you do not need to train the model again - unless you wish to change the task, etc).  

```python
model_parameters = response.json()
>> {
    "labels":["ANIMAL","NON-ANIMAL"],
    "weights":[ [0.0, -0.042705569538778194, 0.04480053671332476], ...]
}
```

## Classifying

You can now classify as much as you like using your trained model.  Simply post your query `text` along with the trained `weights` and `labels` (that you received back from training)

```python
classification_url = "https://immense-dusk-95991.herokuapp.com/label"
classification_example = {
    "text":"this is a wolf",
    "weights":model_parameters.get("weights"),
    "labels":model_parameters.get("labels"),
}

post(classification_url, json=classification_example).json()
>> {...
    "extracted_tokens":["wolf"],
    "classification":["ANIMAL"]
    }

```

You will receive back the text labelled according to your desired labels


## Tutorial

Here is a demo showing how quick and simple the whole process is (from training to using your one-shot classifier)

![](img/tutorial.gif)

## Model
NB: Same as other repo but lite weight version - using Poincare+Positional embeddings and extreme-Learning-Machine
