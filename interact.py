from json import loads 

from src.classifier import UniversalClassifier

classifier = UniversalClassifier()
labels = ["PERSON","ANIMAL"]
train_data = [
    [["lion","ANIMAL"]],
    [["german","ANIMAL"],["shephard","ANIMAL"]],
    [["german","PERSON"],["person","PERSON"]],
    [["shephard","PERSON"]]
]
parameters = loads(classifier.finetune(labels,train_data))

while True:
    predictions = classifier.predict(
        text=input(">"),
        pretrained_weights=parameters.get("weights"),
        custom_labels=parameters.get("labels")
    )
    print(predictions)