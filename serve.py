from os import environ

from flask import Flask,request
from werkzeug.exceptions import BadRequest

from src.classifier import UniversalClassifier
   
app = Flask(__name__)
app.classifier = UniversalClassifier()

@app.route("/")
def home():
    return "Single-Shot Classification"

@app.route(f"/train",methods=["POST"])
def train():
    if request.is_json:
        input_json = request.get_json()
        labels =  input_json.get("labels")
        train_data = input_json.get("labelled_sequences")
        if labels is not None and train_data is not None:
            return app.classifier.finetune(labels,train_data)
    raise BadRequest()
    
@app.route(f"/label",methods=["POST"])
def label():
    if request.is_json:
        input_json = request.get_json()
        text = input_json.get("text")
        weights = input_json.get("weights")
        labels = input_json.get("labels")
        if text is not None and weights is not None and labels is not None:
            return app.classifier.predict(
                text=text,
                pretrained_weights=weights,
                custom_labels=labels
            )
    raise BadRequest()

if __name__ == '__main__':
    app.run(threaded=True, port=environ.get('PORT'), debug=True)