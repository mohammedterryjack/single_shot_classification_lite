from typing import Optional, List, Tuple
from json import dumps

from numpy import argmax, array, zeros

from src.extreme_learning_machine import ExtremeLearningMachine
from src.poincare_encoder import PoincareEmbeddings

class UniversalClassifier:
    def __init__(self) -> None:
        self.encoder = PoincareEmbeddings("src/poincare_embeddings.json")
        self.decoder = ExtremeLearningMachine(input_size=self.encoder.vector_length)
        self.update_ontology([])
    
    def predict(self, text:str, pretrained_weights:List[List[float]], custom_labels:List[str]) -> str:
        self.update_ontology(custom_labels)
        self.decoder.upload_weights(array(pretrained_weights))
        labelled_tokens = list(filter(lambda token_label:token_label[-1], self.predict_labels(text)))
        tokens,labels = [],[]
        if any(labelled_tokens):
            tokens,labels = zip(*labelled_tokens)
        return dumps({"text":text,"class_labels":custom_labels,"extracted_tokens":tokens,"classification":labels},indent=3)

    def predict_labels(self, text:str) -> List[Tuple[str,Optional[str]]]:
        tokens = text.split()
        for token_index,token in enumerate(tokens):
            context = ' '.join(tokens[:token_index])
            label = self.predict_label(token,context)
            yield (token,label)

    def predict_label(self, text:str, context:Optional[str]=None) -> Optional[str]:
        input_vector = self.encode_text(f"{context} {text}")
        output_logits = self.decoder.infer(input_vector)
        class_index = output_logits.argmax()
        return self.class_index_to_label_mapping.get(class_index)
    
    def finetune(self, custom_labels:List[str], train_data:List[List[Tuple[str,str]]]) -> str:
        x,y = zip(*self.encode_data(custom_labels,train_data))
        self.decoder.fit(x,y)
        return dumps(
            {
                "labels":list(self.label_to_class_index_mapping.keys()),
                "weights":self.decoder.download_weights().tolist(),
            }
        )

    def encode_data(self, labels:List[str],labelled_examples:List[List[Tuple[str,str]]]) -> List[Tuple[array,array]]:
        self.update_ontology(labels)

        for labelled_example in labelled_examples:
            for current_index,labelled_token in enumerate(labelled_example):
                token,desired_label = labelled_token
                context = ' '.join(
                    map(
                        lambda token_label:token_label[0],
                        labelled_example[:current_index]
                    )
                )
                input_vector = self.encode_text(text=f"{context} {token}")
                output_vector = self.encode_label(label=desired_label)
                yield (input_vector,output_vector)

    def encode_text(self, text:str) -> array:
        return self.encoder.encode(text)

    def encode_label(self, label:Optional[str]) -> array:
        class_index = 0
        if label is not None:
            class_index = self.label_to_class_index_mapping.get(label,0)
        return self.onehot(index=class_index,length=self.output_width)

    def update_ontology(self, labels:List[str]) -> None:
        self.class_index_to_label_mapping=dict(enumerate(labels,1))
        self.label_to_class_index_mapping = {label:index for index,label in self.class_index_to_label_mapping.items()}
        self.output_width = len(self.label_to_class_index_mapping)+1

    @staticmethod
    def onehot(index:int, length:int) -> array:
        vector = zeros(length)
        vector[index] = 1.
        return vector