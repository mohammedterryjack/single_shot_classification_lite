from typing import Dict, List, Tuple
from json import loads 

from numpy import array,zeros

class PoincareEmbeddings:
    def __init__(self,path_to_file:str) -> None:
        self.vector_length = 100
        with open(path_to_file) as vector_file:
            self.word_vectors = loads(vector_file.read())
    
    def embed_word(self, word:str) -> List[float]:
        return self.word_vectors.get(word,zeros(self.vector_length))

    def encode(self, text:str) -> array:
        words = text.lower().split()
        vectors = list(map(self.embed_word,words))
        return array(vectors).mean(axis=0)