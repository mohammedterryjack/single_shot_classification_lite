from typing import Dict, List, Tuple
from json import loads 
from math import sin,cos

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
        word_vectors = array(list(map(self.embed_word,words)))
        position_vectors = array(list(map(self.embed_position,range(len(words)))))
        vectors = word_vectors + position_vectors
        return vectors.mean(axis=0)
    
    def embed_position(self, position:int) -> List[float]:
        constant=1e5
        position_vector = [0.]*self.vector_length
        for index in range(0,self.vector_length,2):
            double_index = 2*index
            double_index_normalised = double_index/self.vector_length
            denominator = constant ** double_index_normalised
            position_vector[index]=sin(position/denominator)
            position_vector[index+1]=cos(position/denominator)
        return position_vector