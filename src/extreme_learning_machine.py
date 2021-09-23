from numpy import array, exp
from numpy.linalg import pinv
from numpy.random import seed, uniform

class ExtremeLearningMachine:
    def __init__(self, input_size:int, projection_size:int=1000) -> None:
        seed(0) 
        self.random_projection = uniform(low=-.1, high=.1, size=(input_size,projection_size))
        self.sigmoid_activation = lambda x:1. / (1. + exp(-x))

    def infer(self, inputs: array) -> array:
        return self._output_layer(hidden = self._hidden_layer(inputs))

    def _hidden_layer(self, inputs: array) -> array: 
        return self.sigmoid_activation(x=inputs @ self.random_projection)
  
    def _output_layer(self, hidden: array) -> array: 
        return hidden @ self.hidden_layer_to_output_layer_weights    

    def fit(self, inputs: array, outputs: array) -> None:
        self.hidden_layer_to_output_layer_weights = pinv(self._hidden_layer(inputs)) @ outputs
        
    def download_weights(self) -> array:
        return self.hidden_layer_to_output_layer_weights

    def upload_weights(self, weights: array) -> None:
        self.hidden_layer_to_output_layer_weights = weights