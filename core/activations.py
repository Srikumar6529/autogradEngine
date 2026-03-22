import numpy as np
from core.tensor import Tensor
class Sigmoid:
    def parameters(self):
        return []
    def forward(self,x):
        expx = np.exp(-x.data)
        result = 1/(1+expx)
        return Tensor(result)
    def __call__(self, x):
        return self.forward(x)
    
class Relu:
    def parameters(self):
        return []
    
    def forward(self,x):
        result = np.maximum(0,x.data)
        return Tensor(result)
    def __call__(self, x):
        return self.forward(x)
    
class Tanh:
    def parameters(self):
        return []
    
    def forward(self,x):
        result = np.tanh(x.data)
        return Tensor(result)
    
    def __call__(self,x):
        return self.forward(x)
    
class Gelu:
    def parameters(self):
        return []
    
    def forward(self,x):
        sigmoid_part = 1.0 / (1.0 + np.exp(-1.702 * x.data))
        result = x.data * sigmoid_part
        return Tensor(result)

    def __call__(self,x):
        return self.forward(x)
    
class Softmax:
    def parameters(self):
        return []
    
    def forward(self,x,dim = -1):
        x_max = np.max(x.data, axis=dim, keepdims=True)
        x_shifted = x.data - x_max
        exp_values = np.exp(x_shifted)
        exp_sum = np.sum(exp_values, axis=dim, keepdims=True)
        result = exp_values / exp_sum
        return Tensor(result)
    
    def __call__(self, x,dim = -1):
        return self.forward(x,dim)
