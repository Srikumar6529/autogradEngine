import numpy as np
from core.tensor import Tensor
INIT_SCALE_FACTOR = 1.0
DROPOUT_MIN_PROB = 0.0  
DROPOUT_MAX_PROB = 1.0 

class Layer:
    def parameters(self):
        return []
    
    def forward(self,x):
        raise NotImplementedError(
            f"forward() not implemented in {self.__class__.__name__}\n"
            f"  ❌ The Layer base class requires subclasses to implement forward()\n"
            f"  💡 forward() defines how input data is transformed by this layer\n"
            f"  🔧 Add this method to your class:\n"
            f"     def forward(self, x):\n"
            f"         # Your transformation logic here\n"
            f"         return transformed_x"
        )
    
    def __call__(self,x, *args, **kwds):
        return self.forward(x, *args, **kwds)
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
class Linear(Layer):
    def __init__(self,in_features,out_features,bias=True):

        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(INIT_SCALE_FACTOR / in_features)
        weight_data = np.random.randn(in_features,out_features)*scale
        self.weights = Tensor(weight_data)

        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data)
        else:
            self.bias = None

    def forward(self, x):
        output = x.matmul(self.weights)

        if self.bias is not None:
            output = output + self.bias

        return output
    
    
    
    def parameters(self):
        params = [self.weights]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self):
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"
    
   
class Dropout(Layer):

    def __init__(self, p=0.5):
   
        if not DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB:
            raise ValueError(
                f"Invalid dropout probability: {p}\n"
                f"  ❌ p must be between {DROPOUT_MIN_PROB} and {DROPOUT_MAX_PROB}\n"
                f"  💡 p is the probability of DROPPING a neuron (not keeping it!)\n"
                f"     p=0.0 means keep all neurons (no dropout)\n"
                f"     p=0.5 means drop 50% of neurons randomly\n"
                f"     p=1.0 means drop all neurons (zero output)\n"
                f"  🔧 Common values: Dropout(0.1) for light, Dropout(0.3) for moderate, Dropout(0.5) for aggressive"
            )
        self.p = p

    def _should_apply_dropout(self, training):
        return training and self.p > DROPOUT_MIN_PROB

    def _generate_dropout_mask(self, shape):
        keep_prob = 1.0 - self.p
        binary_mask = (np.random.random(shape) < keep_prob).astype(np.float32)
        scale = 1.0 / keep_prob
        return Tensor(binary_mask * scale)

    def forward(self, x, training=True):
        if not self._should_apply_dropout(training):
            return x

        if self.p == DROPOUT_MAX_PROB:
            return Tensor(np.zeros_like(x.data))

        mask = self._generate_dropout_mask(x.data.shape)
        return x * mask
    

    def __call__(self, x, training=True):
        
        return self.forward(x, training)

    def parameters(self):
    
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"
    

class Sequential:
    def __init__(self,*layers):

        if len(layers)==1 and isinstance(layers[0],(list,tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def __repr__(self):
        layer_reprs = ", ".join(repr(layer) for layer in self.layers)
        return f"Sequential({layer_reprs})"

