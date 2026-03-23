from core.tensor import Tensor
from core.activations import Relu
from core.layers import Linear

import numpy as np

EPSILON = 1e-7

def log_softmax(x, dim = -1):
    x_max = np.max(x.data,axis=dim,keepdims=True)
    shifted_x = x.data - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted_x),axis=dim,keepdims=True))
    result = shifted_x - log_sum_exp
    return Tensor(result)

class MSELoss:

    def __init__(self):
        pass

    def forward(self,predictions,targets):
        diff = predictions.data - targets.data
        sqr_diff = diff**2
        mean_diff = np.mean(sqr_diff)
        return Tensor(mean_diff)
    
    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)
    
    def backward(self):
        pass

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self,logits,targets):
        log_softmax_logits = log_softmax(logits)
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)
        selected_log_probs = log_softmax_logits.data[np.arange(batch_size), target_indices]
        cross_entropy = -np.mean(selected_log_probs)
        return Tensor(cross_entropy)

    def __call__(self, logits, targets):
        return self.forward(logits,targets)
    
    def backward(self):
        pass

class BinaryCrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self,predictions,targets):
        eps = EPSILON
        clamped_preds = np.clip(predictions.data, eps, 1 - eps)
        log_preds = np.log(clamped_preds)
        log_one_minus_preds = np.log(1 - clamped_preds)
        bce_per_sample = -(targets.data * log_preds + (1 - targets.data) * log_one_minus_preds)
        bce_loss = np.mean(bce_per_sample)

        return Tensor(bce_loss)

    def __call__(self, predictions, targets):
        return self.forward(predictions,targets)
    
    def backward(self):
        pass
