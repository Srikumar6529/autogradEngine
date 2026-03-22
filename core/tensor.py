import numpy as np
class Tensor:
    def __init__(self,data):
        self.data = np.array(data,dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

    def __add__(self, other):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        
        result = self.data + other.data
        return Tensor(result)
    
    def __sub__(self, other):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        
        result = self.data - other.data
        return Tensor(result)
    
    def __mul__(self, other):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        
        result = self.data * other.data
        return Tensor(result)
    
    def __truediv__(self, other):
        if not isinstance(other,Tensor):
            other = Tensor(other)
        
        result = self.data / other.data
        return Tensor(result)
    def _validate_matmul_shapes(self, other):
        if not isinstance(other, Tensor):
                raise TypeError(
                    f"Matrix multiplication requires Tensor, got {type(other).__name__}\n"
                    f"  ❌ Cannot perform: Tensor @ {type(other).__name__}\n"
                    f"  💡 Matrix multiplication (@) only works between two Tensors\n"
                    f"  🔧 Wrap your data: Tensor({other}) @ other_tensor"
                )
        if len(self.shape) == 0 or len(other.shape) == 0:
            raise ValueError(
                f"Matrix multiplication requires at least 1D tensors\n"
                f"  ❌ Got shapes: {self.shape} @ {other.shape}\n"
                f"  💡 Scalars (0D tensors) cannot be matrix-multiplied; use * for element-wise\n"
                f"  🔧 Reshape scalar to 1D: tensor.reshape(1) or use tensor * scalar"
            )
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Matrix multiplication shape mismatch: {self.shape} @ {other.shape}\n"
                    f"  ❌ Inner dimensions don't match: {self.shape[-1]} vs {other.shape[-2]}\n"
                    f"  💡 For A @ B, A's last dim must equal B's second-to-last dim\n"
                    f"  🔧 Try: other.transpose() to get shape {other.shape[::-1]}, or reshape self"
                )
    
    def matmul(self,other):
        self._validate_matmul_shapes(other)
        a = self.data
        b = other.data
        if len(self.shape)==2 and len(other.shape)==2:
            M,k1 = self.shape
            k2,N = other.shape
            result = np.zeros((M,N),dtype=self.dtype)
            for i in range(M):
                for j in range(N):
                    result[i,j] = np.dot(a[i,:], b[:,j])
        else:
            result = np.matmul(a,b)
        return Tensor(result)
    def __matmul__(self,other):
        return self.matmul(other)
    
    def __getitem__(self, key):
        result_data = self.data[key]
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)
        return Tensor(result_data)
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                new_shape = tuple(shape[0])
        else:
            new_shape = shape
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError(
                    f"Cannot reshape {self.shape} with multiple unknown dimensions\n"
                    f"  ❌ Found {new_shape.count(-1)} dimensions set to -1 in {new_shape}\n"
                    f"  💡 Only one dimension can be inferred; others must be specified\n"
                    f"  🔧 Replace all but one -1 with explicit sizes (total elements: {self.size})"
                )
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)
        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Cannot reshape {self.shape} to {new_shape}\n"
                f"  ❌ Element count mismatch: {self.size} elements vs {target_size} elements\n"
                f"  💡 Reshape preserves data, so total elements must stay the same\n"
                f"  🔧 Use -1 to infer a dimension: reshape(-1, {new_shape[-1] if len(new_shape) > 0 else 1}) lets NumPy calculate"
            )
        reshaped_data = np.reshape(self.data, new_shape)
        return Tensor(reshaped_data)
    
    def transpose(self, dim0=None, dim1=None):
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                provided = f"dim0={dim0}" if dim1 is None else f"dim1={dim1}"
                missing = "dim1" if dim1 is None else "dim0"
                raise ValueError(
                    f"Transpose requires both dimensions to be specified\n"
                    f"  ❌ Got {provided}, but {missing} is None\n"
                    f"  💡 Either provide both dims or neither (default swaps last two)\n"
                    f"  🔧 Use transpose({dim0 if dim0 is not None else 0}, {dim1 if dim1 is not None else 1}) or just transpose()"
                )
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        return Tensor(transposed_data)
    
    def sum(self, axis=None, keepdims=False):
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
    
    def mean(self, axis=None, keepdims=False):
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
    
    def max(self, axis=None, keepdims=False):
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
    
    @property
    def T(self):
        return self.transpose()