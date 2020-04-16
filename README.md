# Gradient-Centralization
MXNet implementation of Gradient Centralization: https://arxiv.org/abs/2004.01461

## Usage
Import `optimizer.py`, then add the suffix `GC` after the name of [arbitrary optimizer](http://mxnet.incubator.apache.org/api/python/optimization/optimization.html?highlight=opt#module-mxnet.optimizer).

```python
import optimizer
opt_params = {'learning_rate': 0.001}
sgd_gc = optimizer.SGDGC(gc_type='gc', **opt_params)
sgd_gcc = optimizer.SGDGC(gc_type='gcc', **opt_params)
adam_gc = optimizer.AdamGC(gc_type='gc', **opt_params)
adam_gcc = optimizer.AdamGC(gc_type='gcc', **opt_params)
```

Parameter `gc_type` controls what types of layers will be centralized: `'gc'` applies GC to both conv and fc layers, while `'gcc'` will centralize only conv gradients.

## Example
```bash
python3 mnist.py --optimizer sgdgc --gc-type gc --lr 0.1 --seed 42
python3 mnist.py --optimizer adamgc --gc-type gcc --lr 0.001 --seed 42
```
