import mxnet as mx

__all__ = []


def _register_gc_opt():
    optimizers = dict()
    for name in dir(mx.optimizer):
        obj = getattr(mx.optimizer, name)
        if hasattr(obj, '__base__') and obj.__base__ == mx.optimizer.Optimizer:
            optimizers[name] = obj
    suffix = 'GC'

    def __init__(self, gc_type='gc', **kwargs):
        assert gc_type.lower() in ['gc', 'gcc']
        self.gc_ndim_thr = 1 if gc_type.lower() == 'gc' else 3
        self._parent_cls = super(self.__class__, self)
        self._parent_cls.__init__(**kwargs)

    def update(self, index, weight, grad, state):
        self._gc_update_impl(
            index, weight, grad, state, self._parent_cls.update)

    def update_multi_precision(self, index, weight, grad, state):
        self._gc_update_impl(
            index, weight, grad, state, self._parent_cls.update_multi_precision)

    def _gc_update_impl(self, indexes, weights, grads, states, update_func):
        # centralize gradients
        if isinstance(indexes, (list, tuple)):
            # multi index case: SGD optimizer
            for grad in grads:
                if len(grad.shape) > self.gc_ndim_thr:
                    grad -= grad.mean(axis=tuple(range(1, len(grad.shape))), keepdims=True)
        else:
            # single index case: all other optimizers
            if len(grads.shape) > self.gc_ndim_thr:
                grads -= grads.mean(axis=tuple(range(1, len(grads.shape))), keepdims=True)
        # update weights using centralized gradients
        update_func(indexes, weights, grads, states)

    inst_dict = dict(
        __init__=__init__,
        update=update,
        update_multi_precision=update_multi_precision,
        _gc_update_impl=_gc_update_impl,
    )

    for k, v in optimizers.items():
        name = k + suffix
        inst = type(name, (v, ), inst_dict)
        mx.optimizer.Optimizer.register(inst)
        globals()[name] = inst
        __all__.append(name)


_register_gc_opt()
