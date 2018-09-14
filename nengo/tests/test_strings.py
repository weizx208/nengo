import collections
import inspect

import numpy as np
import pytest

import nengo
from nengo.utils.compat import is_array


def safe_equal(a, b):
    if is_array(a) or is_array(b):
        return np.array_equal(a, b)
    else:
        return a == b


def get_all_args(nengo_class, stop_classes=()):
    all_args = []
    all_kwargs = collections.OrderedDict()
    for k, cls in enumerate(nengo_class.__mro__):
        if cls is stop_classes:
            break

        argspec = inspect.getfullargspec(cls.__init__)
        n_kwargs = len(argspec.defaults) if argspec.defaults else 0
        args = argspec.args[:-n_kwargs] if n_kwargs > 0 else argspec.args
        kwargs = (zip(argspec.args[-n_kwargs:], argspec.defaults)
                  if n_kwargs > 0 else [])
        if len(args) > 0 and args[0] == 'self':
            args.pop(0)

        if k == 0:  # only add args from this class itself
            all_args.extend(arg for arg in args if arg not in all_args)
        all_kwargs.update(pair for pair in kwargs if pair[0] not in all_kwargs)

    return all_args, all_kwargs


sampled_params = {
    nengo.synapses.LinearFilter: dict(
        num=[(1, 2)], den=[(3, 4)]),
    nengo.synapses.Alpha: dict(analog=[True]),
    nengo.synapses.Lowpass: dict(analog=[True]),
    nengo.dists.PDF: dict(x=[[1, 2, 3]],
                          p=[[0.25, 0.5, 0.25]]),
    nengo.dists.UniformHypersphere: dict(surface=[False]),
    nengo.dists.Choice: dict(options=[[1, 2, 3]],
                             weights=[[1, 3, 2]]),
    nengo.dists.Samples: dict(samples=[[1, 2, 3, 8]]),
}


def _sample_param_basic(param, default=None, rng=None):
    if isinstance(param, nengo.params.IntParam):
        low = param.low if param.low is not None else -2**15
        high = param.high if param.high is not None else 2**15
        return rng.randint(low + (1 if param.low_open else 0),
                           high + (0 if param.high_open else 1))
    if isinstance(param, nengo.params.NumberParam):
        low = param.low if param.low is not None else -1e3
        high = param.high if param.high is not None else 1e3
        return rng.uniform(low, high)
    if isinstance(param, nengo.params.EnumParam):
        values = list(param.values)
        if default in values:
            values.remove(default)
        return values[rng.randint(len(values))]
    if isinstance(param, nengo.params.BoolParam):
        return bool(rng.randint(2)) if default is None else (not default)

    raise NotImplementedError("Cannot sample %s" % param)


def sample_param(nengo_class, param_name, default=None, rng=None):
    if nengo_class in sampled_params:
        if param_name in sampled_params[nengo_class]:
            choices = sampled_params[nengo_class][param_name]
            return choices[rng.randint(len(choices))]

    param = getattr(nengo_class, param_name)
    for _ in range(10):
        value = _sample_param_basic(param, default, rng=rng)
        if default is None or not safe_equal(value, default):
            return value

    raise RuntimeError("Could not find valid sample value")


def check_repr(obj, args, kwargs, argnames=True, always_kwargs=()):
    argreprs = []
    for arg in args:
        v = getattr(obj, arg)
        argreprs.append("%s=%r" % (arg, v) if argnames else str(v))
    for arg, default in kwargs.items():
        v = getattr(obj, arg)
        if arg in always_kwargs or not safe_equal(v, default):
            argreprs.append("%s=%r" % (arg, v))

    ref_repr = "%s(%s)" % (type(obj).__name__, ", ".join(argreprs))
    assert repr(obj) == ref_repr, "ref: %s, obj: %r" % (ref_repr, obj)


def check_single_repr(obj, args_stop_classes=(), **check_repr_args):
    args, kwargs = get_all_args(type(obj), stop_classes=args_stop_classes)
    check_repr(obj, args, kwargs)


def check_nengo_type_reprs(nengo_class, arg_stop_classes=(), rng=None,
                           **check_repr_args):
    # get all constructor arguments
    args, kwargs = get_all_args(nengo_class, stop_classes=arg_stop_classes)

    # sample all values
    arg_values = [sample_param(nengo_class, arg, rng=rng)
                  for arg in args]
    kwarg_values = {arg: sample_param(nengo_class, arg, default, rng=rng)
                    for arg, default in kwargs.items()}

    # check with basic constructor
    obj = nengo_class(*arg_values)
    check_repr(obj, args, kwargs, **check_repr_args)

    # check with single keyword arg
    for arg in kwargs:
        obj = nengo_class(*arg_values, **{arg: kwarg_values[arg]})
        check_repr(obj, args, kwargs, **check_repr_args)

    # check with all keyword args
    obj = nengo_class(*arg_values,
                      **{arg: kwarg_values[arg] for arg in kwargs})
    check_repr(obj, args, kwargs, **check_repr_args)


@pytest.mark.parametrize('cls', [
    nengo.Direct,
    nengo.RectifiedLinear,
    nengo.SpikingRectifiedLinear,
    nengo.Sigmoid,
    nengo.LIFRate,
    nengo.LIF,
    nengo.AdaptiveLIFRate,
    nengo.AdaptiveLIF,
    nengo.Izhikevich,
])
def test_neuron_argreprs(cls, rng):
    """Test repr() for each neuron type."""
    check_nengo_type_reprs(cls, arg_stop_classes=(
        nengo.neurons.NeuronType), rng=rng)


@pytest.mark.parametrize('cls', [
    nengo.synapses.LinearFilter,
    nengo.synapses.Lowpass,
    nengo.synapses.Alpha,
    nengo.synapses.Triangle,
])
def test_synapse_argreprs(cls, rng):
    check_nengo_type_reprs(
        cls, arg_stop_classes=(nengo.synapses.Synapse), rng=rng,
        argnames=False)


@pytest.mark.parametrize('cls', [
    nengo.dists.PDF,
    nengo.dists.Uniform,
    nengo.dists.Gaussian,
    nengo.dists.Exponential,
    nengo.dists.UniformHypersphere,
    nengo.dists.Choice,
    nengo.dists.Samples,
    nengo.dists.SqrtBeta,
    # nengo.dists.SubvectorLength,
    # nengo.dists.CosineSimilarity,
])
def test_dist_argreprs(cls, rng):
    check_nengo_type_reprs(
        cls, arg_stop_classes=(nengo.dists.Distribution), rng=rng,
        argnames=True)


def test_uniformhypersphere_argreprs():
    # check manually due to warning if surface=True and min_magnitude set
    check_single_repr(nengo.dists.UniformHypersphere(surface=True),
                      args_stop_classes=(nengo.dists.Distribution,))
