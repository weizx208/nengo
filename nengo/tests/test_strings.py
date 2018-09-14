import collections
import inspect

import pytest

import nengo


def get_all_args(nengo_class, stop_classes=()):
    all_args = []
    all_kwargs = collections.OrderedDict()
    for cls in nengo_class.__mro__:
        if cls is stop_classes:
            break

        argspec = inspect.getfullargspec(cls.__init__)
        n_kwargs = len(argspec.defaults) if argspec.defaults else 0
        args = argspec.args[:-n_kwargs] if n_kwargs > 0 else argspec.args
        kwargs = (zip(argspec.args[-n_kwargs:], argspec.defaults)
                  if n_kwargs > 0 else [])
        if len(args) > 0 and args[0] == 'self':
            args.pop(0)

        all_args.extend(arg for arg in args if arg not in all_args)
        all_kwargs.update(pair for pair in kwargs if pair[0] not in all_kwargs)

    return all_args, all_kwargs


def sample_param(param, default=None, rng=None):
    if isinstance(param, nengo.params.IntParam):
        low = param.low if param.low is not None else -2**15
        high = param.high if param.high is not None else 2**15
        return rng.randint(low + (1 if param.low_open else 0),
                           high + (0 if param.high_open else 1))
    if isinstance(param, nengo.params.NumberParam):
        low = param.low if param.low is not None else -1e16
        high = param.high if param.high is not None else 1e16
        return rng.uniform(low, high)
    if isinstance(param, nengo.params.EnumParam):
        values = list(param.values)
        if default in values:
            values.remove(default)
        return values[rng.randint(len(values))]
    raise NotImplementedError("Cannot sample %s" % param)


def check_repr(obj, args, kwargs):
    argreprs = []
    for arg in args:
        argreprs.append("%s=%s" % (arg, getattr(obj, arg)))
    for arg, default in kwargs.items():
        v = getattr(obj, arg)
        if v != default:
            argreprs.append("%s=%s" % (arg, v))

    ref_repr = "%s(%s)" % (type(obj).__name__, ", ".join(argreprs))
    assert repr(obj) == ref_repr


def check_nengo_type_reprs(nengo_class, arg_stop_classes=(), rng=None):
    # get all constructor arguments
    args, kwargs = get_all_args(nengo_class, stop_classes=arg_stop_classes)

    # check with empty constructor
    assert len(args) == 0
    check_repr(nengo_class(), args, kwargs)

    # check with single keyword arg
    values = {}
    for arg, default in kwargs.items():
        for _ in range(10):
            values[arg] = sample_param(
                getattr(nengo_class, arg), default, rng=rng)
            if values[arg] != default:
                break
        else:
            raise RuntimeError("Could not find valid sample value")

        check_repr(nengo_class(**{arg: values[arg]}), args, kwargs)

    # check with all keyword args
    check_repr(nengo_class(**{arg: values[arg] for arg in kwargs}),
               args, kwargs)


@pytest.mark.parametrize('NeuronType', [  # noqa: C901
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
def test_neuron_argreprs(NeuronType, rng):
    """Test repr() for each neuron type."""
    check_nengo_type_reprs(NeuronType, arg_stop_classes=(
        nengo.neurons.NeuronType), rng=rng)
