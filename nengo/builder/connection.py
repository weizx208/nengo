import collections

import numpy as np

import nengo.utils.numpy as npext
from nengo.builder.builder import Builder
from nengo.builder.ensemble import gen_eval_points, get_activities
from nengo.builder.node import SimPyFunc
from nengo.builder.operator import DotInc, ElementwiseInc, PreserveValue, Reset
from nengo.builder.signal import Signal
from nengo.builder.synapses import filtered_signal
from nengo.connection import Connection
from nengo.ensemble import Ensemble, Neurons
from nengo.neurons import Direct
from nengo.node import Node
from nengo.utils.builder import full_transform
from nengo.utils.compat import is_iterable


BuiltConnection = collections.namedtuple(
    'BuiltConnection', ['decoders', 'eval_points', 'transform', 'solver_info'])


def get_eval_points(model, conn, rng):
    if conn.eval_points is None:
        return npext.array(
            model.params[conn.pre_obj].eval_points, min_dims=2)
    else:
        return gen_eval_points(
            conn.pre_obj, conn.eval_points, rng, conn.scale_eval_points)


def get_targets(model, conn, eval_points):
    if conn.function is None:
        targets = eval_points[:, conn.pre_slice]
    else:
        targets = np.zeros((len(eval_points), conn.size_mid))
        for i, ep in enumerate(eval_points[:, conn.pre_slice]):
            targets[i] = conn.function(ep)

    return targets


def build_linear_system(model, conn, rng):
    eval_points = get_eval_points(model, conn, rng)
    activities = get_activities(model, conn.pre_obj, eval_points)
    if np.count_nonzero(activities) == 0:
        raise RuntimeError(
            "Building %s: 'activites' matrix is all zero for %s. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (conn, conn.pre_obj))

    targets = get_targets(model, conn, eval_points)
    return eval_points, activities, targets


@Builder.register(Connection)  # noqa: C901
def build_connection(model, conn):
    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    # Get input and output connections from pre and post
    def get_prepost_signal(is_pre):
        target = conn.pre_obj if is_pre else conn.post_obj
        key = 'out' if is_pre else 'in'

        if target not in model.sig:
            raise ValueError("Building %s: the '%s' object %s "
                             "is not in the model, or has a size of zero."
                             % (conn, 'pre' if is_pre else 'post', target))
        if key not in model.sig[target]:
            raise ValueError("Error building %s: the '%s' object %s "
                             "has a '%s' size of zero." %
                             (conn, 'pre' if is_pre else 'post', target, key))

        return model.sig[target][key]

    model.sig[conn]['in'] = get_prepost_signal(is_pre=True)
    model.sig[conn]['out'] = get_prepost_signal(is_pre=False)

    decoders = None
    eval_points = None
    solver_info = None
    transform = full_transform(conn, slice_pre=False)

    # Figure out the signal going across this connection
    if (isinstance(conn.pre_obj, Node) or
            (isinstance(conn.pre_obj, Ensemble) and
             isinstance(conn.pre_obj.neuron_type, Direct))):
        # Node or Decoded connection in directmode
        if (conn.function is None and isinstance(conn.pre_slice, slice) and
                (conn.pre_slice.step is None or conn.pre_slice.step == 1)):
            signal = model.sig[conn]['in'][conn.pre_slice]
        else:
            signal = Signal(np.zeros(conn.size_mid), name='%s.func' % conn)
            fn = ((lambda x: x[conn.pre_slice]) if conn.function is None else
                  (lambda x: conn.function(x[conn.pre_slice])))
            model.add_op(SimPyFunc(
                output=signal, fn=fn, t_in=False, x=model.sig[conn]['in']))
    elif isinstance(conn.pre_obj, Ensemble):
        # Normal decoded connection
        eval_points, activities, targets = build_linear_system(
            model, conn, rng)

        # Use cached solver, if configured
        solver = model.decoder_cache.wrap_solver(conn.solver)

        if conn.solver.weights:
            # include transform in solved weights
            targets = np.dot(targets, transform.T)
            transform = np.array(1., dtype=np.float64)

            decoders, solver_info = solver(
                activities, targets, rng=rng,
                E=model.params[conn.post_obj].scaled_encoders.T)
            model.sig[conn]['out'] = model.sig[conn.post_obj.neurons]['in']
            signal_size = model.sig[conn]['out'].size
        else:
            decoders, solver_info = solver(activities, targets, rng=rng)
            signal_size = conn.size_mid

        # Add operator for decoders
        decoders = decoders.T

        model.sig[conn]['decoders'] = Signal(
            decoders, name="%s.decoders" % conn)
        signal = Signal(np.zeros(signal_size), name=str(conn))
        model.add_op(Reset(signal))
        model.add_op(DotInc(model.sig[conn]['decoders'],
                            model.sig[conn]['in'],
                            signal,
                            tag="%s decoding" % conn))
    else:
        # Direct connection
        signal = model.sig[conn]['in']

    # Add operator for filtering
    if conn.synapse is not None:
        signal = filtered_signal(model, conn, signal, conn.synapse)

    # Add operator for transform
    if isinstance(conn.post_obj, Neurons):
        if not model.has_built(conn.post_obj.ensemble):
            # Since it hasn't been built, it wasn't added to the Network,
            # which is most likely because the Neurons weren't associated
            # with an Ensemble.
            raise RuntimeError("Connection '%s' refers to Neurons '%s' "
                               "that are not a part of any Ensemble." % (
                                   conn, conn.post_obj))

        if conn.post_slice != slice(None):
            raise NotImplementedError(
                "Post-slices on connections to neurons are not implemented")

        gain = model.params[conn.post_obj.ensemble].gain[conn.post_slice]
        if transform.ndim < 2:
            transform = transform * gain
        else:
            transform *= gain[:, np.newaxis]

    model.sig[conn]['transform'] = Signal(transform,
                                          name="%s.transform" % conn)
    if transform.ndim < 2:
        model.add_op(ElementwiseInc(model.sig[conn]['transform'],
                                    signal,
                                    model.sig[conn]['out'],
                                    tag=str(conn)))
    else:
        model.add_op(DotInc(model.sig[conn]['transform'],
                            signal,
                            model.sig[conn]['out'],
                            tag=str(conn)))

    # Build learning rules
    if conn.learning_rule:
        if isinstance(conn.pre_obj, Ensemble):
            model.add_op(PreserveValue(model.sig[conn]['decoders']))
        else:
            model.add_op(PreserveValue(model.sig[conn]['transform']))

        if isinstance(conn.pre_obj, Ensemble) and conn.solver.weights:
            # TODO: make less hacky.
            # Have to do this because when a weight_solver
            # is provided, then learning rules should operate on
            # "decoders" which is really the weight matrix.
            model.sig[conn]['transform'] = model.sig[conn]['decoders']

        rule = conn.learning_rule
        if is_iterable(rule):
            if isinstance(rule, dict):
                # sort keys to ensure consistent ordering
                rule = (rule[k] for k in sorted(rule))
            for r in rule:
                model.build(r)
        elif rule is not None:
            model.build(rule)

    model.params[conn] = BuiltConnection(decoders=decoders,
                                         eval_points=eval_points,
                                         transform=transform,
                                         solver_info=solver_info)
