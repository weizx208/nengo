import collections

import numpy as np

from nengo.builder import Builder, Signal
from nengo.builder.ensemble import gen_eval_points, get_activities
from nengo.builder.node import SimPyFunc
from nengo.builder.operator import Copy, ElementwiseInc
from nengo.connection import Connection
from nengo.transforms import Convolution
from nengo.ensemble import Ensemble, Neurons
from nengo.exceptions import BuildError, ObsoleteError
from nengo.neurons import Direct
from nengo.node import Node
from nengo.solvers import NoSolver, Solver
from nengo.utils.compat import is_iterable, itervalues

built_attrs = ['eval_points', 'solver_info', 'weights', 'transform']


class BuiltConnection(collections.namedtuple('BuiltConnection', built_attrs)):
    """Collects the parameters generated in `.build_connection`.

    These are stored here because in the majority of cases the equivalent
    attribute in the original connection is a `.Distribution`. The attributes
    of a BuiltConnection are the full NumPy arrays used in the simulation.

    See the `.Connection` documentation for more details on each parameter.

    .. note:: The ``decoders`` attribute is obsolete as of Nengo 2.1.0.
              Use the ``weights`` attribute instead.

    Parameters
    ----------
    eval_points : ndarray
        Evaluation points.
    solver_info : dict
        Information dictionary returned by the `.Solver`.
    weights : ndarray
        Connection weights. May be synaptic connection weights defined in
        the connection's transform, or a combination of the decoders
        automatically solved for and the specified transform.
    transform : ndarray
        The transform matrix.
    """

    __slots__ = ()

    def __new__(cls, eval_points, solver_info, weights, transform):
        # Overridden to suppress the default __new__ docstring
        return tuple.__new__(
            cls, (eval_points, solver_info, weights, transform))

    @property
    def decoders(self):
        raise ObsoleteError("decoders are now part of 'weights'. "
                            "Access BuiltConnection.weights instead.",
                            since="v2.1.0")


def get_eval_points(model, conn, rng):
    if conn.eval_points is None:
        view = model.params[conn.pre_obj].eval_points.view()
        view.setflags(write=False)
        return view
    else:
        return gen_eval_points(
            conn.pre_obj, conn.eval_points, rng, conn.scale_eval_points)


def get_targets(conn, eval_points):
    if conn.function is None:
        targets = eval_points[:, conn.pre_slice]
    elif isinstance(conn.function, np.ndarray):
        targets = conn.function
    else:
        targets = np.zeros((len(eval_points), conn.size_mid))
        for i, ep in enumerate(eval_points[:, conn.pre_slice]):
            out = conn.function(ep)
            if out is None:
                raise BuildError("Building %s: Connection function returned "
                                 "None. Cannot solve for decoders." % (conn,))
            targets[i] = out

    return targets


def build_linear_system(model, conn, rng):
    eval_points = get_eval_points(model, conn, rng)
    ens = conn.pre_obj
    activities = get_activities(model.params[ens], ens, eval_points)
    if np.count_nonzero(activities) == 0:
        raise BuildError(
            "Building %s: 'activites' matrix is all zero for %s. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (conn, conn.pre_obj))

    targets = get_targets(conn, eval_points)
    return eval_points, activities, targets


def build_decoders(model, conn, rng):
    encoders = model.params[conn.pre_obj].encoders
    gain = model.params[conn.pre_obj].gain
    bias = model.params[conn.pre_obj].bias

    eval_points = get_eval_points(model, conn, rng)
    targets = get_targets(conn, eval_points)

    x = np.dot(eval_points, encoders.T / conn.pre_obj.radius)
    wrapped_solver = (model.decoder_cache.wrap_solver(solve_for_decoders)
                      if model.seeded[conn] else solve_for_decoders)
    decoders, solver_info = wrapped_solver(
        conn, gain, bias, x, targets, rng=rng)

    return eval_points, decoders.T, solver_info


def solve_for_decoders(conn, gain, bias, x, targets, rng, E=None):
    activities = conn.pre_obj.neuron_type.rates(x, gain, bias)
    if np.count_nonzero(activities) == 0:
        raise BuildError(
            "Building %s: 'activities' matrix is all zero for %s. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (conn, conn.pre_obj))

    decoders, solver_info = conn.solver(activities, targets, rng=rng, E=E)
    return decoders, solver_info


def multiply(x, y):
    if x.ndim <= 2 and y.ndim < 2:
        return x * y
    elif x.ndim < 2 and y.ndim == 2:
        return x.reshape(-1, 1) * y
    elif x.ndim == 2 and y.ndim == 2:
        return np.dot(x, y)
    else:
        raise BuildError("Tensors not supported (x.ndim = %d, y.ndim = %d)"
                         % (x.ndim, y.ndim))


def slice_signal(model, signal, sl):
    assert signal.ndim == 1
    if isinstance(sl, slice) and (sl.step is None or sl.step == 1):
        return signal[sl]
    else:
        size = np.arange(signal.size)[sl].size
        sliced_signal = Signal(np.zeros(size), name="%s.sliced" % signal.name)
        model.add_op(Copy(signal, sliced_signal, src_slice=sl))
        return sliced_signal


@Builder.register(Solver)
def build_solver(model, solver, conn, rng):
    return build_decoders(model, conn, rng)


@Builder.register(NoSolver)
def build_no_solver(model, solver, conn, rng):
    activities = np.zeros((1, conn.pre_obj.n_neurons))
    targets = np.zeros((1, conn.size_mid))
    # No need to invoke the cache for NoSolver
    decoders, solver_info = conn.solver(activities, targets, rng=rng)
    weights = decoders.T
    return None, weights, solver_info


@Builder.register(Connection)  # noqa: C901
def build_connection(model, conn):
    """Builds a `.Connection` object into a model.

    A brief summary of what happens in the connection build process,
    in order:

    1. Solve for decoders.
    2. Combine transform matrix with decoders to get weights.
    3. Add operators for computing the function
       or multiplying neural activity by weights.
    4. Call build function for the synapse.
    5. Call build function for the learning rule.
    6. Add operator for applying learning rule delta to weights.

    Some of these steps may be altered or omitted depending on the parameters
    of the connection, in particular the pre and post types.

    Parameters
    ----------
    model : Model
        The model to build into.
    conn : Connection
        The connection to build.

    Notes
    -----
    Sets ``model.params[conn]`` to a `.BuiltConnection` instance.
    """

    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    # Get input and output connections from pre and post
    def get_prepost_signal(is_pre):
        target = conn.pre_obj if is_pre else conn.post_obj
        key = 'out' if is_pre else 'in'

        if target not in model.sig:
            raise BuildError("Building %s: the %r object %s is not in the "
                             "model, or has a size of zero."
                             % (conn, 'pre' if is_pre else 'post', target))
        if key not in model.sig[target]:
            raise BuildError(
                "Building %s: the %r object %s has a %r size of zero."
                % (conn, 'pre' if is_pre else 'post', target, key))

        return model.sig[target][key]

    model.sig[conn]['in'] = get_prepost_signal(is_pre=True)
    model.sig[conn]['out'] = get_prepost_signal(is_pre=False)

    decoders = None
    encoders = None
    eval_points = None
    solver_info = None
    post_slice = conn.post_slice

    # Figure out the signal going across this connection
    in_signal = model.sig[conn]['in']
    if (isinstance(conn.pre_obj, Node) or
            (isinstance(conn.pre_obj, Ensemble) and
             isinstance(conn.pre_obj.neuron_type, Direct))):
        # Node or Decoded connection in directmode
        sliced_in = slice_signal(model, in_signal, conn.pre_slice)
        if conn.function is None:
            in_signal = sliced_in
        elif isinstance(conn.function, np.ndarray):
            raise BuildError("Cannot use function points in direct connection")
        else:
            in_signal = Signal(np.zeros(conn.size_mid), name='%s.func' % conn)
            model.add_op(SimPyFunc(in_signal, conn.function, None, sliced_in))
    elif isinstance(conn.pre_obj, Ensemble):  # Normal decoded connection
        eval_points, decoders, solver_info = model.build(
            conn.solver, conn, rng)
        if conn.solver.weights:
            model.sig[conn]['out'] = model.sig[conn.post_obj.neurons]['in']

            if isinstance(conn.post_obj, Ensemble):
                encoders = model.params[conn.post_obj].scaled_encoders.T
                encoders = encoders[conn.post_slice]

                # post slice already applied to encoders, don't apply later
                post_slice = None
    else:
        in_signal = slice_signal(model, in_signal, conn.pre_slice)

    # Build transform
    weighted, weights = model.build(
        conn.transform, in_signal, decoders=decoders, encoders=encoders,
        rng=rng)
    model.sig[conn]["weights"] = weights

    # Build synapse
    if conn.synapse is not None:
        weighted = model.build(conn.synapse, weighted)

    # Store the weighted-filtered output in case we want to probe it
    model.sig[conn]['weighted'] = weighted

    if isinstance(conn.post_obj, Neurons):
        # Apply neuron gains (we don't need to do this if we're connecting to
        # an Ensemble, because the gains are rolled into the encoders)
        gains = Signal(model.params[conn.post_obj.ensemble].gain[post_slice],
                       name="%s.gains" % conn)
        model.add_op(ElementwiseInc(
            gains, weighted, model.sig[conn]['out'][post_slice],
            tag="%s.gains_elementwiseinc" % conn))
    else:
        # Copy to the proper slice
        model.add_op(Copy(
            weighted, model.sig[conn]['out'], dst_slice=post_slice,
            inc=True, tag="%s" % conn))

    # Build learning rules
    if conn.learning_rule is not None:
        # TODO: provide a general way for transforms to expose learnable params
        if isinstance(conn.transform, Convolution):
            raise NotImplementedError(
                "Learning on convolutional connections is not supported")

        rule = conn.learning_rule
        rule = [rule] if not is_iterable(rule) else rule
        targets = []
        for r in itervalues(rule) if isinstance(rule, dict) else rule:
            model.build(r)
            targets.append(r.modifies)

        if 'encoders' in targets:
            encoder_sig = model.sig[conn.post_obj]['encoders']
            encoder_sig.readonly = False
        if 'decoders' in targets or 'weights' in targets:
            if weights.ndim < 2:
                raise BuildError(
                    "'transform' must be a 2-dimensional array for learning")
            model.sig[conn]['weights'].readonly = False

    model.params[conn] = BuiltConnection(eval_points=eval_points,
                                         solver_info=solver_info,
                                         transform=conn.transform,
                                         weights=weights.initial_value)
