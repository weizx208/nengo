import numpy as np

import nengo
from nengo.base import FrozenObject
from nengo.dists import Distribution, DistOrArrayParam
from nengo.exceptions import ValidationError
from nengo.params import Parameter, ShapeParam, IntParam, EnumParam, BoolParam
from nengo.utils.compat import is_array_like


class Transform(FrozenObject):
    """A base class for connection transforms."""

    def sample(self, rng=np.random):
        """
        Returns concrete weights needed to implement the specific transform.

        Parameters
        ----------
        rng : `numpy.random.RandomState`, optional
            Random number generator state.

        Returns
        -------
        array_like
            Transform weights
        """
        raise NotImplementedError()

    @property
    def size_in(self):
        """Expected size of input to transform"""
        raise NotImplementedError()

    @property
    def size_out(self):
        """Expected size of output from transform"""
        raise NotImplementedError()


class TransformParam(Parameter):
    """A parameter where the value must be a Transform.

    Also supports casting array_likes to Dense transforms.
    """

    coerce_defaults = False

    def __init__(self, name, default, optional=False, readonly=False):
        super(TransformParam, self).__init__(
            name, default, optional, readonly)

    def coerce(self, conn, transform):
        if not isinstance(transform, Transform):
            transform = Dense((conn.size_out, conn.size_mid), transform)

        if transform.size_in != conn.size_mid:
            if isinstance(transform, Dense) and transform.ndim < 2:
                # we provide a different error message in this case;
                # the transform is not changing the dimensionality of the
                # signal, so the blame most likely lies with the function
                raise ValidationError(
                    "function output size is incorrect; should return a "
                    "vector of size %d" % conn.size_out, attr=self.name,
                    obj=conn)

            raise ValidationError(
                "%s output size (%d) not equal to transform input size "
                "(%d)" % (type(conn.pre_obj).__name__, conn.size_mid,
                          transform.size_in), attr=self.name, obj=conn)

        if transform.size_out != conn.size_out:
            raise ValidationError(
                "Transform output size (%d) does not match connection "
                "output size (%d)" % (transform.size_out, conn.size_out),
                "transform")

        # we don't support repeated indices on 2D transforms because it makes
        # the matrix multiplication more complicated (we'd need to expand
        # the weight matrix for the duplicated rows/columns). it could be done
        # if there were a demand at some point.
        if isinstance(transform, Dense) and len(transform.init_shape) == 2:
            def repeated_inds(x):
                return (not isinstance(x, slice) and
                        np.unique(x).size != len(x))

            if repeated_inds(conn.pre_slice):
                raise ValidationError(
                    "Input object selection has repeated indices",
                    attr=self.name, obj=conn)
            if repeated_inds(conn.post_slice):
                raise ValidationError(
                    "Output object selection has repeated indices",
                    attr=self.name, obj=conn)

        return super(TransformParam, self).coerce(conn, transform)


class ChannelShapeParam(ShapeParam):
    """A parameter where the value must be a shape with channels."""

    def coerce(self, transform, shape):
        if isinstance(shape, ChannelShape):
            if shape.channels_last != transform.channels_last:
                raise ValidationError(
                    "transform has channels_last=%s, but input shape has "
                    "channels_last=%s" % (transform.channels_last,
                                          shape.channels_last),
                    attr=self.name, obj=transform)
            super(ChannelShapeParam, self).coerce(transform, shape.shape)
        else:
            super(ChannelShapeParam, self).coerce(transform, shape)
            shape = ChannelShape(shape, channels_last=transform.channels_last)
        return shape


class Dense(Transform):
    """
    A dense transformation between an input and output signal.

    Parameters
    ----------
    shape : tuple of int
        The shape of the dense matrix: ``(size_out, size_in)``.
    init : :class:`.Distribution` or array_like, optional
        A Distribution used to initialize the transform matrix, or a concrete
        instantiation for the matrix.  If the matrix is square we also allow a
        scalar (equivalent to `np.eye(n) * init`) or a vector (equivalent to
        `np.diag(init)`) to represent the matrix more compactly.
    """

    shape = ShapeParam("shape", length=2, low=1)
    init = DistOrArrayParam("init")

    def __init__(self, shape, init=1.0):
        super(Dense, self).__init__()

        self.shape = shape

        if is_array_like(init):
            init = np.asarray(init)

            # check that the shape of init is compatible with the given shape
            # for this transform
            expected_shape = None
            if shape[0] != shape[1]:
                # init must be 2D if transform is not square
                expected_shape = shape
            elif init.ndim == 1:
                expected_shape = (shape[0],)
            elif init.ndim >= 2:
                expected_shape = shape

            if expected_shape is not None and init.shape != expected_shape:
                raise ValidationError(
                    "Shape of initial value %s does not match expected "
                    "shape %s" % (init.shape, expected_shape), "init")

        self.init = init

    def sample(self, rng=np.random):
        if isinstance(self.init, Distribution):
            return self.init.sample(*self.shape, rng=rng)

        return self.init

    @property
    def init_shape(self):
        """The shape of the initial value."""
        return (self.shape if isinstance(self.init, Distribution) else
                self.init.shape)

    @property
    def size_in(self):
        return self.shape[1]

    @property
    def size_out(self):
        return self.shape[0]


class Convolution(Transform):
    """
    An N-dimensional convolutional transform.

    The dimensionality of the convolution is determined by the input shape.

    Parameters
    ----------
    n_filters : int
        The number of convolutional filters to apply
    input_shape : tuple of int or :class:`.ConvShape`
        Shape of the input signal to the convolution; e.g.,
        ``(height, width, channels)`` for a 2D convolution with
        ``channels_last=True``.
    kernel_size : tuple of int, optional
        Size of the convolutional kernels (1 element for a 1D convolution,
        2 for a 2D convolution, etc.).
    strides : tuple of int, optional
        Stride of the convolution (1 element for a 1D convolution, 2 for
        a 2D convolution, etc.).
    padding : ``"same"`` or ``"valid"``, optional
        Padding method for input signal.  "Valid" means no padding, and
        convolution will only be applied to the fully-overlapping areas of the
        input signal (meaning the output will be smaller).  "Same" means that
        the input signal is zero-padded so that the output is the same shape
        as the input.
    channels_last : bool, optional
        If ``True`` (default), the channels are the last dimension in the input
        signal (e.g., a 28x28 image with 3 channels would have shape
        ``(28, 28, 3)``).  ``False`` means that channels are the first
        dimension (e.g., ``(3, 28, 28)``).
    init : :class:`.Distribution` or :class:`~numpy:numpy.ndarray`
        A predefined kernel, with shape
        ``kernel_size + (input_channels, n_filters)``, or a ``Distribution``
        that will be used to initialize the kernel.

    Notes
    -----
    As is typical in neural networks, this is technically correlation rather
    than convolution (because the kernel is not flipped).
    """

    n_filters = IntParam("n_filters", low=1)
    input_shape = ChannelShapeParam("input_shape", low=1)
    kernel_size = ShapeParam("kernel_size", low=1)
    strides = ShapeParam("strides", low=1)
    padding = EnumParam("padding", values=("same", "valid"))
    channels_last = BoolParam("channels_last")
    init = DistOrArrayParam("init")

    def __init__(self, n_filters, input_shape, kernel_size=(3, 3),
                 strides=(1, 1), padding="valid", channels_last=True,
                 init=nengo.dists.Uniform(-1, 1)):
        super(Convolution, self).__init__()

        self.n_filters = n_filters
        self.channels_last = channels_last  # must be set before input_shape
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.init = init

        if len(kernel_size) != self.dimensions:
            raise ValidationError(
                "Kernel dimensions (%d) do not match input dimensions (%d)" % (
                    len(kernel_size), self.dimensions), "kernel_size")
        if len(strides) != self.dimensions:
            raise ValidationError(
                "Stride dimensions (%d) do not match input dimensions (%d)" % (
                    len(strides), self.dimensions), "strides")
        if not isinstance(init, Distribution):
            if init.shape != self.kernel_shape:
                raise ValidationError(
                    "Kernel shape %s does not match expected shape %s" % (
                        init.shape, self.kernel_shape), "init")

    def sample(self, rng=np.random):
        if isinstance(self.init, Distribution):
            kernel = []
            # we sample this way so that any variancescaling distribution based
            # on n/d is scaled appropriately
            for _ in range(np.prod(self.kernel_size)):
                kernel.append(self.init.sample(self.input_shape.n_channels,
                                               self.n_filters, rng=rng))
            kernel = np.reshape(kernel, self.kernel_shape)
        else:
            kernel = np.array(self.init)
        return kernel

    @property
    def kernel_shape(self):
        """Full shape of kernel."""
        return self.kernel_size + (self.input_shape.n_channels, self.n_filters)

    @property
    def size_in(self):
        return self.input_shape.size

    @property
    def size_out(self):
        return self.output_shape.size

    @property
    def dimensions(self):
        """Dimensionality of convolution."""
        return self.input_shape.dimensions

    @property
    def output_shape(self):
        """Output shape after applying convolution to input."""
        output_shape = np.array(self.input_shape.spatial_shape,
                                dtype=np.float64)
        if self.padding == "valid":
            output_shape -= self.kernel_size
            output_shape += 1
        output_shape /= self.strides
        output_shape = tuple(np.ceil(output_shape).astype(np.int64))
        output_shape = (output_shape + (self.n_filters,) if self.channels_last
                        else (self.n_filters,) + output_shape)

        return ChannelShape(output_shape, channels_last=self.channels_last)


class ChannelShape(object):
    """
    Utility for representing shape information with variable channel position.

    Parameters
    ----------
    shape : tuple of int
        Signal shape
    channels_last : bool, optional
        If True (default), the last item in ``shape`` represents the channels,
        and the rest are spatial dimensions.  Otherwise, the first item in
        ``shape`` is the channel dimension.
    """

    def __init__(self, shape, channels_last=True):
        self.shape = tuple(shape)
        self.channels_last = channels_last

    def __str__(self):
        return "%s(shape=%s, ch_last=%d)" % (
            type(self).__name__, self.shape, self.channels_last)

    @property
    def spatial_shape(self):
        """The spatial part of the shape (omitting channels)."""
        if self.channels_last:
            return self.shape[:-1]
        return self.shape[1:]

    @property
    def size(self):
        """The total number of elements in the represented signal."""
        return np.prod(self.shape)

    @property
    def n_channels(self):
        """The number of channels in the represented signal."""
        return self.shape[-1 if self.channels_last else 0]

    @property
    def dimensions(self):
        """The spatial dimensionality of the represented signal."""
        return len(self.shape) - 1
