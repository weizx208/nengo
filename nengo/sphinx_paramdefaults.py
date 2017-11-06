import inspect
import warnings

from nengo.params import Default, iter_params


class DisplayDefault(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "Default<{!r}>".format(self.value)


def resolve_default(cls, arg, value):
    if value is not Default:
        return value
    else:
        for param in (getattr(cls, name) for name in iter_params(cls)):
            if param.name == arg:
                return DisplayDefault(param.default)
        else:
            warnings.warn(
                "Default value for argument {} of {} could not be "
                "resolved.".format(arg, cls))
            return value


def autodoc_defaults(
        app, what, name, obj, options, signature, return_annotation):
    if what != 'class':
        return signature, return_annotation
    args, varargs, keywords, defaults = inspect.getargspec(obj.__init__)
    if defaults is not None:
        defaults = [
            resolve_default(obj, arg, d)
            for arg, d in zip(args[-len(defaults):], defaults)]
    return (
        inspect.formatargspec(args, varargs, keywords, defaults),
        return_annotation)


def setup(app):
    app.connect('autodoc-process-signature', autodoc_defaults)
