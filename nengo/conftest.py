import hashlib
import inspect
import importlib
import os
import re
from fnmatch import fnmatch
import warnings

import matplotlib
import numpy as np
import pytest

import nengo
import nengo.utils.numpy as npext
from nengo.neurons import (Direct, LIF, LIFRate, RectifiedLinear,
                           Sigmoid, SpikingRectifiedLinear)
from nengo.rc import rc
from nengo.utils.compat import ensure_bytes, is_string
from nengo.utils.testing import Analytics, Logger, Plotter


class TestConfig(object):
    """Parameters affecting all Nengo tests.

    These are essentially global variables used by py.test to modify aspects
    of the Nengo tests. We collect them in this class to provide a
    mini namespace and to avoid using the ``global`` keyword.

    The values below are defaults. The functions in the remainder of this
    module modify these values accordingly.
    """

    test_seed = 0  # changing this will change seeds for all tests
    Simulator = nengo.Simulator
    RefSimulator = nengo.Simulator
    neuron_types = [
        Direct, LIF, LIFRate, RectifiedLinear, Sigmoid, SpikingRectifiedLinear
    ]
    compare_requested = False

    @classmethod
    def is_sim_overridden(cls):
        return cls.Simulator is not nengo.Simulator

    @classmethod
    def is_refsim_overridden(cls):
        return cls.RefSimulator is not nengo.Simulator

    @classmethod
    def is_skipping_frontend_tests(cls):
        return cls.is_sim_overridden() or cls.is_refsim_overridden()


def pytest_configure(config):
    matplotlib.use('agg')
    warnings.simplefilter('always')

    if config.getoption('simulator'):
        TestConfig.Simulator = load_class(config.getoption('simulator')[0])
    if config.getoption('ref_simulator'):
        refsim = config.getoption('ref_simulator')[0]
        TestConfig.RefSimulator = load_class(refsim)

    if config.getoption('neurons'):
        ntypes = config.getoption('neurons')[0].split(',')
        TestConfig.neuron_types = [load_class(n) for n in ntypes]

    if config.getoption('seed_offset'):
        TestConfig.test_seed = config.getoption('seed_offset')[0]

    TestConfig.compare_requested = config.getvalue('compare') is not None


def load_class(fully_qualified_name):
    mod_name, cls_name = fully_qualified_name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


@pytest.fixture(scope="session")
def Simulator(request):
    """The Simulator class being tested.

    Please use this, and not ``nengo.Simulator`` directly. If the test is
    reference simulator specific, then use ``RefSimulator`` below.
    """
    return TestConfig.Simulator


@pytest.fixture(scope="session")
def RefSimulator(request):
    """The reference simulator.

    Please use this if the test is reference simulator specific.
    Other simulators may choose to implement the same API as the
    reference simulator; this allows them to test easily.
    """
    return TestConfig.RefSimulator


def recorder_dirname(request, name):
    """Returns the directory to put test artifacts in.

    Test artifacts produced by Nengo include plots and analytics.

    Note that the return value might be None, which indicates that the
    artifacts should not be saved.
    """
    record = request.config.getvalue(name)
    if is_string(record):
        return record
    elif not record:
        return None

    simulator, nl = TestConfig.RefSimulator, None
    if 'Simulator' in request.funcargnames:
        simulator = request.getfixturevalue('Simulator')
    # 'nl' stands for the non-linearity used in the neuron equation
    if 'nl' in request.funcargnames:
        nl = request.getfixturevalue('nl')
    elif 'nl_nodirect' in request.funcargnames:
        nl = request.getfixturevalue('nl_nodirect')

    dirname = "%s.%s" % (simulator.__module__, name)
    if nl is not None:
        dirname = os.path.join(dirname, nl.__name__)
    return dirname


def parametrize_function_name(request, function_name):
    """Creates a unique name for a test function.

    The unique name accounts for values passed through
    ``pytest.mark.parametrize``.

    This function is used when naming plots saved through the ``plt`` fixture.
    """
    suffixes = []
    if 'parametrize' in request.keywords:
        argnames = []
        for marker in request.keywords.node.iter_markers("parametrize"):
            argnames.extend([x.strip() for names in marker.args[::2]
                             for x in names.split(',')])
        for name in argnames:
            value = request.getfixturevalue(name)
            if inspect.isclass(value):
                value = value.__name__
            suffixes.append('{}={}'.format(name, value))
    return '+'.join([function_name] + suffixes)


@pytest.fixture
def plt(request):
    """A pyplot-compatible plotting interface.

    Please use this if your test creates plots.

    This will keep saved plots organized in a simulator-specific folder,
    with an automatically generated name. savefig() and close() will
    automatically be called when the test function completes.

    If you need to override the default filename, set `plt.saveas` to
    the desired filename.
    """
    dirname = recorder_dirname(request, 'plots')
    plotter = Plotter(
        dirname, request.module.__name__,
        parametrize_function_name(request, request.function.__name__))
    request.addfinalizer(lambda: plotter.__exit__(None, None, None))
    return plotter.__enter__()


@pytest.fixture
def analytics(request):
    """An object to store data for analytics.

    Please use this if you're concerned that accuracy or speed may regress.

    This will keep saved data organized in a simulator-specific folder,
    with an automatically generated name. Raw data (for later processing)
    can be saved with ``analytics.add_raw_data``; these will be saved in
    separate compressed ``.npz`` files. Summary data can be saved with
    ``analytics.add_summary_data``; these will be saved
    in a single ``.csv`` file.
    """
    dirname = recorder_dirname(request, 'analytics')
    analytics = Analytics(
        dirname, request.module.__name__,
        parametrize_function_name(request, request.function.__name__))
    request.addfinalizer(lambda: analytics.__exit__(None, None, None))
    return analytics.__enter__()


@pytest.fixture
def analytics_data(request):
    paths = request.config.getvalue('compare')
    function_name = parametrize_function_name(request, re.sub(
        '^test_[a-zA-Z0-9]*_', 'test_', request.function.__name__, count=1))
    return [Analytics.load(
        p, request.module.__name__, function_name) for p in paths]


@pytest.fixture
def logger(request):
    """A logging.Logger object.

    Please use this if your test emits log messages.

    This will keep saved logs organized in a simulator-specific folder,
    with an automatically generated name.
    """
    dirname = recorder_dirname(request, 'logs')
    logger = Logger(
        dirname, request.module.__name__,
        parametrize_function_name(request, request.function.__name__))
    request.addfinalizer(lambda: logger.__exit__(None, None, None))
    return logger.__enter__()


def function_seed(function, mod=0):
    """Generates a unique seed for the given test function.

    The seed should be the same across all machines/platforms.
    """
    c = function.__code__

    # get function file path relative to Nengo directory root
    nengo_path = os.path.abspath(os.path.dirname(nengo.__file__))
    path = os.path.relpath(c.co_filename, start=nengo_path)

    # take start of md5 hash of function file and name, should be unique
    hash_list = os.path.normpath(path).split(os.path.sep) + [c.co_name]
    hash_string = ensure_bytes('/'.join(hash_list))
    i = int(hashlib.md5(hash_string).hexdigest()[:15], 16)
    s = (i + mod) % npext.maxint
    int_s = int(s)  # numpy 1.8.0 bug when RandomState on long type inputs
    assert type(int_s) == int  # should not still be a long because < maxint
    return int_s


def get_item_name(item):
    """Get a unique backend-independent name for an item (test function)."""
    item_abspath, item_name = str(item.fspath), item.location[2]
    nengo_path = os.path.abspath(os.path.dirname(nengo.__file__))
    item_relpath = os.path.relpath(item_abspath, start=nengo_path)
    item_relpath = os.path.join('nengo', item_relpath)
    item_relpath = item_relpath.replace(os.sep, '/')
    return '%s:%s' % (item_relpath, item_name)


@pytest.fixture
def rng(request):
    """A seeded random number generator.

    This should be used in lieu of np.random because we control its seed.
    """
    # add 1 to seed to be different from `seed` fixture
    seed = function_seed(request.function, mod=TestConfig.test_seed + 1)
    return np.random.RandomState(seed)


@pytest.fixture
def seed(request):
    """A seed for seeding Networks.

    This should be used in lieu of an integer seed so that we can ensure that
    tests are not dependent on specific seeds.
    """
    return function_seed(request.function, mod=TestConfig.test_seed)


def pytest_generate_tests(metafunc):
    marks = [
        getattr(pytest.mark, m.name)(*m.args, **m.kwargs)
        for m in getattr(metafunc.function, 'pytestmark', [])]

    def mark_nl(nl):
        if nl is Sigmoid:
            nl = pytest.param(nl, marks=[pytest.mark.filterwarnings(
                'ignore:overflow encountered in exp')] + marks)
        return nl

    if "nl" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl", [mark_nl(nl) for nl in TestConfig.neuron_types])
    if "nl_nodirect" in metafunc.funcargnames:
        nodirect = [mark_nl(n) for n in TestConfig.neuron_types
                    if n is not Direct]
        metafunc.parametrize("nl_nodirect", nodirect)


def pytest_collection_modifyitems(session, config, items):
    if config.getvalue('noexamples'):
        deselect_by_condition(
            lambda item: getattr(item.obj, 'example', None), items, config)
    if not config.getvalue('slow'):
        skip_slow = pytest.mark.skip("slow tests not requested")
        for item in items:
            if getattr(item.obj, 'slow', None):
                item.add_marker(skip_slow)
    if not TestConfig.compare_requested:
        deselect_by_condition(
            lambda item: getattr(item.obj, 'compare', None), items, config)

    uses_sim = lambda item: 'Simulator' in item.fixturenames
    uses_refsim = lambda item: 'RefSimulator' in item.fixturenames
    if TestConfig.is_skipping_frontend_tests():
        deselect_by_condition(
            lambda item: not (uses_sim(item) or uses_refsim(item)),
            items, config)
        deselect_by_condition(
            lambda item: uses_refsim(item)
            and not TestConfig.is_refsim_overridden(),
            items, config)
        deselect_by_condition(
            lambda item: uses_sim(item)
            and not TestConfig.is_sim_overridden(),
            items, config)

    deselect_by_condition(
        lambda item: getattr(item.obj, 'noassertions', None)
        and not any(
            fixture in item.fixturenames and config.getvalue(option)
            for fixture, option in [
                ('analytics', 'analytics'),
                ('plt', 'plots'),
                ('logger', 'logs'),
            ]),
        items, config)


def deselect_by_condition(condition, items, config):
    remaining = []
    deselected = []
    for item in items:
        if condition(item):
            deselected.append(item)
        else:
            remaining.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining


def pytest_report_collectionfinish(config, startdir, items):
    deselect_reasons = ["Nengo core tests collected"]

    if config.getvalue('noexamples'):
        deselect_reasons.append(
            " example tests deselected (--noexamples passed)")
    if not config.getvalue('slow'):
        deselect_reasons.append(
            " slow tests skipped (pass --slow to run them)")
    if not TestConfig.compare_requested:
        deselect_reasons.append(
            " compare tests deselected (pass --compare to run them).")

    if TestConfig.is_skipping_frontend_tests():
        deselect_reasons.append(
            " frontend tests deselected because --simulator or "
            "--ref-simulator was passed")
        if not TestConfig.is_refsim_overridden():
            deselect_reasons.append(
                " backend tests for non-reference simulator deselected "
                "because only --ref-simulator was passed")
        if not TestConfig.is_sim_overridden():
            deselect_reasons.append(
                " backend tests for reference simulator deselected "
                "because only --simulator was passed")

    for option in ('analytics', 'plots', 'logs'):
        if not config.getvalue(option):
            deselect_reasons.append(
                " {option} not requested (pass --{option} to generate)".format(
                    option=option))

    return deselect_reasons


def pytest_runtest_setup(item):  # noqa: C901
    rc.reload_rc([])
    rc.set('decoder_cache', 'enabled', 'False')
    rc.set('exceptions', 'simplified', 'False')

    if not hasattr(item, 'obj'):
        return  # Occurs for doctests, possibly other weird tests

    test_uses_sim = 'Simulator' in item.fixturenames
    test_uses_refsim = 'RefSimulator' in item.fixturenames
    tests_frontend = not (test_uses_sim or test_uses_refsim)

    if not tests_frontend:
        item_name = get_item_name(item)

        for test, reason in TestConfig.Simulator.unsupported:
            # We add a '*' before test to eliminate the surprise of needing
            # a '*' before the name of a test function.
            if fnmatch(item_name, '*' + test):
                pytest.xfail(reason)


def pytest_terminal_summary(terminalreporter):
    reports = terminalreporter.getreports('passed')
    if not reports or terminalreporter.config.getvalue('compare') is None:
        return
    terminalreporter.write_sep("=", "PASSED")
    for rep in reports:
        for name, content in rep.sections:
            terminalreporter.writer.sep("-", name)
            terminalreporter.writer.line(content)
