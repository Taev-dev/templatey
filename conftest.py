from collections import defaultdict

import pytest

from templatey.templates import _PENDING_FORWARD_REFS
from templatey.templates import anchor_closure_scope


def pytest_addoption(parser):
    parser.addoption(
        '--run-benchmarks',
        action='store_true', default=False, help='Run benchmarks')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-benchmarks'):
        return
    skip_benchmark = pytest.mark.skip(
        reason='Needs --run-benchmark to run benchmarks')

    for item in items:
        if 'benchmark' in item.keywords:
            item.add_marker(skip_benchmark)


@pytest.fixture(autouse=True, scope='function')
def clean_pending_forward_refs_registry():
    """Layers a fresh, clean pending forward refs registry over the
    default one for testing. Note that this only takes effect WITHIN
    tests, so it only applies to templates defined within the test
    functions themselves. Templates defined at a test module level
    will be unaffected.
    """
    token = _PENDING_FORWARD_REFS.set(defaultdict(set))
    try:
        yield
    finally:
        _PENDING_FORWARD_REFS.reset(token)


@pytest.fixture(autouse=True, scope='function')
def apply_anchor_closure_scope():
    """Makes sure that all test functions have a new closure scope, so
    they work correctly with closures.
    """
    with anchor_closure_scope():
        yield
