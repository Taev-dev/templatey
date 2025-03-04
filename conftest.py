import pytest


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
