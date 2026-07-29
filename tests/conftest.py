"""Shared pytest fixtures / setup.

The example plants and several tests build a ``pyadm1.Feedstock`` from bare
substrate IDs (e.g. ``maize_silage_milk_ripeness``), which pyadm1 resolves against
its substrate registry — a repo-root ``data/`` directory that is NOT bundled into
a wheel install and is therefore absent on CI. To keep the test suite
self-contained, we ship the handful of substrate definitions the tests need under
``tests/substrates/`` and point pyadm1's default registry there for the whole test
session. This affects tests only; the shipped package still uses the real
registry.
"""

from pathlib import Path

import pyadm1.substrates.feedstock as _feedstock

_TEST_SUBSTRATE_DIR = Path(__file__).parent / "substrates"

# Resolve bare substrate IDs against the bundled test fixtures (see module docstring).
_feedstock._DEFAULT_DATA_DIR = _TEST_SUBSTRATE_DIR
_feedstock._DEFAULT_XML_DIR = _TEST_SUBSTRATE_DIR
