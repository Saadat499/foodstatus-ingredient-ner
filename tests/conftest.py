"""Shared pytest setup.

The classifier loads data/ingredient_reference.csv with a path relative to
the working directory (at import time), so tests must run from the
repository root regardless of where pytest is launched from.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
