"""
╔══════════════════════════════════════════════════════════════════════╗
║  UncertaintyEngine — Measurement Uncertainty Management System     ║
║  For Physics Experiments                                           ║
║                                                                    ║
║  Supports:                                                         ║
║    • Type A (statistical) uncertainty from repeated measurements   ║
║    • Type B (systematic) uncertainty from instrument specs         ║
║    • Uncertainty propagation through arbitrary derived quantities  ║
║    • Combined & expanded uncertainty (GUM-compliant)               ║
║    • Summary reports in text and table format                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import sympy as sp
import textwrap
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import OrderedDict

# ═══════════════════════════════════════════════════════════════════════
# §1  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UncertaintySource:
    """Represents a single source of uncertainty (Type A or Type B)."""
    