"""
╔══════════════════════════════════════════════════════════════════════╗
║  QUASARlab Integration — Claude AI for Experimental Analysis         ║
║                                                                      ║
║  Gives Claude AI the ability to:                                     ║
║    • Analyse uncertainty budgets and identify dominant sources       ║
║    • Suggest experimental improvements to reduce uncertainty         ║
║    • Validate measurement consistency against expected values        ║
║    • Guide the experimenter through the measurement process          ║
║    • Generate publication-ready interpretations                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import numpy as np
from anthropic import Anthropic
from uncertainty_system import (
    MeasuredQuantity, DerivedQuantity, UncertaintyReport
)

# ═══════════════════════════════════════════════════════════════════════
# §1  EXPERIMENT CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════

class ExperimentContext:
    """
    Converts UncertaintySystem objects into structured context
    that Claude can reason about effectively.
    """

    @staticmethod
    def build_context(derived: DerivedQuantity,
                      experiment_name: str = "",
                      notes: str = "")  -> str:
        
        """
        Sterilise the full experiment state into a structured prompt
        context that Claude can analyse. This includes:
          • The derived quantity being measured
          • The measured quantities and their uncertainties
          • The mathematical relationship between them
          • Any additional notes or context about the experiment
        """
        budget = derived.uncertainty_budget()
        U, k, nu_eff =  derived.expanded_uncertainty(0.95)
        coeffs = derived.sensitivity_coefficients()

        context = {
            "experiment": experiment_name or derived.name,
            "model_equation": f"{derived.symbol} = {derived.formula_str}",
            "result":{
                "symbol": derived.symbol,
                "best_value": round(derived.best_value, 6),
                "unit": derived.unit,
                "combined_std_uncertainty": round(derived.combined.std_uncertainty, 6),
                "relative uncertainty": round(derived.relative_uncertainty * 100, 4),
                "expanded_uncertainty_95pct": round(U, 6),
                "coverage factor, k": round(k, 3),
                "effective_dof": round(nu_eff, 1),
            },
            "input_quantities":[],
            "uncertainty_budget": [],
        }

# Input quantities with full source breakdown
for var_name, qty in derived.varibles.items():
    qty_info = {
        "variable": var_name,
        "name": qty.name,
        "best_value": round(qty.best_value, 6),
        "unit": qty.unit,
        "combined_uncertainty": round(qty.combined_uncertainty, 6),
        "relative_uncertainty": round(qty.relative_uncertainty * 100, 4),
        "sources": []
    }

    for src in qty.sources():
        src_info = {
            "name": src.name,
            "type": f"Type {src.type}",
            "standard_uncertainty": round()
        }