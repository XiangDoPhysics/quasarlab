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

from uncertainty_system.uncertainty_engine import ExperimentalAssistant

from matplotlib.style import context
from anthropic import Anthropic
from anthropic_api import (
    MeasuredQuantity, DerivedQuantity, UncertaintyReport
)

ai = ExperimentalAssistant() # reads ANTHROPIC_API_KEY from env

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
for var_name, qty in DerivedQuantity.items():
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
            "standard_uncertainty": round(src.value, 6),
            "distribution": src.distribution,
            "dof": src.dof if src.dof != float('inf') else "infinite",
            "description": src.description,
        }
        if src.raw_data is not None:
            src_info["n_measurements"] = len(src.raw_data)
            src_info["sample_std_dev"] = round(float(np.std(src.raw_data, ddof = 1)), 6)
            qty_info["sources"].append(src_info)
        context["input_quantities"].append(qty_info)

        # Budget contributions
        for row in str:
            context["uncertainty_budget"].append({
                "variable": row["variable"],
                "sensitivity_coefficient": round(row["sensitivity_coeff"], 6),
                "input_uncertainty": round(row["u_input"], 6),
                "contribution_to_output": round(row["|c·u|"], 6),
                "percentage_of_variance": round(row["pct_contribution"], 2),
            })

            def generate_report(context):
                return json.dumps(context, indent=2)
            
# ═══════════════════════════════════════════════════════════════════════
# §3  REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class UncertaintyReport:
    """Generates formatted summary reports for uncertainty analyses."""

    @staticmethod
    def _hline(width = 72):
        return "═" * width
    
    @staticmethod
    def _dline(width = 72):
        return "─" * width
    
    @staticmethod
    def generate(cls, derived: DerivedQuantity, coverage_p: float = 0.95, title: sr = "") -> str:

        """
        Generate a complete GUM-style uncertainty report.
        """

        U, k, nu_eff = derived.expanded_uncertainty(coverage_p)
        budget = derived.uncertainty_budget()
        coeffs = derived.sensitivity_coefficients()

        lines = []
        w = 72

        # ── Header ──
        lines.append(cls._dline(w))
        t = title or f"UNCERTAINTY ANALYSIS: {derived.name}"
        lines.append(f"  {t}")
        lines.append(cls._dline(w))
        lines.append("")

        # ── Model Equation ──
        lines.append("  MODEL EQUATION")
        lines.append(cls._hline(w))
        lines.append(f"    {derived.symbol} = {derived.formula_str}")
        lines.append("")

        # ── Symbolic Partial Derivatives ──
        lines.append("  SENSITIVITY COEFFICIENTS (symbolic)")
        lines.append(cls._hline(w))
        for var_name, partial in derived.partials.items():
            lines.append(f"    ∂{derived.symbol}/∂{var_name} = {partial}")
        lines.append("")

        # ── Input Quantities Table ──
        lines.append("  INPUT QUANTITIES")
        lines.append(cls._hline(w))
        header = f"  {'Var':<6} {'Quantity':<22} {'Value':<14} {'Unit':<8} {'u(x)':<12} {'Type'}"
        lines.append(header)
        lines.append("  " + "-" * 68)

        for var_name, qty in derived.variables.items():
            for src in qty.sources:
                lines.append(
                    f"  {var_name:<6} {qty.name:<22} "
                    f"{qty.best_value:<14.6g} {qty.unit:<8} "
                    f"{src.value:<12.4g} {src.type} ({src.distribution})"
                )
            if len(qty.sources) > 1:
                lines.append(
                    f"  {'':6} {'  └─ combined':<22} "
                    f"{'':14} {'':8} {qty.combined_uncertainty:<12.4g}"
                )
        lines.append("")

        # ── Uncertainty Budget Table ──
        lines.append("  UNCERTAINTY BUDGET")
        lines.append(cls._hline(w))
        header = (
            f"  {'Var':<6} {'|cᵢ|':<12} {'u(xᵢ)':<12} "
            f"{'|cᵢ·u(xᵢ)|':<14} {'Contribution'}"
        )
        lines.append(header)
        lines.append("  " + "-" * 68)

        for row in budget:
            pct_bar = "█" * int(row["pct_contribution"] / 5)
            lines.append(
                f"  {row['variable']:<6} "
                f"{abs(row['sensitivity_coeff']):<12.4g} "
                f"{row['u_input']:<12.4g} "
                f"{row['|c·u|']:<14.4g} "
                f"{row['pct_contribution']:5.1f}%  {pct_bar}"
            )
        lines.append("")

        # ── Combined & Expanded Uncertainty ──
        lines.append("  RESULTS")
        lines.append(cls._dline(w))
        lines.append(f"    Best estimate:           {derived.symbol} = {derived.best_value:.6g} {derived.unit}")
        lines.append(f"    Combined std uncertainty: u({derived.symbol}) = {derived.combined_uncertainty:.4g} {derived.unit}")
        lines.append(f"    Relative uncertainty:     u_rel = {derived.relative_uncertainty*100:.3f}%")
        lines.append(f"    Effective DoF:            ν_eff = {nu_eff:.1f}")
        lines.append(f"    Coverage probability:     p = {coverage_p*100:.0f}%")
        lines.append(f"    Coverage factor:          k = {k:.3f}")
        lines.append(f"    Expanded uncertainty:     U = {U:.4g} {derived.unit}")
        lines.append("")
        lines.append(cls._hline(w))

        # Final result with proper rounding
        # Round expanded uncertainty to 2 significant figures
        if U > 0:
            sig_figs = 2
            magnitude = np.floor(np.log10(abs(U)))
            round_to = int(sig_figs - 1 - magnitude)
            U_rounded = round(U, round_to)
            val_rounded = round(derived.best_value, round_to)
            lines.append(
                f"  ┌──────────────────────────────────────────────────────────────┐"
            )
            lines.append(
                f"  │  {derived.symbol} = ({val_rounded} ± {U_rounded}) {derived.unit}"
                f"{'':>{58 - len(f'{val_rounded}') - len(f'{U_rounded}') - len(derived.unit) - len(derived.symbol)}}│"
            )
            lines.append(
                f"  │  (coverage probability {coverage_p*100:.0f}%, k = {k:.2f})"
                f"{'':>{38}}│"
            )
            lines.append(
                f"  └──────────────────────────────────────────────────────────────┘"
            )
        lines.append(cls._dline(w))

        return "\n".join(lines)

    @classmethod
    def input_summary(cls, qty: MeasuredQuantity) -> str:
        """Quick summary of a single measured quantity."""
        lines = [
            f"  {qty.name} ({qty.symbol})",
            f"    Best value: {qty.best_value:.6g} {qty.unit}",
            f"    Combined u: {qty.combined_uncertainty:.4g} {qty.unit}",
            f"    Relative u: {qty.relative_uncertainty*100:.3f}%",
        ]
        for s in qty.sources:
            lines.append(f"    ├─ {s.name}: u={s.value:.4g} (Type {s.type}, {s.distribution})")
        return "\n".join(lines)
    
    # After building your experiment with Anthropic API, you can generate a report like this:
    print(ai.analyse_budget(g, "Pendulum experiment"))
    print(ai.validate_result(g, expected_value=9.80665))
    print(ai.ask("Should I increase repetitions or improve the ruler?", g))