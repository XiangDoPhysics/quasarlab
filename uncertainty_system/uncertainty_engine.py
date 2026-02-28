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
from uncertainty_engine import (
    MeasuredQuantity, DerivedQuantity, UncertaintyReport
)


# ═══════════════════════════════════════════════════════════════════════
# §1  EXPERIMENT CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════

class ExperimentContext:
    """
    Converts UncertaintyEngine objects into structured context
    that Claude can reason about effectively.
    """

    @staticmethod
    def build_context(derived: DerivedQuantity,
                      experiment_name: str = "",
                      notes: str = "") -> str:
        """
        Serialize the full experimental state into a structured prompt
        context that Claude can analyse.
        """
        budget = derived.uncertainty_budget()
        U, k, nu_eff = derived.expanded_uncertainty(0.95)
        coeffs = derived.sensitivity_coefficients()

        context = {
            "experiment": experiment_name or derived.name,
            "model_equation": f"{derived.symbol} = {derived.formula_str}",
            "result": {
                "symbol": derived.symbol,
                "best_value": round(derived.best_value, 6),
                "unit": derived.unit,
                "combined_std_uncertainty": round(derived.combined_uncertainty, 6),
                "relative_uncertainty_pct": round(derived.relative_uncertainty * 100, 4),
                "expanded_uncertainty_95pct": round(U, 6),
                "coverage_factor_k": round(k, 3),
                "effective_dof": round(nu_eff, 1),
            },
            "input_quantities": [],
            "uncertainty_budget": [],
        }

        # Input quantities with full source breakdown
        for var_name, qty in derived.variables.items():
            qty_info = {
                "variable": var_name,
                "name": qty.name,
                "best_value": round(qty.best_value, 6),
                "unit": qty.unit,
                "combined_uncertainty": round(qty.combined_uncertainty, 6),
                "relative_uncertainty_pct": round(qty.relative_uncertainty * 100, 4),
                "sources": []
            }
            for src in qty.sources:
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
                    src_info["sample_std_dev"] = round(float(np.std(src.raw_data, ddof=1)), 6)
                qty_info["sources"].append(src_info)
            context["input_quantities"].append(qty_info)

        # Budget contributions
        for row in budget:
            context["uncertainty_budget"].append({
                "variable": row["variable"],
                "sensitivity_coefficient": round(row["sensitivity_coeff"], 6),
                "input_uncertainty": round(row["u_input"], 6),
                "contribution_to_output": round(row["|c·u|"], 6),
                "percentage_of_variance": round(row["pct_contribution"], 2),
            })

        if notes:
            context["experimenter_notes"] = notes

        return json.dumps(context, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# §2  CLAUDE EXPERIMENTAL ASSISTANT
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a physics research assistant embedded in QUASARlab, \
a measurement uncertainty analysis system. You have deep expertise in:

- GUM (Guide to the Expression of Uncertainty in Measurement, JCGM 100:2008)
- Experimental physics methodology and best practices
- Statistical analysis of measurement data
- Error analysis and uncertainty propagation

When analysing experimental data, you should:

1. IDENTIFY the dominant uncertainty sources from the budget
2. EXPLAIN the physical meaning of the results
3. SUGGEST concrete, actionable improvements ranked by impact
4. FLAG any potential issues (e.g., suspiciously low scatter suggesting \
   correlated measurements, or results inconsistent with known values)
5. Use proper physics notation and be quantitatively precise

Always ground your analysis in the actual numbers provided. \
Be direct and specific — avoid vague suggestions."""


class ExperimentalAssistant:
    """
    Claude-powered assistant for analysing experimental uncertainty data.
    Wraps the Anthropic API with physics-specific system prompting
    and structured experimental context.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic()   # reads ANTHROPIC_API_KEY from env
        self.model = model
        self.conversation_history = []

    def _call_claude(self, user_message: str, system: str = SYSTEM_PROMPT) -> str:
        """Send a message to Claude and return the response text."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=self.conversation_history
        )

        assistant_text = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_text
        })
        return assistant_text

    # ── High-Level Analysis Methods ──────────────────────────────

    def analyse_budget(self, derived: DerivedQuantity,
                       experiment_name: str = "") -> str:
        """
        Full uncertainty budget analysis: identifies dominant sources,
        suggests improvements, checks for anomalies.
        """
        context = ExperimentContext.build_context(derived, experiment_name)
        prompt = f"""Analyse this experimental uncertainty budget. Identify the \
dominant uncertainty contributor(s), explain what limits the measurement precision, \
and suggest the most impactful improvements the experimenter could make.

<experiment_data>
{context}
</experiment_data>"""
        return self._call_claude(prompt)

    def validate_result(self, derived: DerivedQuantity,
                        expected_value: float,
                        expected_unit: str = "",
                        reference: str = "") -> str:
        """
        Check whether the measured result is consistent with an expected/known value.
        Computes the En number (normalised error) and interprets it.
        """
        context = ExperimentContext.build_context(derived)
        U, k, _ = derived.expanded_uncertainty(0.95)
        deviation = abs(derived.best_value - expected_value)
        en_number = deviation / U if U > 0 else float('inf')

        prompt = f"""Validate this experimental result against the expected value.

<experiment_data>
{context}
</experiment_data>

<validation>
  Expected value: {expected_value} {expected_unit or derived.unit} {f'({reference})' if reference else ''}
  Measured value: {derived.best_value:.6g} ± {U:.4g} {derived.unit} (95%)
  |Deviation|: {deviation:.4g} {derived.unit}
  Expanded uncertainty U(95%): {U:.4g} {derived.unit}
  Normalised error En = |deviation|/U = {en_number:.3f}
  Criterion: En ≤ 1.0 → consistent at 95% confidence
</validation>

Interpret these results. Is the measurement consistent? If not, \
what systematic effects might explain the discrepancy?"""
        return self._call_claude(prompt)

    def suggest_improvements(self, derived: DerivedQuantity,
                             constraints: str = "") -> str:
        """
        Generate ranked, actionable suggestions for reducing uncertainty.
        Optionally considers practical constraints (budget, equipment available, etc.).
        """
        context = ExperimentContext.build_context(derived)
        prompt = f"""Based on this uncertainty budget, suggest specific experimental \
improvements ranked by expected impact on the total uncertainty.

<experiment_data>
{context}
</experiment_data>

{f'<constraints>{constraints}</constraints>' if constraints else ''}

For each suggestion:
- State which uncertainty source it targets
- Estimate the reduction factor if possible
- Note any practical requirements or trade-offs"""
        return self._call_claude(prompt)

    def guide_measurement(self, experiment_description: str) -> str:
        """
        Before the experiment: Claude helps plan the measurement strategy,
        identify likely dominant uncertainty sources, and suggest how many
        repeated measurements to take.
        """
        prompt = f"""I'm planning this experiment and need help designing \
the measurement strategy to minimise uncertainty:

<experiment_plan>
{experiment_description}
</experiment_plan>

Help me:
1. Identify all the quantities I need to measure
2. List the likely uncertainty sources for each (Type A and Type B)
3. Recommend how many repeated measurements to take and why
4. Suggest what instruments/techniques would give the best precision
5. Flag any common systematic errors to watch out for"""
        return self._call_claude(prompt)

    def interpret_for_publication(self, derived: DerivedQuantity,
                                  experiment_name: str = "") -> str:
        """
        Generate a publication-ready paragraph interpreting the result
        and its uncertainty, suitable for a lab report or paper.
        """
        context = ExperimentContext.build_context(derived, experiment_name)
        prompt = f"""Write a concise, publication-quality paragraph reporting \
this measurement result. Include the value with expanded uncertainty, \
the dominant uncertainty sources, and the effective degrees of freedom. \
Use proper scientific language suitable for a physics journal.

<experiment_data>
{context}
</experiment_data>"""
        return self._call_claude(prompt)

    def ask(self, question: str, derived: DerivedQuantity = None) -> str:
        """
        Free-form question about the experiment or uncertainty analysis.
        Optionally attach experimental context.
        """
        if derived:
            context = ExperimentContext.build_context(derived)
            prompt = f"""{question}

<experiment_data>
{context}
</experiment_data>"""
        else:
            prompt = question
        return self._call_claude(prompt)

    def reset_conversation(self):
        """Clear conversation history to start a fresh analysis session."""
        self.conversation_history = []


# ═══════════════════════════════════════════════════════════════════════
# §3  DEMO — Pendulum experiment with AI analysis
# ═══════════════════════════════════════════════════════════════════════

def demo():
    """
    Full demonstration: build the pendulum experiment,
    then let Claude analyse the results.
    """
    # ── Build the experiment (same as pendulum_example.py) ──
    L = MeasuredQuantity(name="Pendulum length", symbol="L", unit="m", best_value=0.5120)
    L.add_type_b(0.0005, "ruler_resolution", "Ruler ±0.5 mm", "rectangular", True)
    L.add_type_b(0.001, "bob_com", "Bob centre-of-mass ±1 mm", "rectangular", True)

    raw_10T = np.array([14.38, 14.41, 14.35, 14.42, 14.39, 14.36, 14.40, 14.37])
    T = MeasuredQuantity(name="Period", symbol="T", unit="s", best_value=0.0)
    T.add_type_a(raw_10T / 10.0, "timing_scatter", "8 repeated timing measurements")
    T.add_type_b(0.001, "stopwatch_resolution", "Stopwatch ±0.01s / 10 osc", "rectangular", True)
    T.add_type_b(0.005, "reaction_time", "Reaction time ±0.05s / 10 osc", "normal", True)

    g = DerivedQuantity(
        name="Acceleration due to gravity",
        symbol="g", unit="m/s²",
        formula_str="4 * pi**2 * L / T**2",
        variables={"L": L, "T": T}
    )

    # ── Print the standard report ──
    print(UncertaintyReport.generate(g, coverage_p=0.95,
          title="DETERMINATION OF g VIA SIMPLE PENDULUM"))

    # ── AI-powered analysis ──
    print("\n" + "=" * 72)
    print("  CLAUDE AI ANALYSIS")
    print("=" * 72 + "\n")

    ai = ExperimentalAssistant()

    # 1. Budget analysis
    print("━━━ UNCERTAINTY BUDGET ANALYSIS ━━━\n")
    print(ai.analyse_budget(g, "Simple pendulum — determination of g"))

    # 2. Validation against known value
    print("\n━━━ RESULT VALIDATION ━━━\n")
    print(ai.validate_result(g, expected_value=9.80665,
          reference="standard gravity"))

    # 3. Improvement suggestions
    print("\n━━━ IMPROVEMENT SUGGESTIONS ━━━\n")
    print(ai.suggest_improvements(g,
          constraints="Undergraduate lab, budget ~$200, 2-hour session"))

    # 4. Publication paragraph
    print("\n━━━ PUBLICATION PARAGRAPH ━━━\n")
    print(ai.interpret_for_publication(g, "Simple pendulum experiment"))


if __name__ == "__main__":
    demo()