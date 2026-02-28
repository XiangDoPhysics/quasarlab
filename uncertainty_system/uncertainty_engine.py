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
from dataclasses import dataclass, field
from typing import Optional, Callable
import sympy as sp
from collections import OrderedDict
import textwrap


# ═══════════════════════════════════════════════════════════════════════
# §1  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UncertaintySource:
    """Represents a single source of uncertainty (Type A or Type B)."""
    name: str
    type: str                     # "A" or "B"
    value: float                  # standard uncertainty u(x)
    description: str = ""
    distribution: str = "normal"  # "normal", "rectangular", "triangular", "u-shaped"
    dof: float = float('inf')    # degrees of freedom (ν)
    raw_data: Optional[np.ndarray] = None

    def __repr__(self):
        return f"UncertaintySource({self.name}: u={self.value:.4g}, Type {self.type})"


@dataclass
class MeasuredQuantity:
    """A directly measured physical quantity with its uncertainty budget."""
    name: str
    symbol: str
    unit: str
    best_value: float
    sources: list = field(default_factory=list)

    @property
    def combined_uncertainty(self) -> float:
        """Root-sum-of-squares of all uncertainty sources."""
        return np.sqrt(sum(s.value**2 for s in self.sources))

    @property
    def relative_uncertainty(self) -> float:
        if self.best_value == 0:
            return float('inf')
        return self.combined_uncertainty / abs(self.best_value)

    @property
    def effective_dof(self) -> float:
        """Welch-Satterthwaite effective degrees of freedom."""
        u_c = self.combined_uncertainty
        if u_c == 0:
            return float('inf')
        numerator = u_c**4
        denominator = sum(
            (s.value**4 / s.dof) for s in self.sources if s.dof != float('inf')
        )
        # Sources with infinite dof contribute 0 to denominator
        if denominator == 0:
            return float('inf')
        return numerator / denominator

    def add_type_a(self, data: np.ndarray, name: str = "", description: str = ""):
        """
        Add Type A uncertainty from repeated measurements.
        Uses standard error of the mean: u = s / √n
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        if n < 2:
            raise ValueError("Type A evaluation requires at least 2 measurements.")

        mean = np.mean(data)
        std = np.std(data, ddof=1)          # sample standard deviation
        sem = std / np.sqrt(n)              # standard error of the mean
        dof = n - 1                         # degrees of freedom

        self.best_value = mean

        source = UncertaintySource(
            name=name or f"{self.name}_typeA",
            type="A",
            value=sem,
            description=description or f"Statistical scatter from {n} repeated measurements",
            distribution="normal",  # t-distribution, approximated as normal for large n
            dof=dof,
            raw_data=data.copy()
        )
        self.sources.append(source)
        return source

    def add_type_b(self, uncertainty: float, name: str = "",
                   description: str = "", distribution: str = "normal",
                   is_half_width: bool = False):
        """
        Add Type B uncertainty from non-statistical knowledge.

        Parameters
        ----------
        uncertainty : float
            If is_half_width=False, this is already the standard uncertainty u(x).
            If is_half_width=True, this is the half-width 'a' of the distribution,
            and the standard uncertainty is computed from the distribution type.
        distribution : str
            "normal"      → u = uncertainty (or a/2 if half-width, assuming 95%)
            "rectangular" → u = a / √3
            "triangular"  → u = a / √6
            "u-shaped"    → u = a / √2
        is_half_width : bool
            If True, 'uncertainty' is the half-width of the distribution bounds.
        """
        divisors = {
            "normal": 1.0,       # already standard uncertainty
            "rectangular": np.sqrt(3),
            "triangular": np.sqrt(6),
            "u-shaped": np.sqrt(2),
        }

        if distribution not in divisors:
            raise ValueError(
                f"Unknown distribution '{distribution}'. "
                f"Choose from: {list(divisors.keys())}"
            )

        if is_half_width:
            if distribution == "normal":
                # Assume half-width corresponds to ~95% coverage → k=2
                std_u = uncertainty / 2.0
            else:
                std_u = uncertainty / divisors[distribution]
        else:
            std_u = uncertainty

        source = UncertaintySource(
            name=name or f"{self.name}_typeB",
            type="B",
            value=std_u,
            description=description or f"Systematic uncertainty ({distribution} distribution)",
            distribution=distribution,
            dof=float('inf')  # Type B → infinite dof by convention
        )
        self.sources.append(source)
        return source

    def summary_dict(self) -> dict:
        return {
            "quantity": self.name,
            "symbol": self.symbol,
            "unit": self.unit,
            "best_value": self.best_value,
            "combined_u": self.combined_uncertainty,
            "relative_u_pct": self.relative_uncertainty * 100,
            "eff_dof": self.effective_dof,
            "sources": [
                {
                    "name": s.name,
                    "type": s.type,
                    "u": s.value,
                    "distribution": s.distribution,
                    "dof": s.dof,
                    "description": s.description
                }
                for s in self.sources
            ]
        }


# ═══════════════════════════════════════════════════════════════════════
# §2  UNCERTAINTY PROPAGATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

class DerivedQuantity:
    """
    A quantity derived from measured quantities via a formula.
    Uses symbolic differentiation for exact partial derivatives (GUM linear method).
    """

    def __init__(self, name: str, symbol: str, unit: str,
                 formula_str: str, variables: dict):
        """
        Parameters
        ----------
        name : str
            Human-readable name (e.g., "Acceleration due to gravity").
        symbol : str
            LaTeX-like symbol (e.g., "g").
        unit : str
            Unit string (e.g., "m/s²").
        formula_str : str
            Sympy-parseable formula string, e.g. "4 * pi**2 * L / T**2"
        variables : dict
            Mapping of symbol string → MeasuredQuantity objects.
            e.g. {"L": length_qty, "T": period_qty}
        """
        self.name = name
        self.symbol = symbol
        self.unit = unit
        self.formula_str = formula_str
        self.variables = OrderedDict(variables)

        # Build symbolic expression
        self.sym_vars = {k: sp.Symbol(k) for k in self.variables}
        self.expr = sp.sympify(formula_str, locals=self.sym_vars)

        # Compute partial derivatives symbolically
        self.partials = {}
        for var_name, sym in self.sym_vars.items():
            self.partials[var_name] = sp.diff(self.expr, sym)

    @property
    def best_value(self) -> float:
        subs = {self.sym_vars[k]: v.best_value for k, v in self.variables.items()}
        return float(self.expr.evalf(subs=subs))

    def sensitivity_coefficients(self) -> dict:
        """Evaluate ∂f/∂xᵢ at the best-estimate values."""
        subs = {self.sym_vars[k]: v.best_value for k, v in self.variables.items()}
        return {
            k: float(partial.evalf(subs=subs))
            for k, partial in self.partials.items()
        }

    @property
    def combined_uncertainty(self) -> float:
        """
        Combined standard uncertainty via linear error propagation:
            u_c² = Σᵢ (∂f/∂xᵢ)² · u(xᵢ)²
        Assumes uncorrelated input quantities.
        """
        coeffs = self.sensitivity_coefficients()
        variance = 0.0
        for var_name, qty in self.variables.items():
            c_i = coeffs[var_name]
            u_i = qty.combined_uncertainty
            variance += (c_i * u_i)**2
        return np.sqrt(variance)

    @property
    def relative_uncertainty(self) -> float:
        bv = self.best_value
        if bv == 0:
            return float('inf')
        return self.combined_uncertainty / abs(bv)

    @property
    def effective_dof(self) -> float:
        """Welch-Satterthwaite for the derived quantity."""
        coeffs = self.sensitivity_coefficients()
        u_c = self.combined_uncertainty
        if u_c == 0:
            return float('inf')

        numerator = u_c**4
        denominator = 0.0
        for var_name, qty in self.variables.items():
            c_i = coeffs[var_name]
            u_i = qty.combined_uncertainty
            nu_i = qty.effective_dof
            contribution = (c_i * u_i)**2
            if nu_i != float('inf') and contribution > 0:
                denominator += contribution**2 / nu_i
        if denominator == 0:
            return float('inf')
        return numerator / denominator

    def uncertainty_budget(self) -> list:
        """
        Returns a list of dicts showing each variable's contribution
        to the total uncertainty budget.
        """
        coeffs = self.sensitivity_coefficients()
        u_c_sq = self.combined_uncertainty**2
        budget = []
        for var_name, qty in self.variables.items():
            c_i = coeffs[var_name]
            u_i = qty.combined_uncertainty
            contribution = (c_i * u_i)**2
            pct = (contribution / u_c_sq * 100) if u_c_sq > 0 else 0
            budget.append({
                "variable": var_name,
                "quantity": qty.name,
                "best_value": qty.best_value,
                "unit": qty.unit,
                "u_input": u_i,
                "sensitivity_coeff": c_i,
                "|c·u|": abs(c_i * u_i),
                "variance_contribution": contribution,
                "pct_contribution": pct
            })
        return budget

    def expanded_uncertainty(self, coverage_p: float = 0.95) -> tuple:
        """
        Compute expanded uncertainty U = k · u_c for the given coverage probability.
        Uses the t-distribution with effective degrees of freedom.
        """
        from scipy.stats import t as t_dist

        nu_eff = self.effective_dof
        u_c = self.combined_uncertainty

        if np.isinf(nu_eff) or nu_eff > 1000:
            # Normal approximation
            from scipy.stats import norm
            k = norm.ppf((1 + coverage_p) / 2)
        else:
            nu_eff_int = max(1, int(round(nu_eff)))
            k = t_dist.ppf((1 + coverage_p) / 2, df=nu_eff_int)

        U = k * u_c
        return U, k, nu_eff


# ═══════════════════════════════════════════════════════════════════════
# §3  REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class UncertaintyReport:
    """Generates formatted summary reports for uncertainty analyses."""

    @staticmethod
    def _hline(width=72):
        return "─" * width

    @staticmethod
    def _dline(width=72):
        return "═" * width

    @classmethod
    def generate(cls, derived: DerivedQuantity, coverage_p: float = 0.95,
                 title: str = "") -> str:
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