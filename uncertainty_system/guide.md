# UncertaintyEngine — Step-by-Step Guide

## Measurement Uncertainty Management System for Physics Experiments

A modular, GUM-compliant Python framework for calculating, propagating, and reporting measurement uncertainties.

---

## Architecture Overview

The system has three core components:

```
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  MeasuredQuantity    │────▶│  DerivedQuantity      │────▶│ UncertaintyReport│
│                      │     │                       │     │                  │
│  • Type A (stats)    │     │  • Symbolic formula    │     │  • Budget table  │
│  • Type B (system.)  │     │  • Auto-differentiate  │     │  • Combined u    │
│  • Combined u(x)     │     │  • Propagate u         │     │  • Expanded U    │
│  • Welch-Satterth.   │     │  • Welch-Satterth.     │     │  • Final result  │
└─────────────────────┘     └──────────────────────┘     └──────────────────┘
```

---

## Step-by-Step Workflow

### Step 1 — Define Measured Quantities

Each directly measured physical quantity is a `MeasuredQuantity` object holding its best value, unit, and uncertainty sources.

```python
from uncertainty_engine import MeasuredQuantity

L = MeasuredQuantity(
    name="Pendulum length",
    symbol="L",
    unit="m",
    best_value=0.5120
)
```

### Step 2 — Add Type A Uncertainty (Statistical)

Feed in raw repeated measurements. The system computes the mean, sample standard deviation, standard error of the mean, and degrees of freedom automatically.

```python
import numpy as np

raw_T = np.array([1.438, 1.441, 1.435, 1.442, 1.439, 1.436, 1.440, 1.437])

T = MeasuredQuantity(name="Period", symbol="T", unit="s", best_value=0.0)
T.add_type_a(
    data=raw_T,
    name="timing_scatter",
    description="Statistical scatter from 8 repeated measurements"
)
# T.best_value is now set to the mean
# u(T)_A = s/√n is computed and stored
```

**What happens internally:**
- Mean: x̄ = (1/n)Σxᵢ
- Sample std: s = √[Σ(xᵢ − x̄)²/(n−1)]
- Standard error: u = s/√n
- Degrees of freedom: ν = n − 1

### Step 3 — Add Type B Uncertainty (Systematic)

For uncertainties from instrument specifications, calibration certificates, or engineering judgement. Supports four distribution models:

| Distribution | Divisor | Typical use case |
|---|---|---|
| `normal` | 1 (or /2 if half-width, assuming k=2) | Calibration certificates with stated confidence |
| `rectangular` | √3 | Instrument resolution, digital readouts |
| `triangular` | √6 | Better-known bounds with central tendency |
| `u-shaped` | √2 | Environmental oscillations (e.g. temperature cycling) |

```python
# Ruler resolution: ±0.5 mm half-width, rectangular
L.add_type_b(
    uncertainty=0.0005,
    name="ruler_resolution",
    description="Ruler least count ±0.5 mm",
    distribution="rectangular",
    is_half_width=True     # system divides by √3 automatically
)
```

**Key parameter:** `is_half_width`
- `True` → you provide the half-width *a* of the distribution bounds; the system converts to standard uncertainty using the appropriate divisor.
- `False` → you provide the standard uncertainty directly.

### Step 4 — Define the Derived Quantity with a Formula

Write the formula as a string that SymPy can parse. Map each symbol to its `MeasuredQuantity`.

```python
from uncertainty_engine import DerivedQuantity

g = DerivedQuantity(
    name="Acceleration due to gravity",
    symbol="g",
    unit="m/s²",
    formula_str="4 * pi**2 * L / T**2",
    variables={"L": L, "T": T}
)
```

**What happens internally:**
1. SymPy parses the formula into a symbolic expression
2. Partial derivatives ∂g/∂L and ∂g/∂T are computed symbolically
3. Sensitivity coefficients cᵢ = ∂f/∂xᵢ are evaluated at best-estimate values
4. Combined uncertainty: u_c(g) = √[Σ(cᵢ · u(xᵢ))²]
5. Welch-Satterthwaite effective degrees of freedom for the expanded uncertainty

### Step 5 — Generate the Report

```python
from uncertainty_engine import UncertaintyReport

report = UncertaintyReport.generate(g, coverage_p=0.95)
print(report)
```

This produces a complete report containing the model equation, symbolic partial derivatives, input quantities table, uncertainty budget with percentage contributions, combined and expanded uncertainty, and the final result in (value ± U) format.

---

## Adapting for Different Experiments

### New Experiment Checklist

1. **Identify all measured quantities** — what do you measure directly?
2. **For each quantity, catalogue uncertainty sources:**
   - Repeated measurements → Type A
   - Instrument resolution → Type B (rectangular)
   - Calibration uncertainty → Type B (usually normal)
   - Environmental effects → Type B (distribution depends on knowledge)
3. **Write the formula** connecting measured quantities to the result
4. **Build the code** following the 5-step pattern above

### Example: Resistivity from Resistance Measurement

```python
# ρ = R · A / l  where R = V/I
V = MeasuredQuantity(name="Voltage", symbol="V", unit="V", best_value=0.0)
V.add_type_a(data=voltage_readings)
V.add_type_b(0.001, name="dmm_accuracy", distribution="rectangular", is_half_width=True)

I = MeasuredQuantity(name="Current", symbol="I", unit="A", best_value=0.0)
I.add_type_a(data=current_readings)
I.add_type_b(0.0005, name="dmm_accuracy", distribution="rectangular", is_half_width=True)

A = MeasuredQuantity(name="Cross-section area", symbol="A", unit="m²", best_value=3.14e-6)
A.add_type_b(0.05e-6, name="micrometer", distribution="rectangular", is_half_width=True)

l = MeasuredQuantity(name="Wire length", symbol="l", unit="m", best_value=1.000)
l.add_type_b(0.0005, name="ruler", distribution="rectangular", is_half_width=True)

rho = DerivedQuantity(
    name="Resistivity",
    symbol="rho",
    unit="Ω·m",
    formula_str="V * A / (I * l)",
    variables={"V": V, "I": I, "A": A, "l": l}
)
```

### Supported Formula Syntax

The formula string accepts standard SymPy expressions:

| Operation | Syntax |
|---|---|
| Multiplication | `a * b` |
| Division | `a / b` |
| Power | `a**2`, `a**0.5` |
| Square root | `sqrt(x)` |
| Trig functions | `sin(x)`, `cos(x)`, `tan(x)`, `atan(x)` |
| Logarithms | `log(x)` (natural), `log(x, 10)` |
| Constants | `pi`, `E` |

---

## Key Methods Reference

### MeasuredQuantity

| Property/Method | Returns |
|---|---|
| `.combined_uncertainty` | RSS of all sources |
| `.relative_uncertainty` | u/|x̄| |
| `.effective_dof` | Welch-Satterthwaite ν_eff |
| `.add_type_a(data, ...)` | Adds statistical source |
| `.add_type_b(uncertainty, ...)` | Adds systematic source |
| `.summary_dict()` | Full dict for programmatic use |

### DerivedQuantity

| Property/Method | Returns |
|---|---|
| `.best_value` | Evaluated formula at best estimates |
| `.combined_uncertainty` | Propagated u_c |
| `.relative_uncertainty` | u_c/|f| |
| `.sensitivity_coefficients()` | Dict of ∂f/∂xᵢ values |
| `.uncertainty_budget()` | List of contribution dicts |
| `.expanded_uncertainty(p)` | (U, k, ν_eff) tuple |

### UncertaintyReport

| Method | Returns |
|---|---|
| `.generate(derived, coverage_p)` | Full formatted report string |
| `.input_summary(qty)` | Quick summary for one quantity |

---

## Dependencies

```
numpy
sympy
scipy
```

Install via: `pip install numpy sympy scipy`

---

## Theoretical Foundation

This implementation follows the **GUM** (Guide to the Expression of Uncertainty in Measurement, JCGM 100:2008):

- **Type A**: Evaluated by statistical analysis of repeated observations
- **Type B**: Evaluated by other means (manufacturer specs, calibration data, physical reasoning)
- **Propagation**: First-order Taylor expansion (linear approximation) assuming uncorrelated inputs
- **Effective DoF**: Welch-Satterthwaite equation for combining finite and infinite degrees of freedom
- **Expanded uncertainty**: U = k·u_c where k is from the t-distribution at the desired coverage probability