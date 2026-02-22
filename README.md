# QuasarAI


**AI-powered physics research assistant.** Automating data analysis, error calculations, experimental design evaluation, and workflow optimization so researchers can focus on what matters most: discovery.

Built on [Claude](https://www.anthropic.com/claude) by Anthropic, QuasarAI combines advanced language understanding with physics-first reasoning to deliver a rigorous, transparent, and extensible research tool.

---

## Overview

QuasarAI assists physicists, students, and researchers by:

- Analyzing experimental data from scientific instruments
- Performing and explaining **error analysis**
- Evaluating experimental design and methodology
- Suggesting **improvements to experimental setups**
- Assisting with **literature review** and scientific reasoning
- Critically assessing experimental results and conclusions

---

## Core Features

### Data Analysis
- Interpret raw and processed data from physics experiments
- Identify trends, anomalies, and inconsistencies
- Support regression, fitting, and statistical analysis

### Error Analysis
- Propagation of uncertainties
- Identification of systematic vs statistical errors
- Recommendations to reduce measurement uncertainty

### Experimental Evaluation
- Review experimental procedures and assumptions
- Detect conceptual or methodological flaws
- Compare results against theoretical expectations

### Experimental Improvement Suggestions
- Propose modifications to experimental setups
- Optimize measurement precision and reliability
- Highlight missing controls or calibration steps

### Literature Review Assistance
- Summarize relevant research concepts
- Compare methodologies across studies
- Help contextualize results within existing literature

---

## Installation

### Prerequisites

- Python 3.13+
- Anthropic API key ([get one here](https://console.anthropic.com/))

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/QuasarAI.git
cd QuasarAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your API key:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

On Windows:
```powershell
set ANTHROPIC_API_KEY=your-api-key-here
```

---

## Usage

### Interactive Mode

Start an interactive conversation with the AI:

```bash
python main.py
```

This opens a chat interface where you can ask physics questions, discuss experiments, and analyze data.

### Command-Line Modes

#### Analyze Experimental Data

```bash
python main.py --mode analyze \
  --input "Young's double slit experiment with 632nm laser" \
  --file data.csv
```

#### Evaluate Experimental Design

```bash
python main.py --mode evaluate \
  --input "Measuring spring constant using Hooke's law with masses from 100g to 500g"
```

#### Perform Error Analysis

```bash
python main.py --mode error \
  --input "Measured g = 9.81 +/- 0.15 m/s^2 using pendulum with length L = 1.00 +/- 0.01 m"
```

#### Literature Review Help

```bash
python main.py --mode literature \
  --input "Quantum tunneling in semiconductor devices"
```

### Example Interactive Session

```
You: I measured the resistance of a wire and got R = 12.3 Ohm with a multimeter
     that has +/-0.5% accuracy. How should I express the uncertainty?

QuasarAI: For your resistance measurement with a multimeter having +/-0.5% accuracy:

The absolute uncertainty is:
  dR = 0.5% x 12.3 Ohm = 0.0615 Ohm

Rounding to one significant figure (standard practice):
  dR ~ 0.06 Ohm

Your result should be expressed as:
  R = 12.30 +/- 0.06 Ohm

Note: I kept two decimal places in the measurement to match the precision
of the uncertainty. This follows the convention that your last significant
digit should be in the same decimal place as the uncertainty.

Would you like help with error propagation if you're using this in a
calculation?
```

---

## Advanced Usage

### Saving Conversations

```python
from main import QuasarAI

agent = QuasarAI()
response = agent.chat("Explain the photoelectric effect")
agent.save_conversation("my_session.json")
```

### Loading Previous Sessions

```python
agent = QuasarAI()
agent.load_conversation("my_session.json")
# Continue previous conversation
```

### Custom Analysis Workflows

```python
agent = QuasarAI()

# Analyze data
result = agent.analyze_data(
    "Cavendish experiment for measuring G",
    data_file=Path("cavendish_data.csv")
)

# Follow up with error analysis
error_result = agent.error_analysis(
    "Based on the above analysis, calculate uncertainty in G"
)
```

---

## Example Use Cases

**Experimental Design Review**
> "I'm planning to measure the speed of sound using resonance in a tube. What potential errors should I consider?"

**Data Interpretation**
> "Here's my voltage vs. current data for a diode. Can you help me identify the forward voltage threshold?"

**Error Budget Analysis**
> "I'm measuring the focal length of a lens. Which contributes more uncertainty: the object distance (+/-0.5 cm) or image distance (+/-0.3 cm)?"

**Methodology Critique**
> "I got a value for the electron charge that's 15% off from the accepted value. What went wrong?"

---

## Current Status

**Status:** Active Prototype Development

**Implemented:**
- Claude-powered conversational interface
- Specialized physics system prompt
- Multiple operation modes (analyze, evaluate, error, literature)
- Conversation history management
- Interactive chat mode

**In Progress:**
- Data file parsing and visualization
- Automated error propagation calculations
- Integration with common data formats (CSV, JSON, HDF5)
- Physics-specific tools and plugins

**Planned:**
- Symbolic math and unit checking
- Integration with lab instruments
- Automated lab report evaluation
- Experiment-specific agents (optics, electronics, thermodynamics)
- Web interface

---

## Target Users

- Physics undergraduates and postgraduates
- Experimental physicists
- Research assistants and lab demonstrators
- Educators and science communicators

---

## Contributing

Contributions, ideas, and feedback are welcome!

Areas where help is needed:
- Additional physics-specific analysis tools
- Data visualization modules
- Integration with scientific Python libraries (numpy, scipy, pandas)
- Unit testing and validation
- Documentation improvements

---

## License

MIT License

---

## Acknowledgments

- Powered by [Claude](https://www.anthropic.com/claude) from Anthropic
- Inspired by the needs of physics educators and researchers
- Built with love for experimental physics

---

## Disclaimer

QuasarAI is a tool to **assist** with physics analysis and reasoning. Always:
- Verify critical results independently
- Understand the underlying physics yourself
- Use proper experimental controls and methods
- Consult with supervisors/peers on important research

AI can make mistakes. You are responsible for the scientific validity of your work.
