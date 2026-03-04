# Network Science Lab Course

A hands-on lab course exploring complex networks through interactive Python notebooks. Over six weeks, you will build, analyze, and visualize networks — learning how real-world systems like social networks, the internet, and biological networks organize themselves and behave.

## Quick Start (GitHub Codespaces)

1. Click the green **Code** button at the top of this repository.
2. Select the **Codespaces** tab.
3. Click **Create codespace on main**.
4. Wait for the environment to finish setting up (this takes a few minutes on first launch).
5. Open `01-intro/lab.ipynb` and run the first code cell to verify your environment.

Everything is configured automatically. No installation required.

## Quick Start (Local Setup)

**Prerequisites:** Python 3.12 or later, [uv](https://docs.astral.sh/uv/) installed.

```bash
git clone <repository-url>
cd <repository-name>
uv sync
```

To start working with notebooks, either:

- Open the repository in VS Code (with the Jupyter extension installed), or
- Run `uv run jupyter lab` to launch JupyterLab in your browser.

## How to Open and Run Notebooks

1. Open any `.ipynb` file in your editor or JupyterLab.
2. Select the Python kernel if prompted (it should be auto-detected from the virtual environment).
3. Run cells individually with **Shift+Enter**, or run all cells from the **Run** menu.
4. Start with `01-intro/lab.ipynb` and run the smoke test cell to confirm your environment works.

## How to Submit Assignments

*[Submission instructions will be provided by your instructor.]*

## Course Structure

| Week | Folder | Topic |
|------|--------|-------|
| 1 | `01-intro/` | Introduction to Networks |
| 2 | `02-properties/` | Network Properties and Measures |
| 3 | `03-small-worlds/` | Small-World Networks |
| 4 | `04-models-hubs/` | Network Models and Hubs |
| 5 | `05-communities/` | Community Detection |
| 6 | `06-dynamics/` | Dynamics on Networks |

Each week folder contains a `lab.ipynb` (guided lab exercises) and an `assignment.ipynb` (graded assignment).

## References

- _Networks, Crowds, and Markets: Reasoning about a Highly Connected World_ — David Easley & Jon Kleinberg
- _Network Science_ — Albert-László Barabási ([networksciencebook.com](https://www.networksciencebook.com/))
- _A First Course in Network Science_ — Menczer, Fortunato & Davis ([book site](https://cambridgeuniversitypress.github.io/FirstCourseNetworkScience/) | [GitHub](https://github.com/CambridgeUniversityPress/FirstCourseNetworkScience))
- [Network Science practice assignments](https://github.com/netspractice/network-science) — assignment model and inspiration
- [netsci-labs repository](https://github.com/zademn/netsci-labs) — earlier lab material

## Getting Help

*[Contact information will be provided by your instructor.]*
