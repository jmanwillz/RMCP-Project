# RMCP Project

This repository serves as the codebase for the project titled "Learning Discrete Representations From Mass Spectrometry Imaging Using VQ-VAE".

The project is supervised by [Dr. Devon Jarvis](https://jarvisdevon.github.io/).

Please refer to the following documents:

1. [Research Proposal](./documentation/Research_Propsal.pdf)
2. [Project Report](./documentation/Project_Report.pdf)

## Setting up the python virtual environment

1. Create the virtual environment

```bash
python -m venv .venv
```

2. Activate the environment

```bash
source .venv/bin/activate
```

3. Install the requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Install the project as a package

```bash
pip install -e .
```

## Useful Commands

1. Save the libraries to the requirements.txt

```bash
pip freeze > requirements.txt
```
