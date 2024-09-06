# Installation Guide

This guide will walk you through the installation and setup process for the Prompt Factory project. The system is designed to transform vehicle trajectory data into linguistic prompts and query large language models (LLMs) to guide the next actions of an ego vehicle.

## Prerequisites

Before installing the project, ensure that you have the following software and tools installed:

- **Python 3.x**: Ensure you have Python 3.8 or higher installed. Recommended version is 3.11.x.
- **Conda**: If you're not using Conda, it’s recommended for managing environments.
- **Jupyter Notebook**: For running development notebooks in the `dev/` folder.
- **Git**: To clone the repository.

## Step-by-Step Installation

### 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone https://github.com/lmmartinez97/llm-ft.git
cd llm-ft
```

### 2. Set Up the Conda Environment

If you are using Conda, you can create a new environment from the `environment-file.yml` file included in the repository.

#### Option 1: Create from `environment-file.yml`

1. Create the environment from the file:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:

   ```bash
   conda activate prompts
   ```

#### Option 2: Manually Set Up the Environment

Alternatively, you can manually create a Conda environment and install dependencies using `requirements.txt`:

1. Create and activate the Conda environment:

   ```bash
   conda create -n prompts python=3.11
   conda activate prompts
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### 3. Install Jupyter Notebook (If Not Already Installed)

To run the development notebooks in the `dev/` folder, make sure you have Jupyter Notebook installed:

```bash
conda install jupyter
```

Alternatively, using pip:

```bash
pip install notebook
```

### 4. Set Up OpenAI API Access

If you're planning to test API integration, you'll need to set up an OpenAI API key:

1. Visit [OpenAI's API page](https://beta.openai.com/signup/) to sign up or log in.
2. Once logged in, generate an API key under the **API** section.
3. Store the key in an environment variable for use in notebooks:

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

   You can add this to your `.bashrc` or `.zshrc` file for persistent access.

### 5. Verify the Installation

To ensure everything is set up correctly, you can run the following command to check that the required packages are installed:

```bash
conda list
```

Alternatively, run the following in the Python interpreter to check if key dependencies are available:

```python
import openai
import pandas as pd
import numpy as np
```

## Running the Project

Once everything is installed:

1. **Group Extraction**: You can run group extractors by navigating to the `src/group_extractor/` folder and executing the scripts or Jupyter notebooks as needed.
   Example:

   ```bash
   python src/group_extractor/highd_group_extractor.py
   ```

2. **Prompt Population**: Generate linguistic prompts by running the `highd_prompt_populator.py` in the `src/prompt_population/` folder.

3. **Testing API Integration**: Test the OpenAI API connection using `openai_api_test.ipynb` located in the `dev/prompt_wizard/` folder.

---

That’s it! You should now be set up to use the Prompt Factory project. If you encounter any issues during installation, refer to the [Troubleshooting Guide](docs/troubleshooting.md).
