
# Prompt Factory

## Overview

The Prompt Factory transforms numerical vehicle trajectory data into linguistic prompts designed for high-level cognitive decision-making in autonomous driving systems. These prompts are used as inputs to large language models (LLMs) to guide the ego vehicle's next actions based on surrounding traffic conditions. The project processes historical vehicle data into structured prompts and eventually queries an LLM (like GPT-4) to generate response-answer pairs.

Once generated, these linguistic databases will serve as fine-tuning data for locally-executed LLMs.

## Directory Structure

### 1. **data/**

Contains input and output data related to vehicle trajectories, surrounding traffic, and processed prompts. It is both input and output of the data pipeline.

### 2. **src/**

Contains the core processing scripts for extracting vehicle groups, populating prompts, and integrating with OpenAI's API.

- **group_extractor/**:
  - `highd_group_extractor.py`: Extracts groups of vehicles from the highD dataset.
  - `read_csv.py`: A utility script to read CSV files that contain vehicle trajectory data.
- **prompt_population/**:
  - `highd_prompt_populator.py`: Populates predefined templates with vehicle group data, transforming numerical data into prompts using groups from the highD dataset.
  - **templates/**: A folder containing prompt templates like:
    - `answer_template.txt`: Template that tells the model the expected answer structure.
    - `instructions_template.txt`: Template for prompt instructions.
    - `task_template.txt`: Template for the task that the LLM needs to solve.
    - `role_template.txt`: Template to define the knowledge history of the LLM.

### 3. dev/

This directory is dedicated to small-scale development work and prototyping. Jupyter notebooks are placed here for experimental and incremental development before they are finalized and ported to the src folder.

- **group_extractor/**:
  - `highd_group_extractor.ipynb`: Jupyter notebook for developing vehicle group extraction methods on the highD dataset.
  - `round_group_extractor.ipynb`: Jupyter notebook for developing vehicle group extraction methods on the roundD dataset.
- **prompt_populator/**:
  - `highd_prompt_populator.ipynb`: Jupyter notebook version for developing prompt population methods based on highD datasets.
  - **templates/**:
    - **highD_templates/**: Templates specifically designed for highD dataset prompts.
    - **rounD_templates/**: Templates aimed at roundabout scenarios (currently under development).
- **prompt_wizard/**:
  - `batch_generator.ipynb`: Jupyter notebook for generating multiple prompts in batch mode.
  - `openai_api_test.ipynb`: Jupyter notebook for testing API integration with GPT-4 (gpt4o) to generate response-answer pairs.
- **test_save/**: Contains temporary and validation data used during development for testing purposes.

### 3. **docs/**

Intended for future documentation. Currently unavailable.

### 4. **tests/**

Directory for test cases and validation scripts. Not implemented yet.

## How It Works

1. **Group Extraction**:
  The system ingests raw vehicle trajectory data and processes it into groups. These groups represent the ego vehicle's surroundings within a window of configurable length using the `lookback` parameter. The output is a structured JSON format that details the historical trajectories of both the ego vehicle and surrounding vehicles.

2. **Prompt Population**:
  Once the vehicle groups are generated, the `prompt_populator.py` script uses predefined templates to convert the numerical data into linguistic prompts. The templates define various prompt types, including instructions, roles, tasks, and expected answers.

3. **Batch Generation and API Integration**:
  Under development.

4. **Testing and Validation**:
  Small-scale test runs are stored in the `test_save/` directory, providing quick validation of the process before larger runs are initiated.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required libraries as listed in `requirements.txt`

### Installation

1. Clone the repository:

  ```bash
    git clone https://github.com/lmmartinez97/llm-ft.git
  ```

2. Navigate to the project directory:

  ```bash
    cd llm-ft
  ```
  
3. Install the dependencies:

  ```bash
    pip install -r requirements.txt
  ```

### Usage

1. **Group Extraction**: 
   Use `highd_group_extractor.py` or `round_group_extractor.ipynb` to extract vehicle group data from raw trajectory datasets.

2. **Prompt Population**:
   Run `prompt_populator.py` to transform the extracted group data into linguistic prompts.

3. **Batch Generation**: 
   Use `batch_generator.ipynb` to generate a batch of prompts.

4. **API Testing**:
   Utilize `openai_api_test.ipynb` to query the OpenAI API with the generated prompts (gpt4o model).

### Contribution

Since this is an internal project, contributions are focused on bug fixing and feature improvements. Feel free to add to the project through new group extractors, prompt templates, or API improvements.
