
# Prompt Factory

## Overview

The Prompt Factory transforms numerical vehicle trajectory data into linguistic prompts designed for high-level cognitive decision-making in autonomous driving systems. These prompts are used as inputs to large language models (LLMs) to guide the ego vehicle's next actions based on surrounding traffic conditions. The project processes historical vehicle data into structured prompts and eventually queries an LLM (like GPT-4) to generate response-answer pairs.

Once generated, these linguistic databases will serve as fine-tuning data for locally-executed LLMs.

## Directory Structure

### 1. **data/**

Contains input and output data related to vehicle trajectories, surrounding traffic, and processed prompts. It is both input and output of the data pipeline.

### 2. **src/**

Contains the core processing scripts for extracting vehicle groups, populating prompts, and integrating with OpenAI's API.

- **group_extraction/**: 
  - `highd_group_extractor.py`: Extracts groups of vehicles from the highD dataset, based on 5-second windows.
  - `round_group_extractor.ipynb`: Extracts data specifically related to roundabout scenarios.
  - `read_csv.py`: A utility script to read CSV files that contain vehicle trajectory data.
- **prompt_population/**:
  - `prompt_populator.py`: Populates predefined templates with vehicle group data, transforming numerical data into prompts.
- **prompts/**: A folder containing prompt templates like:
  - `answer_template.txt`: Template for expected answers.
  - `instructions_template.txt`: Template for prompt instructions.
  - `task_template.txt`: Template for tasks associated with prompt generation.
- **prompt_wizard/**: 
  - `batch_generator.ipynb`: Facilitates batch generation of prompts.
  - `openai_api_test.ipynb`: A notebook for testing the OpenAI API with the GPT-4 model (gpt4o), allowing you to generate response-answer pairs based on the created prompts.
- **test_save/**: Contains small-scale test runs for validation purposes.

### 3. **docs/**

Intended for future documentation.

### 4. **tests/**

Directory for test cases and validation scripts.

## How It Works

1. **Group Extraction**:
  The system ingests raw vehicle trajectory data (stored elsewhere) and processes it into groups. These groups represent the ego vehicle's surroundings within a 5-second window (`lookback5`). The output is a structured JSON format that details the historical trajectories of both the ego vehicle and surrounding vehicles.

2. **Prompt Population**:
  Once the vehicle groups are generated, the `prompt_populator.py` script uses predefined templates to convert the numerical data into linguistic prompts. The templates define various prompt types, including instructions, roles, tasks, and expected answers.

3. **Batch Generation and API Integration**:
  Using `batch_generator.ipynb`, multiple prompts can be generated at once. The `openai_api_test.ipynb` notebook then integrates with the OpenAI GPT-4 model, querying it with the generated prompts to obtain response-answer pairs. This feature is currently under development.

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
