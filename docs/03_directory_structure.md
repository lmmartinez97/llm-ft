# Directory Structure

### 1. **data/**

Contains input and output data related to vehicle trajectories, surrounding traffic, and processed prompts. It is both input and output of the data pipeline.

### 2. **src/**

Contains the core processing scripts for extracting vehicle groups, populating prompts, and integrating with OpenAI's API.

- **group_extractor/**:
  - `highd_group_extractor.py`: Extracts groups of vehicles from the highD dataset.
  - `highd_cli_processor.py`: Guides the user through the extration process on the highD dataset with a command-line interactive interface. Allows visualization of extracted groups.
  - `read_csv.py`: A utility script to read CSV files that contain vehicle trajectory data.
- **prompt_population/**:
  - `highd_prompt_populator.py`: Populates predefined templates with vehicle group data, transforming numerical data into prompts using groups from the highD dataset.
  - **templates/**: A folder containing prompt templates like:
    - `answer_template.txt`: Template that tells the model the expected answer structure.
    - `instructions_template.txt`: Template for prompt instructions.
    - `task_template.txt`: Template for the task that the LLM needs to solve.
    - `role_template.txt`: Template to define the knowledge history of the LLM.

### 3. **dev/**

This directory is dedicated to small-scale development work and prototyping. Jupyter notebooks are placed here for experimental and incremental development before they are finalized and ported to the `src/` folder.

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

### 4. **docs/**

Refer to the [documentation index](00_docs_index.md) for content.
This directory will also house further documentation, such as API references, developer guides, and tutorials.

### 5. **tests/**

Directory for test cases and validation scripts. Not implemented yet.
