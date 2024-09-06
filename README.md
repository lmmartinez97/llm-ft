
# Prompt Factory

## Overview

The Prompt Factory transforms numerical vehicle trajectory data into linguistic prompts designed for high-level cognitive decision-making in autonomous driving systems. These prompts are used as inputs to large language models (LLMs) to guide the ego vehicle's next actions based on surrounding traffic conditions. The project processes historical vehicle data into structured prompts and eventually queries an LLM (like GPT-4) to generate response-answer pairs.

Once generated, these linguistic databases will serve as fine-tuning data for locally-executed LLMs.

## Getting Started

For installation instructions, refer to the [Installation Guide](docs/01_installation.md).

### Usage

Refer to the [User Guide](docs/02_user_guide.md) for instructions on the different tools available
For a high level explanation of the process, see [here](docs/04_how_it_works.md).
The directory structure can be found [here](docs/03_directory_structure.md).

### Contribution

Since this is an internal project, contributions are focused on bug fixing and feature improvements. Feel free to add to the project through new group extractors, prompt templates, or API improvements.
