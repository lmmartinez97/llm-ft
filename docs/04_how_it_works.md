## How It Works

1. **Group Extraction**:
  The system ingests raw vehicle trajectory data and processes it into groups. These groups represent the ego vehicle's surroundings within a window of configurable length using the `lookback` parameter. The output is a structured JSON format that details the historical trajectories of both the ego vehicle and surrounding vehicles.

2. **Prompt Population**:
  Once the vehicle groups are generated, the `prompt_populator.py` script uses predefined templates to convert the numerical data into linguistic prompts. The templates define various prompt types, including instructions, roles, tasks, and expected answers.

3. **Batch Generation and API Integration**:
  Under development.

4. **Testing and Validation**:
  Small-scale test runs are stored in the `test_save/` directory, providing quick validation of the process before larger runs are initiated.