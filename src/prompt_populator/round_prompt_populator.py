# Setup Warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Global imports
import json
import os
import pandas as pd
import seaborn as sns

# Typing import

from typing import List, Tuple

# Specific imports
from rich import print
from termcolor import cprint
from time import time
from typing import Dict, Tuple, List

sns.set_theme('notebook')
sns.set_style("whitegrid")
sns.set_context("paper")
sns.color_palette("hls", 8)

def print_bl():
    print("\n")

def print_red(*args):
    for arg in args:
        cprint(arg, "red", end=' ')  # Using end=' ' to print all arguments on the same line

def print_green(*args):
    for arg in args:
        cprint(arg, "green", end=' ')  # Using end=' ' to print all arguments on the same line

def print_highlight(*args):
    for arg in args:
        cprint(arg, "magenta", "on_white", end=' ')  # Using end=' ' to print all arguments on the same line

def print_blue(*args):
    for arg in args:
        cprint(arg, "blue", end=' ')  # Using end=' ' to print all arguments on the same line

class GroupsLoader:
    """
    Class to load the groups from the json files.

    Attributes:
        groups_path (str): The path to the groups json files.
        dataset_index (int): The dataset index.
        groups (List[pd.DataFrame]): The list of groups dataframes.
        ego_vehicles (List[int]): The list of ego vehicle IDs.

    Methods:
        load_groups: Load the groups from the json files.
    """

    def __init__(self, groups_path: str, dataset_index: int = 1):
        self.groups_path = groups_path
        self.dataset_index = dataset_index
        self.groups = []
        self.ego_vehicles = []

    def load_groups(self) -> Tuple[List[pd.DataFrame], List[int]]:
        """
        Load the groups from the json files.

        Args:
            None

        Returns:
            A tuple containing the groups and the ego vehicles.
        
        Raises:
            FileNotFoundError: If the groups or ego vehicles file does not exist.
            ValueError: If the JSON content is invalid or cannot be processed.
        """
        try:
            with open(f"{self.groups_path}/groups_{self.dataset_index}.json", 'r') as f:
                data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Groups file not found: {e.filename}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from groups file: {e.msg}")

        try:
            self.groups = [pd.read_json(data[f]) for f in data.keys()]
        except ValueError as e:
            raise ValueError(f"Error reading JSON data into DataFrame: {e}")

        try:
            with open(f"{self.groups_path}/ego_vehicles_{self.dataset_index}.json", 'r') as f:
                self.ego_vehicles = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Ego vehicles file not found: {e.filename}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from ego vehicles file: {e}")

        return self.groups, self.ego_vehicles
    
class RounDPromptPopulator:
    """
    Class to populate the prompts.

    Takes the groups and ego vehicles and populates the prompt template. Saves the prompts in a JSON array.

    Attributes:
        groups_location (str): The path to the groups JSON files.
        template_path (str): The path to the template files.
        dataset_index (int): The dataset index.
        prompts (List[str]): The list of populated prompt strings.
    """

    def __init__(self, groups_location: str, template_path: str, dataset_index: int = 1):
        self.groups_location = groups_location
        self.template_path = template_path
        self.dataset_index = dataset_index
        self.prompts = []

        ### Error handling for loading groups and templates
        try:
            self.groups_loader = GroupsLoader(self.groups_location, self.dataset_index)
            self.groups, self.ego_vehicles = self.groups_loader.load_groups()
        except Exception as e:
            raise RuntimeError(f"Failed to load groups and ego vehicles: {e}")
        try:
            self.instructions_template, self.task_template, self.role_template, self.answer_template = self.load_templates()
        except Exception as e:
            raise RuntimeError(f"Failed to load templates: {e}")

    def load_templates(self) -> List[str]:
        """
        Load the prompt templates.

        Returns:
            A list containing the instructions, task, and role templates.
        
        Raises:
            FileNotFoundError: If a template file is not found.
            IOError: If there is an error reading a template file.
        """
        ### Error handling for loading templates
        try:
            with open(f"{self.template_path}/instructions_template.txt", 'r') as f:
                instructions_template = f.read()
            with open(f"{self.template_path}/task_template.txt", 'r') as f:
                task_template = f.read()
            with open(f"{self.template_path}/role_template.txt", 'r') as f:
                role_template = f.read()
            with open(f"{self.template_path}/answer_template.txt", 'r') as f:
                answer_template = f.read()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Template file not found: {e.filename}")
        except IOError as e:
            raise IOError(f"Error reading template file: {e}")

        return instructions_template, task_template, role_template, answer_template
    
    def relative_group(self, group_index: int, frame_tame_diff_ms: int = 400) -> pd.DataFrame:
        """
        Transforms frame numbering into temporal relative frame numbering.

        Args:
            group_index (int): The index of the group to transform.
            frame_tame_diff_ms (int): The time difference between frames in milliseconds.

        Returns:
            A DataFrame containing the transformed group.
        
        Raises:
            IndexError: If the group_index is out of range.
            KeyError: If expected keys are not found in the DataFrame.
        """
        ### Error handling for group index
        if group_index >= len(self.groups):
            raise IndexError(f"Group index {group_index} out of range.")

        group = self.groups[group_index]

        group['frame'] = (-group['frame'].max() + group['frame']) * frame_tame_diff_ms / 1000

        return group
    
    def row2str(self, row: pd.Series) -> str:
        """
        Convert a row to a string.

        Args:
            row (pd.Series): The row to convert.

        Returns:
            A string containing the row information.
        
        Raises:
            KeyError: If expected keys are not found in the row.
        """
        info = (
            f"At t={row['frame']} s, vehicle with id {row['trackId']} is at position ({row['xCenter_new']:.2f}, {row['yCenter_new']:.2f}) with heading {row['heading']:.2f} degrees."
            f"It's x axis speed is {row['xVelocity_new']:.2f} m/s and y axis speed is {row['yVelocity_new']:.2f} m/s. It's x axis cceleration is {row['xAcceleration_new']:.2f} m/s^2 "
            f"and y axis acceleration is {row['yAcceleration_new']:.2f} m/s^2. The width of the vehicle is {row['width']:.2f} m and its length is "
            f"{row['length']:.2f} m."
        )

        return info
    
    def get_prompt_static_info(self, group_index: int = 0) -> str:
        """
        Get the static information for the prompt.

        Args:
            group_index (int): The index of the group to get the static information.

        Returns:
            A string containing the static information for the prompt.
        
        Raises:
            KeyError: If expected keys are not found in the DataFrame.
        """
        group = self.relative_group(group_index)
        ego_vehicle_id = list(self.ego_vehicles.keys())[group_index]

        vehicle_ids = ", ".join([str(id) for id in group['trackId'].unique()])
        info = f"Vehicles present in the group have ids: {vehicle_ids}. "
        info += f"The ego vehicle is vehicle with id {ego_vehicle_id}."

        return info

    def populate_prompt(self, group_index: int = 0) -> List[Dict[str, str]]:
        """
        Populate the prompts. Takes a template and a vehicle group and returns a list of dictionaries with populated prompts.

        Args:
            group_index (int): The index of the group to populate.

        Returns:
            list: A list containing dictionaries with roles 'system' and 'user' and their corresponding content.
        
        Raises:
            ValueError: If the group DataFrame is empty.
        """
        if group_index >= len(self.groups):
            raise IndexError(f"Group index {group_index} out of range.")
        
        group = self.relative_group(group_index)
        
        if group.empty:
            raise ValueError(f"No data found for group index {group_index}.")
            
        frame_groups = group.groupby('frame')

        # Generate system prompt
        system_prompt = f"{self.role_template}\n\n{self.instructions_template}\n\n{self.answer_template}\n\n"

        # Generate user prompt
        user_prompt = (
            f"{self.get_prompt_static_info(group_index)}\n\n"
            "The information for each vehicle is as follows:\n"
        )
        
        for frame, frame_group in frame_groups:
            for index, row in frame_group.iterrows():
                user_prompt += f"{self.row2str(row)}\n"

        # Add task information
        user_prompt += f"\n{self.task_template}"

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        return message
    
    def save_prompt_to_file(self, group_index: int = 0, filename: str = 'prompt.txt'):
        """
        Save the populated prompt to a file. Only for testing purposes. Main saving function is NOT this one.

        Args:
            group_index (int): The index of the group to populate.
            filename (str): The name of the file to save the prompt.
        
        Raises:
            IOError: If there is an error writing to the file.
        """
        ### Error handling for group index
        if group_index >= len(self.groups):
            raise IndexError(f"Group index {group_index} out of range.")
        
        ### Error handling for writing to file
        try:
            prompt = self.populate_prompt(group_index)
            print(f"Saving prompt to file {filename}...")
            print(f"System prompt:\n{prompt[0]['content']}")
            print(f"User prompt:\n{prompt[1]['content']}")

            with open(filename, 'w') as file:
                for message in prompt:
                    file.write(f"{message['content']}\n")
        except IOError as e:
            raise IOError(f"Error writing to file {filename}: {e}")
        
    def save_top_20_longest_prompts(self, filename: str = 'top_20_longest_prompts.json'):
        """
        Generate all prompts, find the 20 longest, and save them to a file.

        Args:
            filename (str): The name of the file to save the longest prompts.
        
        Raises:
            IOError: If there is an error writing to the file.
        """
        all_prompts = []

        for group_index in range(len(self.groups)):
            try:
                prompt = self.populate_prompt(group_index)
                combined_prompt = f"{prompt[0]['content']}\n{prompt[1]['content']}"
                all_prompts.append({
                    'group_index': group_index,
                    'length': len(combined_prompt),
                    'prompt': prompt
                })
            except Exception as e:
                print(f"Error generating prompt for group {group_index}: {e}")

        # Sort prompts by length in descending order and select the top 20
        top_20_prompts = sorted(all_prompts, key=lambda x: x['length'], reverse=True)[:20]

        # Extract the relevant information for saving
        top_20_prompts_content = [
            {
                'group_index': item['group_index'],
                'prompt': item['prompt']
            }
            for item in top_20_prompts
        ]

        # Save the top 20 longest prompts to a JSON file
        try:
            with open(filename, 'w') as file:
                json.dump(top_20_prompts_content, file, indent=4)
        except IOError as e:
            raise IOError(f"Error writing to file {filename}: {e}")

    def generate_and_save_all_prompts(self, folder: str = None):
        """
        Generate messages with system and user content for all groups and save them to a JSON file.

        Args:
            filename (str): Folder where the prompts will be saved.
        
        Raises:
            IOError: If there is an error writing to the file.
        """
        all_prompts = []

        for group_index in range(len(self.groups)):
            ### Error handling for generating prompt
            try:
                prompt = self.populate_prompt(group_index)
                #combine user and system prompts
                prompt = f"{prompt[0]['content']}\n{prompt[1]['content']}"
                all_prompts.append(prompt)
            except Exception as e:
                print(f"Error generating prompt for group {group_index}: {e}")
                

        filename = folder + f"/prompts_{self.dataset_index}.json"
        try:
            with open(filename, 'w') as file:
                json.dump(all_prompts, file, indent=4)
        except IOError as e:
            raise IOError(f"Error writing to file {filename}: {e}")

def main():
    if os.path.exists("/Users/lmiguelmartinez/Tesis/datasets/rounD/data/"):
        groups_location = "/Users/lmiguelmartinez/Tesis/llm-ft/data/rounD/groups_200"
        prompts_destination = "/Users/lmiguelmartinez/Tesis/llm-ft/data/rounD/prompts_200"
        template_path = "/Users/lmiguelmartinez/Tesis/llm-ft/src/prompt_populator/templates/rounD_templates"
    else:
        groups_location = "/home/lmmartinez/Tesis/datasets/rounD/groups_200"
        prompts_destination = "/home/lmmartinez/Tesis/llm-ft/data/rounD/prompts_200"
        template_path = "/home/lmmartinez/Tesis/llm-ft/src/prompt_populator/templates/rounD_templates"
    print(f"Loading groups from {groups_location}. \n")

    for i in range(1,23):
        print_blue(f"Generating prompts for dataset {i}... \n")
        prompt_populator = RounDPromptPopulator(groups_location=groups_location, template_path=template_path, dataset_index=i)
        prompt_populator.generate_and_save_all_prompts(folder=prompts_destination)
        print_green(f"Prompts for dataset {i} generated successfully. Saved at {prompts_destination}.\n")
        #print_green(f"Top 20 longest prompts for dataset {i} saved at {generator_destination}.\n")

if __name__ == "__main__":
    main()