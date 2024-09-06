import os
import random

from highd_group_extractor import HighDGroupExtractor

def process_file(dataset_location: str, dataset_index: int, sampling_period: float, 
                 lookback_window: float, save_path: str, show_plots: bool) -> None:
    """
    Processes a single file from the dataset.

    Args:
        dataset_location (str): Path to the dataset directory.
        dataset_index (int): Index of the dataset file to process (between 1 and 60).
        sampling_period (float): Sampling period in milliseconds.
        lookback_window (float): Lookback window size in seconds.
        save_path (str): Directory to save the output JSON files.
        show_plots (bool): Whether to generate plots or animations.

    Returns:
        None
    """
    print(f"Processing file {dataset_index} in {dataset_location}...")

    # Create an instance of the extractor with the provided dataset location and index
    extractor = HighDGroupExtractor(dataset_location=dataset_location, dataset_index=dataset_index)

    # Apply filtering and windowing
    extractor.filter_data(sampling_period=sampling_period)
    extractor.get_frame_windows(window_size=lookback_window)
    extractor.get_ego_vehicles()
    extractor.get_groups()

    # Save the result as JSON files
    extractor.save_groups(save_path=save_path)

    # Plotting or animations loop
    if show_plots:
        num_groups = len(extractor.groups)
        if num_groups == 0:
            print("No groups available for plotting or animating.")
            return
        while True:
            plot_choice = get_valid_input("Do you want to plot or animate a random group? (Enter 'plot', 'animate', or 'exit'): ", ["plot", "animate", "exit"])
            if plot_choice == "exit":
                break
            random_group = random.randint(0, num_groups - 1)  # Randomly select a group number
            if plot_choice == "plot":
                # Plot a random group
                print(f"Plotting group {random_group}...")
                extractor.plot_groups(group_num=random_group, save=False)
            elif plot_choice == "animate":
                # Animate a random group
                print(f"Animating group {random_group}...")
                extractor.animate_groups(group_num=random_group, save=False)

def process_dataset(dataset_location: str, sampling_period: float, lookback_window: float, save_path: str) -> None:
    """
    Processes the entire dataset by iterating over files from index 1 to 60.

    Args:
        dataset_location (str): Path to the dataset directory.
        sampling_period (float): Sampling period in milliseconds.
        lookback_window (float): Lookback window size in seconds.
        save_path (str): Directory to save the output JSON files.

    Returns:
        None
    """
    for dataset_index in range(1, 61):  # Assuming dataset indices from 1 to 60
        process_file(dataset_location, dataset_index, sampling_period, lookback_window, save_path, False)

def get_valid_input(prompt: str, valid_options: list[str]) -> str:
    """
    Prompts the user for input and checks if the response is valid.

    Args:
        prompt (str): The question to ask the user.
        valid_options (list[str]): A list of valid options the user can input.

    Returns:
        str: The valid input provided by the user.
    """
    while True:
        user_input = input(prompt).strip().lower()
        if user_input == "exit":
            print("Exiting the program.")
            exit(0)
        if user_input in valid_options:
            return user_input
        print(f"Invalid input. Please enter one of the following: {', '.join(valid_options)}.")

def main() -> None:
    """
    Main function that runs the CLI for the program.

    The program allows the user to process vehicle trajectory data, either processing the entire dataset or a single file.
    The user can provide a dataset location, sampling period, lookback window, and specify whether to generate plots or animations.

    Returns:
        None
    """
    while True:
        # Step 1: Ask for dataset location
        dataset_location = input("Enter the path to the dataset directory: ").strip()
        if dataset_location == "exit":
            print("Exiting the program.")
            exit(0)
        if not os.path.exists(dataset_location):
            print(f"Invalid path: {dataset_location}. Please provide a valid directory.")
            continue

        # Step 2: Ask for sampling period
        while True:
            sampling_period_input = input("Enter the sampling period (in milliseconds): ").strip().lower()
            if sampling_period_input == "exit":
                print("Exiting the program.")
                exit(0)
            try:
                sampling_period = float(sampling_period_input)
                if sampling_period <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Please enter a valid positive number for the sampling period.")

        # Step 3: Ask for lookback window
        while True:
            lookback_window_input = input("Enter the lookback window (in seconds): ").strip().lower()
            if lookback_window_input == "exit":
                print("Exiting the program.")
                exit(0)
            try:
                lookback_window = float(lookback_window_input)
                if lookback_window <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Please enter a valid positive number for the lookback window.")

        # Step 4: Ask whether to process the entire dataset or one file
        process_choice = get_valid_input("Do you want to process the entire dataset or a single file? (Enter 'all' or 'one'): ", ["all", "one", "exit"])

        if process_choice == 'all':
            # Step 5: Ask for location to save the JSON files
            while True:
                save_path = input("Enter the directory to save the JSON files: ").strip()
                if save_path == "exit":
                    print("Exiting the program.")
                    exit(0)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                process_dataset(dataset_location, sampling_period, lookback_window, save_path)
                break
        
        elif process_choice == 'one':
            # Step 6: Ask for the dataset index (file number)
            while True:
                dataset_index_input = input("Enter the dataset index (between 1 and 60): ").strip().lower()
                if dataset_index_input == "exit":
                    print("Exiting the program.")
                    exit(0)
                try:
                    dataset_index = int(dataset_index_input)
                    if 1 <= dataset_index <= 60:
                        break
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Please enter a number between 1 and 60.")

            # Step 7: Ask for location to save the JSON file
            while True:
                save_path = input("Enter the directory to save the JSON file: ").strip()
                if save_path == "exit":
                    print("Exiting the program.")
                    exit(0)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                process_file(dataset_location, dataset_index, sampling_period, lookback_window, save_path, show_plots())
                break
        
        # Step 8: Ask if the user wants to process more data
        repeat = get_valid_input("Do you want to process more data? (yes/no): ", ["yes", "no", "exit"])
        if repeat == "no":
            print("Exiting the program.")
            break

def show_plots() -> bool:
    """
    Asks the user whether to show plots or not.

    Returns:
        bool: True if the user wants to see plots, False otherwise.
    """
    return get_valid_input("Do you want to generate plots or animations? (yes/no): ", ["yes", "no", "exit"]) == "yes"

if __name__ == "__main__":
    main()
