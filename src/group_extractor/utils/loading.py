""" This module contains functions to load data from different sources. """

# Global imports
import csv

# Specific imports
from typing import Dict, Union

def csv_to_dict(filename: str) -> Dict[str, Union[int, float, str]]:
    def convert_value(value: str) -> Union[int, float, str]:
        """Helper function to convert values to the appropriate data type."""
        try:
            # Try to convert to an integer
            if '.' not in value:
                return int(value)
            # Try to convert to a float if it's not an int
            return float(value)
        except ValueError:
            # If it fails to convert, return as string
            return value
        
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        first_row = next(csv_reader)
        return {key: convert_value(value) for key, value in first_row.items()}