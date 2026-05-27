import sys
import os
import re
import glob

from NanoSSM.tools.common.common_utils import common_log, ERROR

def read_file_lines(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            return lines
    except FileNotFoundError:
        raise FileNotFoundError(f"file not found: {file_path}")
    except Exception as e:
        common_log(f"Error reading file {file_path}: {e}", ERROR)
        raise e

def get_methyl_files(pattern, directory):
    try:

        methyl_files = glob.glob(os.path.join(directory, pattern))

        file_names = [os.path.basename(file) for file in methyl_files]
        common_log(f">> Match methyl files: {file_names}\n")
        return file_names
    except Exception as e:
        common_log(f"Error getting methyl files: {e}\n", ERROR)
        raise e
