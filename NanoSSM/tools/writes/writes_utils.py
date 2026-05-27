import uuid
import os

def add_random_suffix(file_path):

    dir_name, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)

    suffix = uuid.uuid4().hex[:8]

    new_name = f"{name}_{suffix}{ext}"
    new_file_path = os.path.join(dir_name, new_name)

    return new_file_path
