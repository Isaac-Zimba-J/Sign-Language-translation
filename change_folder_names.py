import os

# Generate list of numbers from 0 to 25 (since you want A = 0, B = 1, ..., Z = 25)
num_list = [i for i in range(26)]
# Generate list of letters from A to Z
letter_list = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Directory containing the folders to be renamed
dir_path = './train'

# Create a dictionary to map letters to their corresponding numbers
letter_to_num = {letter_list[i]: num_list[i] for i in range(len(letter_list))}

# Get list of folders in the directory
folders = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]

# Sort the list of folders alphabetically
folders.sort()

# Iterate through the sorted list of folders and rename them
for folder_name in folders:
    # Ensure the name starts with a letter
    if folder_name[0].upper() in letter_to_num:
        # Get the first character of the folder name and convert to uppercase
        first_char = folder_name[0].upper()
        # Get the corresponding number for the letter
        new_name = str(letter_to_num[first_char])
        
        # Create the full path for the old and new folder names
        old_folder_path = os.path.join(dir_path, folder_name)
        new_folder_path = os.path.join(dir_path, new_name)
        
        # Rename the folder
        os.rename(old_folder_path, new_folder_path)
        print(f"Renamed {old_folder_path} to {new_folder_path}")
    else:
        print(f"Skipping {folder_name}, it does not start with an alphabet letter or is not a directory.")
