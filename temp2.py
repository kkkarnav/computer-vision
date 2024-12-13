import os

# Define the directory and output file
root_dir = "D:/code/computer-vision/active/mosaics"
output_file = "D:/code/computer-vision/active/mosaic_files.txt"

# Open the output file for writing
with open(output_file, "w") as file:
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".tif"):
                # Write the full path of the .tif file
                file.write(f"{dirpath}/{filename}\n")

print(f"Paths written to {output_file}")
