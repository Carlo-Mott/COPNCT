import os
def filelister(path, with_path=1, file_extension=".csv", avoid_files=None, startswith=None):
    """
    List all csv files in a directory as a list of strings.
    """
    files = []
    for file in os.listdir(path):
        if startswith and not file.startswith(startswith):
            continue
        if avoid_files and file.endswith(avoid_files):
            continue
        if file.endswith(file_extension):
            if with_path:
                files.append(os.path.join(path, file))
            else:
                files.append(file)
    return files
        
    