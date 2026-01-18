import os

def replace_path(src_name, last_path, ext):
                        
    dirname, filename = os.path.split(src_name)
    base, file_extension = os.path.splitext(filename)

                   
    parts = dirname.split(os.path.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = last_path
            break

              
    new_dirname = os.path.sep.join(parts)

            
    new_filename = base + ext

              
    dst_name = os.path.join(new_dirname, new_filename)

    return dst_name

def replace_last_path(path,new_folder_name):
                    
    dirname, basename = os.path.split(path)

                          
    new_path = os.path.join(dirname, new_folder_name)
    return new_path

