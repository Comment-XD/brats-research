import glob
import numpy as np
import os
import json

print("\n"+ os.getcwd())
ROOT_PATH = f"{os.getcwd()}/dataset/brats2020/MICCAI_BraTS2020_TrainingData"

def preprocess(root_path:str, shuffle:bool=True) -> dict[str]:
    """
    
    Converts the Brats dataset into a Json format

    Args:
        root_path (str): the root path of the original dataset
        split (str, optional): . Defaults to "train".

    Returns:
        _type_: Dictionary[data paths]
    """

    json_file = {"train": []}   
    data_paths = glob.glob(f"BraTS20_Train*", root_dir=root_path)

    if shuffle:
        np.random.shuffle(data_paths)

    for data_path in data_paths:
        
        val = np.random.rand()

        data_dir = os.path.join(root_path + "/", data_path)

        image_paths = glob.glob("*[!g].nii*", root_dir=data_dir)
        label_paths = glob.glob("*seg.nii*", root_dir=data_dir) 
        
        data_dict = {}
        data_dir = data_dir.split("/")[-1]
        print(data_dir)
        if val > 0.8:
            data_dict = {"image": [os.path.join(data_dir + "/", img_path) for img_path in image_paths], 
                         "label": os.path.join(data_dir + "/", *label_paths), 
                         "fold": 0}
        else:
            data_dict = {"image": [os.path.join(data_dir + "/", img_path) for img_path in image_paths], 
                         "label": os.path.join(data_dir + "/", *label_paths)}
            
        json_file["train"].append(data_dict)
        
    return json_file

dataset0 = open("dataset_0.json", "w")
json_file = preprocess(ROOT_PATH)

json.dump(json_file, dataset0, indent=4)
    