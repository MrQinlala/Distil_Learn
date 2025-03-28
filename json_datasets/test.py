import pandas as pd
import os

input_dir = "../accountant/"
output_dir = "json_datasets"

data_proj = {
    "val":"",
    "test":"",
    "dev":""
}

for proj_name,real_path in data_proj.items():
    current_dir = os.path.join(input_dir,real_path)
    ds = pd.read_parquet(current_dir)

    save_dir = os.path.join(output_dir,proj_name)
    ds.to_json(save_dir,orient="record",indent = 2)