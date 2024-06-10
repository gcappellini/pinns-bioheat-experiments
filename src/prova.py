import json
import os

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
file_path = os.path.join(src_dir, 'simulations', 'data1.json')

properties = {
    "L0": None, "tauf": None, "k": None, "p0": None, "d": None,
    "rhoc": None, "cb": None, "h": None, "dT": None, "alpha": None,
    "W": None, "steep": None, "tchange": None
}

def get_properties(n):
    global L0, tauf, k, p0, d, rhoc, cb, h, dT, alpha, W, steep, tchange
    file_path = os.path.join(src_dir, 'simulations', f'data{n}.json')

    # Open the file and load the JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)

    properties.update(data['Parameters'])


get_properties(1)
print(properties)