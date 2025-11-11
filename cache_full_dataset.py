#!/usr/bin/env python3
"""
Load the full Wikipedia dataset once and save as pickle files for fast reloading.
This only needs to be run ONCE.

Usage: python cache_full_dataset.py
"""

import pickle
from util import load_full_dataset_from_folder

if __name__ == '__main__':
    print("Loading full Wikipedia dataset (this will take ~5-10 minutes)...")
    print("This only needs to be done ONCE.\n")

    name_to_id, id_to_name, edge_dict = load_full_dataset_from_folder()

    print("\nSaving to pickle files...")
    with open('pickles/edge_dict_full.pkl', 'wb') as f:
        pickle.dump(edge_dict, f)
    with open('pickles/id_to_name_full.pkl', 'wb') as f:
        pickle.dump(id_to_name, f)
    with open('pickles/name_to_id_full.pkl', 'wb') as f:
        pickle.dump(name_to_id, f)

    print("\n✅ Done! Full dataset cached.")
    print(f"   {len(id_to_name):,} articles")
    print(f"   {sum(len(v) for v in edge_dict.values()):,} edges")
    print("\nNext time you run baseline.py, it will load instantly from pickles!")
