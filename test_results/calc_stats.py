<<<<<<< HEAD
import os
=======
"""
This script loops through each .pkl in this directory and prints out their mean # of nodes touched when going from
start to target, their std. deviation, and the max. It also
"""

>>>>>>> 31b4ab78ef78770105a9da92b6b88309b8912418
import pickle
import numpy as np
from pathlib import Path

<<<<<<< HEAD
"""
This script loops through each .pkl in this directory and prints out their mean # of nodes touched when going from
start to target, their std. deviation, and the max.
"""


=======
>>>>>>> 31b4ab78ef78770105a9da92b6b88309b8912418
def analyze_pkl_files(directory="."):
    """Analyze all .pkl files in the directory."""
    pkl_files = list(Path(directory).glob("*.pkl"))
    
    if not pkl_files:
        print("No .pkl files found in the directory.")
        return
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            
            # Extract lengths (first element) from each tuple
            if isinstance(data, list):
                ideal_path = [item[0] for item in data if isinstance(item, tuple) and len(item) > 0]

                nodes_touched = [len(item[1]) for item in data if isinstance(item, tuple) and len(item) > 0]
            else:
                continue
            
            if nodes_touched:
<<<<<<< HEAD
=======
                # To get shortest path statistics, we just print out the shortest path for each
                # .pkl file
>>>>>>> 31b4ab78ef78770105a9da92b6b88309b8912418
                ideal_path = np.array(ideal_path)
                mean = np.mean(ideal_path)
                std = np.std(ideal_path)
                max_val = np.max(ideal_path)
                print(f"Shortest path approach in {pkl_file.name}:")
                print(f"  Mean: {mean:.3f}")
                print(f"  Std Dev: {std:.3f}")
                print(f"  Max: {max_val}")
                print()

<<<<<<< HEAD
=======
                # Calculate stats for the given .pkl file
>>>>>>> 31b4ab78ef78770105a9da92b6b88309b8912418
                nodes_touched = np.array(nodes_touched)
                mean = np.mean(nodes_touched)
                std = np.std(nodes_touched)
                max_val = np.max(nodes_touched)
                
                print(f"{pkl_file.name}:")
                print(f"  Mean: {mean:.3f}")
                print(f"  Std Dev: {std:.3f}")
                print(f"  Max: {max_val}")
                print()
        except Exception as e:
            print(f"Error processing {pkl_file.name}: {e}")

if __name__ == "__main__":
    analyze_pkl_files()
