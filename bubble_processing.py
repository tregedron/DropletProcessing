from scripts.Cluster import Cluster
from scripts.LiquidPhase import LiquidPhase
from scripts.Profile import Profile, ProfileOnFloor, ProfileLiquidPhase
import numpy as np
from chemfiles import Topology, Frame, Atom, UnitCell, Trajectory, Residue
import pandas as pd
from tqdm import tqdm
import os
import time
import argparse


def process_trajectory_pbc(trj_path, topol_path):
    print("Working on trajectory", trj_path)
    print("topology: ", topol_path)

    path_out_dir = os.path.join("results", trj_path.split("/")[-1].split(".")[0])
    os.makedirs(path_out_dir, exist_ok=True)

    profile = ProfileLiquidPhase(100)

    df = pd.DataFrame(columns=['time', 'in liquid'])
    selection = None

    with Trajectory(trj_path) as trajectory:
        trajectory.set_topology(topol_path)

        for frame in tqdm(trajectory):
            if frame.step > 300000:
                liquid_phase = LiquidPhase(frame)
                liquid_phase.calculate_mass_center()
                profile.update_profile(liquid_phase)
                df.loc[len(df.index)] = [frame.step, liquid_phase.size]

    profile.save(path_out_dir)
    df.to_csv(os.path.join(path_out_dir, f"num_of_time"), sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Droplets processing module')
    parser.add_argument('-trj', '--trajectory', default=None, type=str,
                        help='trajectory file in xtc format (or not in xtc)')
    parser.add_argument('-top', '--topology', default=None, type=str,
                        help='topology file in gro format')
    parser.add_argument('-cf', '--cutoff', default=5, type=float,
                        help='cutoff in DBSCAN clusterization')
    parser.add_argument('-nn', '--neighbours', default=10, type=int,
                        help='neighbours in DBSCAN clusterization')
    parser.add_argument('-res', '--path-to-results', default="results", type=str,
                        help='path to directory to store results')
    args = parser.parse_args()

    start_time = time.time()

    # trj_temp = os.path.join("data", "8000AR_NVT_bubble.xtc")
    # top_temp = os.path.join("data", "8000AR_NVT_bubble.gro")

    path_to_trj = args.trajectory
    # trj_path = os.path.join("data", "NVT_0_4.xtc")
    path_to_top = args.topology
    if path_to_top is None:
        path_to_top = path_to_trj.split(".")[0] + ".gro"

    process_trajectory_pbc(path_to_trj, path_to_top)
    print("--- %s seconds ---" % (time.time() - start_time))