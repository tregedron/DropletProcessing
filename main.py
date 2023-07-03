from scripts.Cluster import Cluster
from scripts.Droplet import Droplet
import numpy as np
from chemfiles import Topology, Frame, Atom, UnitCell, Trajectory, Residue
import pandas as pd
from tqdm import tqdm
import os
import time
import argparse


def process_trajectory(trj_path, topol_path, cutoff, neighbours):
    print("Working on trajectory", trj_path)
    print("topology: ", topol_path)
    print("Cut-off: ", cutoff, " neighbours: ", neighbours)

    path_out_dir = os.path.join("results", trj_path.split("/")[-1].split(".")[0])
    os.makedirs(path_out_dir, exist_ok=True)

    df = pd.DataFrame(columns=['time', 'number'])
    selection = None

    with Trajectory(trj_path) as trajectory:
        with Trajectory(os.path.join(path_out_dir, f"clustered_{cutoff}_{neighbours}_full.xyz"), "w") as outtrajectory:
            trajectory.set_topology(topol_path)

            for frame in tqdm(trajectory):
                outframe = Frame()

                cluster = Cluster(frame, selection, cutoff=cutoff, neighbours=neighbours)
                cluster.cluster_frame()

                number_of_clusters = len(set(cluster.clustering)) - 1
                cluster_indexes = []
                for cluster_id in range(number_of_clusters):
                    cluster_indexes.append([])

                for atom_ind, cluster in enumerate(cluster.clustering):
                    if cluster == -1:
                        outframe.add_atom(Atom(f"Gaz"), frame.positions[atom_ind])
                    else:
                        outframe.add_atom(Atom(f"AR{cluster}"), frame.positions[atom_ind])
                        cluster_indexes[cluster].append(atom_ind)

                outtrajectory.write(outframe)

                df.loc[len(df.index)] = [frame.step, number_of_clusters-1]


    df.to_csv(os.path.join(path_out_dir, f"num_of_time_{cutoff}_{neighbours}_full"), sep='\t')

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
    args = parser.parse_args()

    start_time = time.time()

    trj_temp = os.path.join("data", "5000AR_NVT.xtc")
    top_temp = os.path.join("data", "5000AR_NVT.gro")

    process_trajectory(trj_temp, top_temp, args.cutoff, args.neighbours)
    # for cut_off_iter in range(1, 8, 2):
    #     for neihbours_iter in range(1, 21, 5):
    #         process_trajectory(trj_temp, top_temp, cut_off_iter, neihbours_iter)

    print("--- %s seconds ---" % (time.time() - start_time))
