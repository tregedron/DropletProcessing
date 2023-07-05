from scripts.Cluster import Cluster
from scripts.Droplet import Droplet, DropletOnFloor
from scripts.Profile import Profile, ProfileOnFloor
import numpy as np
from chemfiles import Topology, Frame, Atom, UnitCell, Trajectory, Residue
import pandas as pd
from tqdm import tqdm
import os
import time
import argparse


def process_trajectory_angle(trj_path, topol_path, cutoff, neighbours):
    print("Working on trajectory", trj_path)
    print("topology: ", topol_path)
    print("Cut-off: ", cutoff, " neighbours: ", neighbours)

    path_out_dir = os.path.join("results", trj_path.split("/")[-1].split(".")[0])
    os.makedirs(path_out_dir, exist_ok=True)

    df = pd.DataFrame(columns=['time', 'in droplet', 'height', 'radius', 'angle'])
    selection = None

    with Trajectory(trj_path) as trajectory:
        trajectory.set_topology(topol_path)
        for frame in tqdm(trajectory):
            cluster = Cluster(frame, selection, cutoff=cutoff, neighbours=neighbours)
            cluster.cluster_frame(pbc_z=False)
            number_of_clusters = len(set(cluster.clustering)) - 1
            if number_of_clusters == 1:
                droplet = DropletOnFloor(frame, np.where(cluster.clustering != -1)[0])
                droplet.calculate_mass_center()
                droplet.calc_h_r()
                droplet.find_alpha()
                df.loc[len(df.index)] = [frame.step, droplet.size, droplet.height, droplet.radius, droplet.alpha]

    df.to_csv(os.path.join(path_out_dir, f"num_of_time_{cutoff}_{neighbours}_full"), sep='\t')


def process_trajectory_full(trj_path, topol_path, cutoff, neighbours):
    print("Working on trajectory", trj_path)
    print("topology: ", topol_path)
    print("Cut-off: ", cutoff, " neighbours: ", neighbours)

    path_out_dir = os.path.join("results", trj_path.split("/")[-1].split(".")[0])
    os.makedirs(path_out_dir, exist_ok=True)

    profile = Profile(200)

    df = pd.DataFrame(columns=['time', 'in droplet', 'height', 'radius', 'angle'])
    selection = None

    with Trajectory(trj_path) as trajectory:
        with Trajectory(os.path.join(path_out_dir, f"clustered_{cutoff}_{neighbours}_full.xyz"),
                        "w") as outtrajectory:
            trajectory.set_topology(topol_path)

            for frame in tqdm(trajectory):
                if frame.step % 1000 == 0:
                    outframe = Frame()

                    cluster = Cluster(frame, selection, cutoff=cutoff, neighbours=neighbours)
                    cluster.cluster_frame(pbc_z=False)

                    number_of_clusters = len(set(cluster.clustering)) - 1
                    if number_of_clusters == 1:
                        for atom_ind, cluster_num in enumerate(cluster.clustering):
                            if cluster_num == -1:
                                outframe.add_atom(Atom(f"Gaz"), [0, 0, 0])
                            else:
                                outframe.add_atom(Atom(f"AR{cluster_num}"), frame.positions[atom_ind])

                        droplet = DropletOnFloor(frame, np.where(cluster.clustering != -1)[0])
                        droplet.calculate_mass_center()
                        droplet.calc_h_r()
                        droplet.find_alpha()
                        outframe.add_atom(Atom(f"Mass"), droplet.mass_center)
                        outframe.add_atom(Atom(f"Summ"),
                                          np.array([droplet.mass_center[0],
                                                    droplet.mass_center[1],
                                                    droplet.floor+droplet.height]))
                        outframe.add_atom(Atom(f"Floo"),
                                          np.array([droplet.mass_center[0], droplet.mass_center[1], droplet.floor]))
                        outframe.add_atom(Atom(f"Rad"),
                                          np.array([droplet.mass_center[0]+droplet.radius, droplet.mass_center[1], droplet.floor]))
                        outframe.add_atom(Atom(f"Rad"),
                                          np.array([droplet.mass_center[0] - droplet.radius, droplet.mass_center[1],
                                                    droplet.floor]))
                        outframe.add_atom(Atom(f"Rad"),
                                          np.array([droplet.mass_center[0], droplet.mass_center[1] + droplet.radius,
                                                    droplet.floor]))
                        outframe.add_atom(Atom(f"Rad"),
                                          np.array([droplet.mass_center[0], droplet.mass_center[1] - droplet.radius,
                                                    droplet.floor]))

                        outtrajectory.write(outframe)

                        profile.update_profile(droplet)

                        df.loc[len(df.index)] = [frame.step, droplet.size, droplet.height, droplet.radius, droplet.alpha]

    profile.save(path_out_dir)
    df.to_csv(os.path.join(path_out_dir, f"num_of_time_{cutoff}_{neighbours}_full"), sep='\t')

def process_trajectory_full_pbc(trj_path, topol_path, cutoff, neighbours):
    print("Working on trajectory", trj_path)
    print("topology: ", topol_path)
    print("Cut-off: ", cutoff, " neighbours: ", neighbours)

    path_out_dir = os.path.join("results", trj_path.split("/")[-1].split(".")[0])
    os.makedirs(path_out_dir, exist_ok=True)

    profile = Profile(100)

    df = pd.DataFrame(columns=['time', 'in droplet'])
    selection = None

    with Trajectory(trj_path) as trajectory:
        with Trajectory(os.path.join(path_out_dir, f"clustered_{cutoff}_{neighbours}_full.xyz"),
                        "w") as outtrajectory:
            trajectory.set_topology(topol_path)

            for frame in tqdm(trajectory):
                if frame.step > 1000000:
                    outframe = Frame()

                    print("clustering")

                    cluster = Cluster(frame, selection, cutoff=cutoff, neighbours=neighbours)
                    cluster.cluster_frame(pbc_z=True)

                    number_of_clusters = len(set(cluster.clustering)) - 1

                    print(number_of_clusters)

                    if number_of_clusters == 1:
                        for atom_ind, cluster_num in enumerate(cluster.clustering):
                            if cluster_num == -1:
                                outframe.add_atom(Atom(f"Gaz"), [0, 0, 0])
                            else:
                                outframe.add_atom(Atom(f"AR{cluster_num}"), frame.positions[atom_ind])

                            droplet = Droplet(frame, np.where(cluster.clustering != -1)[0])
                            droplet.calculate_mass_center()
                            outframe.add_atom(Atom(f"Mass"), droplet.mass_center)
                            outtrajectory.write(outframe)
                            profile.update_profile(droplet)
                            df.loc[len(df.index)] = [frame.step, droplet.size]

    profile.save(path_out_dir)
    df.to_csv(os.path.join(path_out_dir, f"num_of_time_{cutoff}_{neighbours}_full"), sep='\t')


def create_profile(trj_path, topol_path, cutoff, neighbours):
    print("Working on trajectory", trj_path)
    print("topology: ", topol_path)
    print("Cut-off: ", cutoff, " neighbours: ", neighbours)

    path_out_dir = os.path.join("results", trj_path.split("/")[-1].split(".")[0])
    os.makedirs(path_out_dir, exist_ok=True)

    profile = Profile(200)
    df = pd.DataFrame(columns=['time', 'in droplet', 'height', 'radius', 'angle'])
    selection = None

    with Trajectory(trj_path) as trajectory:
        trajectory.set_topology(topol_path)
        for frame in tqdm(trajectory):
            cluster = Cluster(frame, selection, cutoff=cutoff, neighbours=neighbours)
            cluster.cluster_frame(pbc_z=False)
            number_of_clusters = len(set(cluster.clustering)) - 1
            if number_of_clusters == 1:
                droplet = DropletOnFloor(frame, np.where(cluster.clustering != -1)[0])
                droplet.calculate_mass_center()
                profile.update_profile(droplet)
                df.loc[len(df.index)] = [frame.step, droplet.size, droplet.height, droplet.radius, droplet.alpha]

    profile.save(path_out_dir)

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

    trj_temp = os.path.join("data", "10000AR_NVT.xtc")
    top_temp = os.path.join("data", "10000AR_NVT.gro")

    process_trajectory_full(trj_temp, top_temp, args.cutoff, args.neighbours)
    print("--- %s seconds ---" % (time.time() - start_time))
