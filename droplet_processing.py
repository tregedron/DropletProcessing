from scripts.Cluster import Cluster
from scripts.Droplet import Droplet, DropletOnFloor
from scripts.Profile import Profile, ProfileOnFloor
from collections import Counter
import numpy as np
from chemfiles import Topology, Frame, Atom, UnitCell, Trajectory, Residue, Selection
import pandas as pd
from tqdm import tqdm
import os
import time
import argparse


def process_droplet_pbc_angle_only(trj_path,
                                   topol_path,
                                   cutoff,
                                   neighbours,
                                   selection=None,
                                   pbc_z=False,
                                   path_to_results="results"):
    print("Working on trajectory", trj_path)
    print("topology: ", topol_path)
    print("Cut-off: ", cutoff, " neighbours: ", neighbours)

    path_out_dir = os.path.join(path_to_results, trj_path.split("/")[-1].split(".")[0])
    os.makedirs(path_out_dir, exist_ok=True)

    df = pd.DataFrame(columns=['time', 'in droplet', 'height', 'radius', 'angle'])
    if selection is not None:
        selection = Selection(selection)

    with Trajectory(trj_path) as trajectory:
        trajectory.set_topology(topol_path)
        for frame in tqdm(trajectory):
            cluster = Cluster(frame, selection, cutoff=cutoff, neighbours=neighbours)
            cluster.cluster_frame(pbc_z=pbc_z)
            cluster.clustering = np.array(cluster.clustering)
            number_of_clusters = len(set(cluster.clustering)) - 1
            if number_of_clusters == 1:
                droplet = DropletOnFloor(frame, np.where(cluster.clustering != -1)[0])
                droplet.calculate_mass_center()
                droplet.calc_h_r()
                droplet.find_alpha()
                df.loc[len(df.index)] = [frame.step, droplet.size, droplet.height, droplet.radius, droplet.alpha]

    df.to_csv(os.path.join(path_out_dir, f"num_of_time_{cutoff}_{neighbours}_full"), sep='\t')


def process_droplet_pbc_no_xyz(trj_path,
                               topol_path,
                               cutoff,
                               neighbours,
                               selection=None,
                               pbc_z=False,
                               path_to_results="results"):

    print("Working on trajectory", trj_path)
    print("topology: ", topol_path)
    print("Cut-off: ", cutoff, " neighbours: ", neighbours)

    path_out_dir = os.path.join(path_to_results, trj_path.split("/")[-1].split(".")[0])
    os.makedirs(path_out_dir, exist_ok=True)

    with Trajectory(topol_path) as trajectory_gro:
        frame = trajectory_gro.read()
        box = np.ceil(frame.cell.lengths).astype(int)

    profile = Profile(slices=box, scaling=1)
    print(profile.slices)

    df = pd.DataFrame(columns=['time', 'in droplet', 'height', 'radius', 'angle'])
    if selection is not None:
        selection = Selection(selection)

    with Trajectory(trj_path) as trajectory:
        trajectory.set_topology(topol_path)
        for frame in tqdm(trajectory):
            if frame.step > 500000:
                cluster = Cluster(frame, selection, cutoff=cutoff, neighbours=neighbours)
                cluster.cluster_frame(pbc_z=pbc_z)
                cluster.clustering = np.array(cluster.clustering)
                number_of_clusters = len(set(cluster.clustering)) - 1
                if number_of_clusters == 1:
                    droplet = DropletOnFloor(frame, np.where(cluster.clustering != -1)[0])
                    droplet.calculate_mass_center()
                    droplet.calc_h_r()
                    droplet.find_alpha()

                    profile.update_profile(droplet)

                    df.loc[len(df.index)] = [frame.step, droplet.size, droplet.height, droplet.radius, droplet.alpha]

    profile.save(path_out_dir)
    df.to_csv(os.path.join(path_out_dir, f"num_of_time_{cutoff}_{neighbours}_full"), sep='\t')


def process_droplet_pbc_full_info(trj_path,
                                  topol_path,
                                  cutoff,
                                  neighbours,
                                  selection=None,
                                  pbc_z=False,
                                  path_to_results="results"):
    print("Working on trajectory", trj_path)
    print("topology: ", topol_path)
    print("Cut-off: ", cutoff, " neighbours: ", neighbours)

    path_out_dir = os.path.join(path_to_results, trj_path.split("/")[-1].split(".")[0])
    os.makedirs(path_out_dir, exist_ok=True)

    with Trajectory(topol_path) as trajectory_gro:
        frame = trajectory_gro.read()
        box = np.ceil(frame.cell.lengths).astype(int)

    profile = Profile(slices=box, scaling=1)
    print("profile now: ", profile.slices)

    df = pd.DataFrame(columns=['time', 'in droplet', 'height', 'radius', 'angle'])
    if selection is not None:
        selection = Selection(selection)

    with Trajectory(trj_path) as trajectory:
        with Trajectory(os.path.join(path_out_dir, f"clustered_{cutoff}_{neighbours}_full.xyz"),
                        "w") as outtrajectory:
            trajectory.set_topology(topol_path)

            for frame in tqdm(trajectory):
                if frame.step > 500000:
                    outframe = Frame()

                    cluster = Cluster(frame, selection, cutoff=cutoff, neighbours=neighbours)
                    cluster.cluster_frame(pbc_z=pbc_z)
                    cluster.clustering = np.array(cluster.clustering)

                    number_of_clusters = len(set(cluster.clustering)) - 1
                    if number_of_clusters == 1:
                        for atom_ind, cluster_num in enumerate(cluster.clustering):
                            if cluster_num == -1:
                                outframe.add_atom(Atom(f"Gaz"), [0, 0, 0])
                            else:
                                outframe.add_atom(Atom(f"H2O_{cluster_num}"), frame.positions[atom_ind])
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


def process_free_droplet(trj_path,
                         topol_path,
                         cutoff,
                         neighbours,
                         selection=None,
                         pbc_z=True,
                         path_to_results="results"):
    print("Working on trajectory", trj_path)
    print("topology: ", topol_path)
    print("Cut-off: ", cutoff, " neighbours: ", neighbours)

    path_out_dir = os.path.join(path_to_results, trj_path.split("/")[-1].split(".")[0])
    os.makedirs(path_out_dir, exist_ok=True)

    with Trajectory(topol_path) as trajectory_gro:
        frame = trajectory_gro.read()
        box = np.ceil(frame.cell.lengths).astype(int)

    profile = Profile(slices=box, scaling=1)

    df = pd.DataFrame(columns=['time', 'in droplet'])
    if selection is not None:
        selection = Selection(selection)

    with Trajectory(trj_path) as trajectory:
        trajectory.set_topology(topol_path)
        for frame in tqdm(trajectory):
            if frame.step > 2500000:
                cluster = Cluster(frame, selection, cutoff=cutoff, neighbours=neighbours)
                cluster.cluster_frame(pbc_z=pbc_z)
                cluster.clustering = np.array(cluster.clustering)

                counter = dict(Counter(cluster.clustering))

                counter.pop(-1)
                if len(counter.keys()) >= 1:
                    biggest_ind = max(counter, key=counter.get)

                    droplet = Droplet(frame, np.where(cluster.clustering == biggest_ind)[0])
                    droplet.calculate_mass_center()
                    mass_center = droplet.mass_center

                    droplet = Droplet(frame, np.where(cluster.clustering != -2)[0], mass_center=mass_center)

                    profile.update_profile(droplet)
                    df.loc[len(df.index)] = [frame.step, droplet.size]

    profile.save(path_out_dir, name="profile_full")
    df.to_csv(os.path.join(path_out_dir, f"num_of_time_{cutoff}_{neighbours}_full"), sep='\t')


def process_equil_info(trj_path,
                       topol_path,
                       cutoff,
                       neighbours,
                       selection=None,
                       pbc_z=False,
                       path_to_results="results"):
    print("Working on trajectory", trj_path)
    print("topology: ", topol_path)
    print("Cut-off: ", cutoff, " neighbours: ", neighbours)

    path_out_dir = os.path.join(path_to_results, trj_path.split("/")[-1].split(".")[0])
    os.makedirs(path_out_dir, exist_ok=True)

    df = pd.DataFrame(columns=['time', 'number_of_clusters', 'number_in_gas'])
    if selection is not None:
        selection = Selection(selection)

    with Trajectory(trj_path) as trajectory:
        trajectory.set_topology(topol_path)

        for frame in tqdm(trajectory):

            cluster = Cluster(frame, selection, cutoff=cutoff, neighbours=neighbours)
            cluster.cluster_frame(pbc_z=pbc_z)
            cluster.clustering = np.array(cluster.clustering)

            number_of_clusters = len(set(cluster.clustering)) - 1
            number_in_gas = cluster.clustering[cluster.clustering == -1].shape[0]

            df.loc[len(df.index)] = [frame.step, number_of_clusters, number_in_gas]

    df.to_csv(os.path.join(path_out_dir, f"equil_{cutoff}_{neighbours}"), sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Droplets processing module')
    parser.add_argument('-trj', '--trajectory', default=None, type=str,
                        help='trajectory file in xtc format (or not in xtc)')
    parser.add_argument('-top', '--topology', default=None, type=str,
                        help='topology file in gro format')
    parser.add_argument('-cf', '--cutoff', default=5, type=float,
                        help='cutoff in DBSCAN clusterization')
    parser.add_argument('-nn', '--neighbours', default=3, type=int,
                        help='neighbours in DBSCAN clusterization')
    parser.add_argument('-res', '--path-to-results', default="results", type=str,
                        help='path to directory to store results')
    args = parser.parse_args()

    start_time = time.time()

    path_to_trj = args.trajectory
    path_to_top = args.topology

    # path_to_trj = os.path.join("data", "NVT_0_4.xtc")
    # path_to_top = "data/10000AR_NVT.gro"
    #
    # if path_to_top is None:
    #     path_to_top = path_to_trj.split(".")[0] + ".gro"

    # # water
    # process_droplet_pbc_no_xyz(trj_path=path_to_trj,
    #                               topol_path=path_to_top,
    #                               cutoff=args.cutoff,
    #                               neighbours=args.neighbours,
    #                               selection="name OH2",
    #                               pbc_z=True,
    #                               path_to_results=args.path_to_results)
    process_free_droplet("data/3000AR_NVT_3.xtc", "data/3000AR_NVT_3.gro", 7, 10, selection=None, path_to_results="results")
    # argon
    # process_droplet_pbc_no_xyz(trj_path=path_to_trj,
    #                            topol_path=path_to_top,
    #                            cutoff=args.cutoff,
    #                            neighbours=args.neighbours,
    #                            pbc_z=False,
    #                            path_to_results=args.path_to_results)

    print("--- %s seconds ---" % (time.time() - start_time))
