from scripts.Cluster import Cluster
from scripts.Droplet import Droplet, DropletOnFloor
from scripts.Profile import Profile, ProfileOnFloor
from scripts.RDF_module import RDF
from collections import Counter
import numpy as np
from chemfiles import Topology, Frame, Atom, UnitCell, Trajectory, Residue, Selection
import pandas as pd
from tqdm import tqdm
import os
import time
import argparse


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

    if selection is not None:
        selection = Selection(selection)

    df = pd.DataFrame(columns=['time', 'in droplet'])

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


def process_part_droplet(trj_path,
                         topol_path,
                         cutoff,
                         neighbours,
                         selection=None,
                         pbc_z=True,
                         path_to_results="results"):
    print("Working on trajectory", trj_path)
    print("topology: ", topol_path)
    print("Cut-off: ", cutoff, " neighbours: ", neighbours)

    relaxation_finished = 15000000

    path_out_dir = os.path.join(path_to_results, trj_path.split("/")[-1].split(".")[0])
    os.makedirs(path_out_dir, exist_ok=True)

    with Trajectory(topol_path) as trajectory_gro:
        frame = trajectory_gro.read()
        box = np.ceil(frame.cell.lengths).astype(int)

    profile_droplet = Profile(slices=box, scaling=1)
    profile_AR = Profile(slices=box, scaling=1)
    profile_part = Profile(slices=box, scaling=1)
    destrib_function_all = RDF(mode="point")
    destrib_function_drop = RDF(mode="point")

    if selection is not None:
        selection = Selection(selection)

    selection_part = Selection("name PA")
    N_in_particle = len(selection_part.evaluate(frame))

    df = pd.DataFrame(columns=['time', 'in droplet'])

    with Trajectory(trj_path) as trajectory:
        trajectory.set_topology(topol_path)
        for frame in tqdm(trajectory):
            if frame.step > relaxation_finished:
                droplet_cluster = Cluster(frame, selection, cutoff=cutoff, neighbours=neighbours)
                particle_cluster = Cluster(frame, selection_part, cutoff=cutoff, neighbours=neighbours)

                droplet_cluster.cluster_frame(pbc_z=pbc_z)
                droplet_cluster.clustering = np.array(droplet_cluster.clustering)

                particle_cluster.cluster_frame(pbc_z=pbc_z)
                particle_cluster.clustering = np.array(particle_cluster.clustering)
                if N_in_particle != np.count_nonzero(particle_cluster.clustering == 0):
                    print(f"Hmmm, particle divided {N_in_particle}, {np.count_nonzero(particle_cluster.clustering == 0)}")


                particle = Droplet(frame, selection_part.evaluate(frame))
                particle.calculate_mass_center()
                mass_center_particle = particle.mass_center
                profile_part.update_profile(particle)

                full_AR = Droplet(frame, selection.evaluate(frame))
                full_AR.calculate_mass_center()
                profile_AR.update_profile(full_AR)

                counter = dict(Counter(droplet_cluster.clustering))

                counter.pop(-1)
                if len(counter.keys()) >= 1:
                    biggest_ind = max(counter, key=counter.get)

                    droplet = Droplet(frame, np.where(droplet_cluster.clustering == biggest_ind)[0])
                    droplet.calculate_mass_center()
                    mass_center_droplet = droplet.mass_center

                    profile_droplet.update_profile(droplet)

                    df.loc[len(df.index)] = [frame.step, droplet.size]

                destrib_function_all.update_rdf_point(frame, mass_center_particle, selection.evaluate(frame))
                destrib_function_drop.update_rdf_point(frame, mass_center_particle, np.where(droplet_cluster.clustering == biggest_ind)[0])

    profile_droplet.save(path_out_dir, name="profile_droplet")
    profile_part.save(path_out_dir, name="profile_part")
    profile_AR.save(path_out_dir, name="profile_full_AR")
    df.to_csv(os.path.join(path_out_dir, f"num_of_time_{cutoff}_{neighbours}_full"), sep='\t')

    destrib_function_all.normalize_rdf(path_out_dir, add_name="full")
    destrib_function_drop.normalize_rdf(path_out_dir, add_name="drop")


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

    process_part_droplet("data/AR_PA/NVT.xtc", "data/AR_PA/NVT.gro", 7, 10, selection="name AR",
                         path_to_results="results")
