from scripts.Profile import Profile, ProfileOnFloor, find_angle_profile_2d, ProfileLiquidPhase
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob
import os
import time
import argparse


def plot_prof_and_circ(directories):
    fontsize = 22

    profile_full = Profile(slices)
    profile_full.load(path=directories[0], name=f"profile_{scaling}.npy")
    profile_nonzero = Profile(slices)
    profile_nonzero.load(path=directories[0], name=f"nonzero_bulk_profile_{scaling}.npy")
    for dir in directories:
        profile_sobel = Profile(slices)
        # profile_sobel.load(path=dir, name=f"sobel_profile_XY_{scaling}.npy")
        # profile_sobel.fit_profile(dir, add_name="int_fit_sobel", sobel=True, weigth_type="no")
        profile_sobel.load(path=dir, name=f"sobel_profile_XY_{scaling}.npy")
        profile_sobel.fit_profile(dir, add_name="int_fit_sobel", sobel=True, weigth_type="softmax")
        # profile_sobel.load(path=dir, name=f"sobel_profile_XY_{scaling}.npy")
        # profile_sobel.fit_profile(dir, add_name="int_fit_sobel", sobel=True, weigth_type="minmax")

    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    fig.tight_layout()

    ax.set_xlabel('X', fontsize=fontsize)
    ax.set_ylabel('Z', fontsize=fontsize)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_ylim([0, 150])

    ax.pcolormesh(profile_full.profile[:, int(profile_full.slices[1] / 2), :].T, cmap='YlOrRd')
    array = np.array([np.nonzero(profile_nonzero.profile[:, int(profile_nonzero.slices[1] / 2), :])[0],
                      np.nonzero(profile_nonzero.profile[:, int(profile_nonzero.slices[1] / 2), :])[1]])
    print(array)
    ax.plot(array[0], array[1], "o", color="blue", markersize=2, alpha=0.7)
    array = np.array([np.nonzero(profile_sobel.profile[:, int(profile_sobel.slices[1] / 2), :])[0],
                      np.nonzero(profile_sobel.profile[:, int(profile_sobel.slices[1] / 2), :])[1]])
    print(array)
    ax.plot(array[0], array[1], "x", color="green", markersize=2, alpha=0.7)
    circle1 = plt.Circle((75.5, 25.20), 67.96, color='g', fill=False)
    circle2 = plt.Circle((75.4, 22.0), 70.24, color='b', fill=False)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    plt.savefig(os.path.join(directories[0], f'xz_profile_3_prof.png'), bbox_inches='tight', dpi=500)


def plot_profiles(directories):
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    fig.tight_layout()

    fontsize = 22

    ax.set_xlabel('X', fontsize=fontsize)
    ax.set_ylabel('Z', fontsize=fontsize)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_xlim([0, 153])
    ax.set_ylim([0, 150])
    color = ["#fa0a05", "#fa7a05", "#faf505", "#6ddb24", "#50fd02", "#06f9ef", "#095cf6", "#3702fd", "#d405fa",
             "#f807a6", "#fb0474"]

    for num, path_to_profile in enumerate(directories):
        profile = Profile(slices)
        profile_nonzero = Profile(slices)
        profile_nonzero.load(path=path_to_profile, name=f"sobel_profile_XY_{scaling}.npy")
        profile_nonzero.fit_profile(path_to_profile, add_name="int_fit_sobel", sobel=True)
        array = np.array([np.nonzero(profile_nonzero.profile[:, int(profile_nonzero.slices[1] / 2), :])[0] -
                          profile_nonzero.slices[0] / 2 + 77,
                          np.nonzero(profile_nonzero.profile[:, int(profile_nonzero.slices[1] / 2), :])[1]])
        ax.plot(array[0], array[1], "o", color=color[num], markersize=2, alpha=0.7)

    plt.savefig(os.path.join("results_water_N", f'ints_sobel.png'), bbox_inches='tight', dpi=500)


def process_all(directories=[], save_result=""):
    df = pd.DataFrame(columns=['name', 'avg_in_drop_moment',
                               'angle_moment', 'd_angle_moment', 'height_moment', 'radius_moment',
                               'angle_points_prof', 'height_points_prof', 'radius_points_prof',
                               'angle_sobel_no', 'angle_sobel_minmax', 'angle_sobel_softmax',
                               'angle_interface_no', 'angle_interface_minmax', 'angle_interface_softmax'])
    scaling = 1

    for path_to_profile in directories:
        print(path_to_profile)

        # gathering statistics from frame analysis
        df_frame = pd.read_csv(os.path.join(path_to_profile, "num_of_time_5.0_3_full"), sep='\t', index_col=0)
        angle_moment = 180 - np.mean(df_frame["angle"])
        d_angle = np.std(df_frame["angle"])
        N_avg = np.mean(df_frame["in droplet"])/3.0
        height_moment = np.mean(df_frame["height"])
        radius_moment = np.mean(df_frame["radius"])

        # analysing profile
        profile = Profile(np.array([0, 0, 0]))
        profile.load(path=path_to_profile, name=f"profile_{scaling}.npy")
        profile.plot_profile(path=path_to_profile, name="voxel")

        height_points_prof, radius_points_prof, angle_points_prof = profile.find_angle_points_prof(path=path_to_profile, add_name="_points_prof")

        profile.load(path=path_to_profile, name=f"profile_{scaling}.npy")
        profile.process_profile_to_border_density(path=path_to_profile, averaged=True)
        profile.process_profile_sobel(path=path_to_profile, averaged=True)

        profile.load(path=path_to_profile, name=f'border_bulk_profile_{scaling}_{0.6}.npy')
        profile.plot_profile(path=path_to_profile, name=f"border_bulk_profile_{scaling}_{0.6}")
        angle_interface_no = profile.fit_profile(path=path_to_profile, add_name="int_fit", sobel=False, weigth_type="no")
        profile.load(path=path_to_profile, name=f'border_bulk_profile_{scaling}_{0.6}.npy')
        angle_interface_minmax = profile.fit_profile(path=path_to_profile, add_name="int_fit", sobel=False, weigth_type="minmax")
        profile.load(path=path_to_profile, name=f'border_bulk_profile_{scaling}_{0.6}.npy')
        angle_interface_softmax = profile.fit_profile(path=path_to_profile, add_name="int_fit", sobel=False, weigth_type="softmax")

        profile.load(path=path_to_profile, name=f'sobel_profile_XYZ_min_{scaling}.npy')
        profile.plot_profile(path=path_to_profile, name=f"sobel_profile_{scaling}")
        angle_sobel_no = profile.fit_profile(path=path_to_profile, add_name="sobel_fit_XYZ", sobel=True, weigth_type="no")
        profile.plot_profile(path=path_to_profile, name=f"sobel_profile_cut_{scaling}")
        profile.load(path=path_to_profile, name=f'sobel_profile_XYZ_min_{scaling}.npy')
        angle_sobel_minmax = profile.fit_profile(path=path_to_profile, add_name="sobel_fit_XYZ", sobel=True, weigth_type="minmax")
        profile.load(path=path_to_profile, name=f'sobel_profile_XYZ_min_{scaling}.npy')
        angle_sobel_softmax = profile.fit_profile(path=path_to_profile, add_name="sobel_fit_XYZ", sobel=True, weigth_type="softmax")

        profile.load(path=path_to_profile, name=f'sobel_profile_XYZ_{scaling}.npy')
        profile.plot_profile(path=path_to_profile, name=f"sobel_profile_XYZ_{scaling}")

        df.loc[len(df.index)] = [os.path.basename(path_to_profile), N_avg,
                                 angle_moment, d_angle, height_moment, radius_moment,
                                 angle_points_prof, height_points_prof, radius_points_prof,
                                 angle_sobel_no, angle_sobel_minmax, angle_sobel_softmax,
                                 angle_interface_no, angle_interface_minmax, angle_interface_softmax]

    df.to_csv(os.path.join(save_result, f"gathered_data.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Droplets processing module')
    parser.add_argument('-path', '--path-to-profile', default="results_AR_N", type=str,
                        help='path to saved profile in np format')
    args = parser.parse_args()
    directory = args.path_to_profile

    # directories = glob(os.path.join("results_AR_N_2", "*AR_NVT"))
    # directories = glob(os.path.join("results_water_N", "*H2O"))
    # directories = ['results_water_N/5000H2O']
    # directories = ['results_AR_N_2/100AR_NVT', 'results_AR_N_2/1000AR_NVT', 'results_AR_N_2/10000AR_NVT']
    # directories = ['results_AR_N_2/50AR_NVT_ext', 'results_AR_N_2/100AR_NVT', 'results_AR_N_2/200AR_NVT', 'results_AR_N_2/300AR_NVT', 'results_AR_N_2/500AR_NVT', 'results_AR_N_2/1000AR_NVT', 'results_AR_N_2/2000AR_NVT', 'results_AR_N_2/5000AR_NVT', 'results_AR_N_2/10000AR_NVT']
    # directories = ['results_AR_N_2/1000AR_NVT']
    # directories = ['results_water_N/50H2O', 'results_water_N/100H2O', 'results_water_N/200H2O', 'results_water_N/300H2O', 'results_water_N/500H2O', 'results_water_N/1000H2O', 'results_water_N/2000H2O', 'results_water_N/3000H2O', 'results_water_N/4000H2O', 'results_water_N/5000H2O', 'results_water_N/7500H2O']
    directories = ['results/3000AR_NVT_3']

    start_time = time.time()
    # process_all(directories=directories, save_result="results_water_N")
    # profile = Profile(np.array([0, 0, 0]))
    # profile.load(path='results/3000AR_NVT_3', name=f"profile_full_1.npy")
    # profile.plot_profile(path='results/3000AR_NVT_3', name="voxel_full")
    # profile.process_profile_g_of_r(path='results/3000AR_NVT_3', averaged=True)

    df_AR = pd.read_csv("results/3000AR_NVT_3/GofR_full.csv", sep=",")
    df_PA = pd.read_csv("results/part/GofR_full.csv", sep=",")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(df_AR["r"][:100], df_AR["g"][:100], 'b-', markersize=2)
    ax.plot(df_PA["r"][:100], df_PA["g"][:100], 'g-', markersize=2)
    plt.axhline(y=1, color='r', linestyle='-')
    plt.savefig(os.path.join('results/3000AR_NVT_3', f'g(r)_full.png'), bbox_inches='tight')
    fig.tight_layout()
    plt.close()
    print("--- %s seconds ---" % (time.time() - start_time))
