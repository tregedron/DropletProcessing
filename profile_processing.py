from scripts.Profile import Profile, ProfileOnFloor, find_angle_profile_2d
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Droplets processing module')
    parser.add_argument('-path', '--path-to-profile', default="results_AR_N", type=str,
                        help='path to saved profile in np format')
    args = parser.parse_args()
    directory = args.path_to_profile

    slices = 100
    directories = glob(os.path.join(directory, "*"))
    print(directories)
    start_time = time.time()
    for path_to_profile in directories:
        profile = Profile(slices)
        try:
            profile.load(path_to_profile, f'profile_{slices}.npy')
            profile.plot_profile(path_to_profile)
            profile.process_profile_g_of_r(path_to_profile, averaged=True)
            profile.process_profile_to_border(path_to_profile, averaged=True)
            profile.process_profile_2d(path_to_profile, averaged=True)

            border_profile = Profile(slices)
            border_profile.load(path_to_profile, f'border_profile_{slices}.npy')
            border_profile.plot_profile(path_to_profile, name="border")
            border_profile.process_profile_2d(path_to_profile, add_name="border", averaged=True)
            find_angle_profile_2d(path_to_profile)
        except:
            print(f"not found profile in {path_to_profile}")

    print("--- %s seconds ---" % (time.time() - start_time))