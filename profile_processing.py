from scripts.Profile import Profile, ProfileOnFloor
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Droplets processing module')
    parser.add_argument('-path', '--path-to-profile', default="results/8000AR_NVT_bubble", type=str,
                        help='path to saved profile in np format')
    args = parser.parse_args()
    path_to_profile = args.path_to_profile

    start_time = time.time()
    profile = Profile(100)
    profile.load(path_to_profile, 'profile_100.npy')
    profile.plot_profile(path_to_profile)
    profile.process_profile_g_of_r(path_to_profile, averaged=True)
    profile.process_profile_to_border(path_to_profile, averaged=True)
    profile.process_profile_2d(path_to_profile, averaged=True)

    border_profile = Profile(100)
    border_profile.load(path_to_profile, 'border_profile_100.npy')
    border_profile.plot_profile(path_to_profile, name="border")
    border_profile.process_profile_2d(path_to_profile, name="border", averaged=True)

    print("--- %s seconds ---" % (time.time() - start_time))