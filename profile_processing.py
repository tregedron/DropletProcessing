from scripts.Profile import Profile, ProfileOnFloor
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Droplets processing module')
    parser.add_argument('-path', '--path-to-profile', default="results/10000AR_NVT", type=str,
                        help='path to saved profile in np format')
    args = parser.parse_args()
    path_to_profile = args.path_to_profile
    start_time = time.time()
    profile = Profile(200)
    profile.load(path_to_profile)
    profile.process_profile_to_border(path_to_profile)

    print("--- %s seconds ---" % (time.time() - start_time))