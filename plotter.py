import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FixedLocator)
import re

def process_angle_distribution(dr_path):
    print(dr_path)
    out_dir = dr_path.split("/")[:-1]
    out_dir = os.path.join(*out_dir)
    df_frame = pd.read_csv(dr_path, sep='\t', index_col=0)
    # print(df_frame)
    angle = np.mean(df_frame["angle"])
    d_angle = np.std(df_frame["angle"])
    N_avg=np.mean(df_frame["in droplet"])
    print(angle, d_angle, N_avg)
    return angle, d_angle, N_avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-path', '--path-to-results', default="results_water_N", type=str,
                        help='path to results directory')
    args = parser.parse_args()
    path_to_results = args.path_to_results

    # names = []
    # N = []
    # N_drop_avg = []
    # angle_list = []
    # d_angle_list = []
    #
    # for dirpath, dirnames, filenames in os.walk(os.path.join(path_to_results)):
    #     for filename in [f for f in filenames if "num_of_time" in f and ".pdf" not in f and ".png" not in f]:
    #         file = os.path.join(dirpath, filename)
    #         names.append(file.split("/")[-2])
    #         N.append(int(re.findall(r'\d+', file.split("/")[-2])[0]))
    #         angle_d_angle = process_angle_distribution(file)
    #         angle_list.append(angle_d_angle[0])
    #         d_angle_list.append(angle_d_angle[1])
    #         N_drop_avg.append(angle_d_angle[2])

    # df = pd.DataFrame({'Name': names, "N": N, "N in droplet" : N_drop_avg, 'alpha': angle_list, "d_alpha": d_angle_list})
    # df.to_csv(os.path.join(path_to_results, "angle_in_Systems.csv"), sep='\t')

    # df = pd.read_csv(os.path.join(path_to_results, "angle_in_Systems.csv"), sep=",")
    df = pd.read_csv("results_water_N/gathered_data.csv", sep=",")

    print(df)

    # font_size = 22
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111)
    # ax.set_ylabel(r"$\alpha$", fontsize=font_size)
    # ax.set_xlabel("$\it{N}$", fontsize=font_size)
    # plt.xscale('log')
    # # ax.set_xlim([10, 12000])
    # lower_lim = np.floor(np.min(180 - df["alpha"] - df["d_alpha"]) / 10)
    # lower_lim = (lower_lim - 1) * 10
    # higher_lim = np.ceil(np.max(180 - df["alpha"] + df["d_alpha"]) / 10)
    # higher_lim = (higher_lim+1) * 10
    # ax.set_ylim([lower_lim, higher_lim])
    #
    # tick_array = [50, 100, 200, 300, 500, 1000, 2000, 5000, 10000]
    # labels = [x for x in tick_array]
    # plt.xticks(ticks=tick_array, labels=labels, fontsize=10)
    #
    # tick_array_y = np.arange(lower_lim, higher_lim + 10, 10)
    # labels_y = [x for x in tick_array_y]
    # plt.yticks(ticks=tick_array_y, labels=labels_y, fontsize=18)
    # ax.yaxis.set_minor_locator(FixedLocator(list(np.arange(lower_lim + 5, higher_lim + 15, 10))))
    #
    # ax.errorbar(df["N in droplet"], 180 - df["alpha"], df["d_alpha"], fmt='o', color="black", label='wetting angle', markersize=0,
    #             linewidth=1, capsize=3)
    # ax.plot(df["N in droplet"], 180 - df["alpha"], "o", color="green", label='wetting angle', markersize=7)
    # ax.plot(df["N in droplet"], df["alpha fit"], "x", color="purple", label='wetting angle', markersize=7)
    # ax.plot(df["N in droplet"], df["alpha exp"], "x", color="yellow", label='wetting angle', markersize=7)
    # ax.plot(df["N in droplet"], df["alpha minmax"], "x", color="red", label='wetting angle', markersize=7)
    #
    # plt.savefig(os.path.join(path_to_results, 'alpha_of_N_water_true.png'), bbox_inches='tight', dpi=400)
    # fig.tight_layout()
    # plt.close()

    def linear(x, a, b):  # this is your 'straight line' y=f(x)
        return a * x + b

    font_size = 22
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_ylabel(r"$\alpha$", fontsize=font_size)
    ax.set_xlabel("$\it{N}$", fontsize=font_size)
    plt.xscale('log')
    # ax.set_xlim([10, 12000])
    lower_lim = np.floor(np.min(df["angle_moment"] - df["d_angle_moment"]) / 10)
    lower_lim = (lower_lim - 1) * 10
    higher_lim = np.ceil(np.max(df["angle_moment"] + df["d_angle_moment"]) / 10)
    higher_lim = (higher_lim+1) * 10
    ax.set_ylim([lower_lim, higher_lim])

    tick_array = [50, 100, 200, 300, 500, 1000, 2000, 5000, 10000]
    labels = [x for x in tick_array]
    plt.xticks(ticks=tick_array, labels=labels, fontsize=10)

    tick_array_y = np.arange(lower_lim, higher_lim + 10, 10)
    labels_y = [x for x in tick_array_y]
    plt.yticks(ticks=tick_array_y, labels=labels_y, fontsize=18)
    ax.yaxis.set_minor_locator(FixedLocator(list(np.arange(lower_lim + 5, higher_lim + 15, 10))))

    ax.errorbar(df["avg_in_drop_moment"], df["angle_moment"], df["d_angle_moment"], fmt='o', color="black", markersize=0, linewidth=1, capsize=3)
    ax.plot(df["avg_in_drop_moment"], df["angle_moment"], "o", color="green",  label='Instant configurations', markersize=7)
    ax.plot(df["avg_in_drop_moment"], df["angle_points_prof"], "x", color="purple", label='Profile points', markersize=7)
    ax.plot(df["avg_in_drop_moment"], df["angle_sobel_softmax"], "v", color="blue", label='Sobel Profile', markersize=7)
    ax.plot(df["avg_in_drop_moment"], df["angle_interface_softmax"], "^", color="red", label='Interface Density', markersize=7)
    plt.legend()

    plt.savefig(os.path.join(path_to_results, 'alpha_of_N_water_true.png'), bbox_inches='tight', dpi=400)
    fig.tight_layout()
    plt.close()

    font_size = 22
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_ylabel(r"cos($\alpha$)", fontsize=font_size)
    ax.set_xlabel("$1/r, A^{-1}$", fontsize=font_size)
    df = pd.read_csv("results_water_N/gathered_data_1.csv", sep=",")
    plt.yticks(fontsize=18)
    tick_array = [0.01, 0.03, 0.05, 0.07, 0.09]
    labels = [x for x in tick_array]
    plt.xticks(ticks=tick_array, labels=labels, fontsize=18)
    # lower_lim = np.floor(np.min(df["angle_moment"] - df["d_angle_moment"]) / 10)
    # lower_lim = (lower_lim - 1) * 10
    # higher_lim = np.ceil(np.max(df["angle_moment"] + df["d_angle_moment"]) / 10)
    # higher_lim = (higher_lim+1) * 10
    # ax.set_ylim([lower_lim, higher_lim])
    # ax.errorbar(1.0/df["radius_moment"], np.cos(df["angle_moment"]), df["d_angle_moment"], fmt='o', color="black", markersize=0, linewidth=1, capsize=3)
    df = pd.read_csv("results_water_N/gathered_data.csv", sep=",")
    ax.plot(1.0/df["radius_moment"], np.cos(df["angle_moment"]/180*np.pi), "o", color="green",  label='instant configurations', markersize=7)
    popt, pcov = curve_fit(linear, 1.0/df["radius_moment"], np.cos(df["angle_moment"]/180*np.pi))
    print(np.corrcoef(linear(1.0 / df["radius_moment"], *popt), np.cos(df["angle_moment"] / 180 * np.pi),
                      rowvar=False))
    print(list(1.0/df["radius_moment"].values)+[0])
    ax.plot(1.0/df["radius_moment"], linear(1.0/df["radius_moment"], *popt), 'g-', label='fit: a=%f, b=%f' % tuple(popt))

    ax.plot(1.0/df["radius_points_prof"], np.cos(df["angle_points_prof"]/180*np.pi), "x", color="purple", label='profile points', markersize=7)
    popt, pcov = curve_fit(linear, 1.0 / df["radius_points_prof"], np.cos(df["angle_points_prof"] / 180 * np.pi))
    print(np.corrcoef(linear(1.0 / df["radius_points_prof"], *popt), np.cos(df["angle_points_prof"] / 180 * np.pi), rowvar=False))
    ax.plot(1.0 / df["radius_points_prof"], linear(1.0 / df["radius_points_prof"], *popt), '-', color="purple",
            label='fit: a=%f, b=%f' % tuple(popt))

    ax.plot(1.0 / df["r_int"], np.cos(df["angle_interface_softmax"] / 180 * np.pi), "^", color="red",
            label='interface', markersize=7)
    popt, pcov = curve_fit(linear, 1.0 / df["r_int"], np.cos(df["angle_interface_softmax"] / 180 * np.pi))
    print(np.corrcoef(linear(1.0 / df["r_int"], *popt), np.cos(df["angle_interface_softmax"] / 180 * np.pi),
                      rowvar=False))
    ax.plot(1.0 / df["r_int"], linear(1.0 / df["r_int"], *popt), '-', color="red",
            label='fit: a=%f, b=%f' % tuple(popt))

    df = pd.read_csv("results_water_N/gathered_data_1.csv", sep=",")
    ax.plot(1.0 / df["r_sobel"], np.cos(df["angle_sobel_softmax"] / 180 * np.pi), "v", color="blue",
            label='sobel', markersize=7)
    popt, pcov = curve_fit(linear, 1.0 / df["r_sobel"], np.cos(df["angle_sobel_softmax"] / 180 * np.pi))
    print(np.corrcoef(linear(1.0 / df["r_sobel"], *popt), np.cos(df["angle_sobel_softmax"] / 180 * np.pi),
                      rowvar=False))
    ax.plot(1.0 / df["r_sobel"], linear(1.0 / df["r_sobel"], *popt), 'b-',
            label='fit: a=%f, b=%f' % tuple(popt))

    plt.legend()
    plt.savefig(os.path.join(path_to_results, 'alpha_of_R.png'), bbox_inches='tight', dpi=400)
