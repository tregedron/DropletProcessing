import numpy as np
from utils.utils import custom_distance, custom_distance_without_Z_pbc, custom_distance_2d
import matplotlib.pyplot as plt
import pandas as pd
import os

class RDF:
    """
        Class is used to process the radial distribution functions.
    """

    def __init__(self, mode="atoms", r_cut=100, bins=500):
        """
            number of bins in the radial function
        """
        self.R_cut = r_cut
        self.bins = bins

        if mode == "atoms":
            print("calculation of RDF between atoms")
            self.calc_point = None
        elif mode == "point":
            print("calculation of RDF between atoms and point")
            self.calc_point = np.array(3)

        self.rdf = np.zeros(bins)
        self.particles_accounted = 0
        self.n_times_calculated = 0

    def update_rdf_atoms(self, frame):
        distances = custom_distance(frame.positions, frame.cell.lengths, center=None)
        pass

    def update_rdf_point(self, frame, mass_center, indexes):
        distances = custom_distance(frame.positions, frame.cell.lengths, center=mass_center)[indexes]
        distances = distances[distances < self.R_cut]

        distances = np.floor(distances / self.R_cut * self.bins)
        distances = distances.astype(int)

        for box in distances:
            self.rdf[box] = self.rdf[box]+1
        self.n_times_calculated += 1
        self.particles_accounted += distances.shape[0]

    def normalize_rdf(self, path, add_name=""):
        self.rdf = self.rdf/self.n_times_calculated

        N_avg = self.particles_accounted/self.n_times_calculated
        R_min = np.min(np.nonzero(self.rdf))
        dr = self.R_cut/self.bins
        print(N_avg, R_min)
        for r, N_r in enumerate(self.rdf):
            self.rdf[r] = self.rdf[r]/(4*np.pi*dr*((r+0.5)*dr)**2)

        self.plot_and_save(path, name=f"rho_of_R_{add_name}")

        self.rdf[r] = self.rdf[r]*(4 / 3 * np.pi * (self.R_cut ** 3 - R_min ** 3)) / N_avg

        self.plot_and_save(path, name=f"G_of_R_{add_name}")

    def plot_and_save(self, path, name="some_destrib"):

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.plot(self.rdf, 'b-', markersize=2)
        plt.axhline(y=1, color='r', linestyle='-')
        plt.savefig(os.path.join(path, f'{name}.png'), bbox_inches='tight')
        fig.tight_layout()
        plt.close()

        df = pd.DataFrame(columns=['r', 'g'])
        for r, g in enumerate(self.rdf):
            df.loc[len(df.index)] = [(r + 0.5)*self.R_cut/self.bins, g]
        df.to_csv(os.path.join(path, f"{name}.csv"))

        print(self.n_times_calculated, self.particles_accounted)





