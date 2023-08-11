from utils.utils import custom_distance
from sklearn.cluster import DBSCAN
from collections import Counter

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn import preprocessing
from math import atan
import numpy as np
import pandas as pd
import os


class Profile:
    """
        Class is used to deal with the "profile" of system. The profile is treated as a density voxel map.
    """
    def __init__(self, slices=None, scaling=1):
        """
            :param slices: the number of layers (slices) for system to be divided on. The voxel map will be
            slices[0] x slices[1] x slices[2].
            self.profile - voxel system profile (slices[0] x slices[1] x slices[2])
            self.num_of_averaging - number of frames where the profile was calculated
        """
        if slices is None:
            slices = np.array([100, 100, 100])

        self.scaling = scaling
        self.slices = np.ceil(slices*self.scaling).astype(int)

        self.profile = np.zeros((self.slices[0], self.slices[1], self.slices[2]))
        self.num_of_averaging = 0

    def update_profile(self, droplet):
        """
            Calculate the profile of current frame (droplet or liquid phase).
            :param droplet: !!!DROPLET CAN ALSO BE LIQUID PHASE!!! droplet or liquid phase for profile to be calculated.
            Contains particle positions, simulation cells and other information.

            distances - are "distances" from particles to mass center with respect to the pbc, it can either be
            positive or negative, it is used to make profile "centered" to (spacing/2, spacing/2, spacing/2) voxel.
            Hence, the droplet is always in the center of profile.

            boxes - are the voxels the particle belongs to.
        """
        self.num_of_averaging += 1

        cell = np.array(droplet.cell)

        distances = droplet.positions[:, :] - droplet.mass_center

        distances = np.where(distances >= 0.5 * cell, distances - cell, distances)
        distances = np.where(distances <= -1 * 0.5 * cell, distances + cell, distances)

        distances = np.floor(distances/cell * self.slices) + np.floor(self.slices/2)
        distances = distances.astype(int)
        for box in distances:
            self.profile[box[0], box[1], box[2]] += 1

    def save(self, path, name="profile"):
        """
            The method is used to save profiles to np file.
            :param path: path to the folder where to store profile.
            :param name: name for profile as a file.
        """
        with open(os.path.join(path, f'{name}_{self.scaling}.npy'), 'wb') as f:
            np.save(f, self.profile / self.num_of_averaging)
            np.save(f, self.scaling)

    def load(self, path, name):
        """
            The method is used to load profiles from np file.
            :param path: path to the folder where profile is stored.
            :param name: name of profile as a file.
        """
        with open(os.path.join(path, name), 'rb') as f:
            self.profile = np.load(f)
            if self.slices[0] != self.profile.shape[0] or self.slices[1] != self.profile.shape[1] or self.slices[2] != self.profile.shape[2]:
                self.slices = np.array(self.profile.shape).astype(int)
            self.scaling = np.load(f, allow_pickle=True)

        print("loaded profile", name, "size", self.slices, "scaling was", self.scaling)

    def plot_profile(self, path, name="voxel"):
        """
            The method is used to save profiles as plots.
            :param path: path to the folder where to store profile.
            :param name: name for set of pictures.
        """
        fontsize = 22
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"}, constrained_layout=True)
        ax.voxels(self.profile, alpha=0.1)

        #label axes
        ax.set_xlabel('X', fontsize=fontsize, labelpad=15)
        ax.set_ylabel('Y', fontsize=fontsize, labelpad=15)
        ax.set_zlabel('Z', fontsize=fontsize, labelpad=5)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(18)
        # ortographic projection
        ax.set_proj_type('ortho')

        ax.view_init(0, 0)
        plt.savefig(os.path.join(path, f"{name}_0_0.png"), format="png", dpi=400)

        ax.view_init(90, 0)
        plt.savefig(os.path.join(path, f"{name}_90_0.png"), format="png", dpi=400)

        ax.view_init(0, 90)
        plt.savefig(os.path.join(path, f"{name}_0_90.png"), format="png", dpi=400)

        ax.view_init(30, 30)
        plt.savefig(os.path.join(path, f"{name}_30_30.png"), format="png", dpi=400)

        plt.close()

        fig = plt.figure(figsize=(6, 6), constrained_layout=True)
        ax = fig.add_subplot(111)
        fig.tight_layout()

        ax.set_xlabel('X', fontsize=fontsize)
        ax.set_ylabel('Z', fontsize=fontsize)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax.pcolormesh(self.profile[:, int(self.slices[1]/2), :].T, cmap='YlOrRd')
        plt.savefig(os.path.join(path, f'xz_profile_{name}.png'), bbox_inches='tight')

        ax.set_xlabel('Y', fontsize=fontsize)
        ax.set_ylabel('Z', fontsize=fontsize)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax.pcolormesh(self.profile[int(self.slices[0] / 2), :, :].T, cmap='YlOrRd')
        plt.savefig(os.path.join(path, f'yz_profile_{name}.png'), bbox_inches='tight')
        plt.close()

    def process_profile_to_border_density(self, path, averaged=True, cutoff=0.6):
        """
            The method is used to extract border of "object" (droplet of liquid phase) from profiles. Now the density
            approach is used, but the convolutions can be employed in the future. The threshold can be changed in
            the future.
            :param path: path to the folder where profile is stored.
            :param averaged: defines whether the profile was already averaged with the respect of number of frames
            or not.
            :param cutoff:
        """

        if not averaged:
            self.profile = self.profile / self.num_of_averaging

        average_bulk = self.profile[int(self.slices[0]/2-3):int(self.slices[0]/2+4), int(self.slices[1]/2-3):int(self.slices[1]/2+4), int(self.slices[2]/2-3):int(self.slices[2]/2+4)].mean()

        print("Non zero voxels: ", self.profile[np.nonzero(self.profile)].shape)

        border_bulk = np.where(self.profile < cutoff * average_bulk, self.profile, 0)
        border_bulk = np.where(border_bulk > 0.2 * average_bulk, self.profile, 0)

        filter_1 = np.array([[
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]])

        for x in range(1, self.slices[0]-1):
            for y in range(1, self.slices[1]-1):
                for z in range(1, self.slices[2]-1):
                    if border_bulk[x, y, z] != 0:
                        arr = border_bulk[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] * filter_1
                        if not np.any((arr == 0)):
                            border_bulk[x, y, z] = 0
                        if np.all((arr[:, :, 2] == 0)):
                            border_bulk[x, y, z] = 0

        print("was", border_bulk[np.nonzero(border_bulk)].shape)

        array = np.array(
            [np.nonzero(border_bulk)[0], np.nonzero(border_bulk)[1], np.nonzero(border_bulk)[2]])
        array = array.T

        clustering = DBSCAN(eps=1, min_samples=3, metric='euclidean', n_jobs=2).fit(array).labels_

        counter = dict(Counter(clustering))
        print(counter)
        biggest_ind = max(counter, key=counter.get)

        for j, ind in enumerate(array):
            if clustering[j] != biggest_ind:
                border_bulk[ind[0], ind[1], ind[2]] = 0

        print("now", border_bulk[np.nonzero(border_bulk)].shape)

        with open(os.path.join(path, f'border_bulk_profile_{self.scaling}_{cutoff}.npy'), 'wb') as f:
            np.save(f, border_bulk)
            np.save(f, self.scaling)
            print(f"saved {os.path.join(path, f'border_bulk_profile_{self.scaling}_{cutoff}.npy')}")

    def process_profile_sobel(self, path, averaged=True):
        """
            :param path: path to the folder where profile is stored.
            :param averaged: defines whether the profile was already averaged with the respect of number of frames
            or not.
        """

        if not averaged:
            self.profile = self.profile / self.num_of_averaging

        normalised = self.profile/np.max(self.profile)

        sobel_z = np.array([[
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], [
            [-2, 0, 2],
            [-4, 0, 4],
            [-2, 0, 2]
        ], [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]])
        sobel_y = np.array([[
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], [
            [-2, -4, -2],
            [0, 0, 0],
            [2, 4, 2]
        ], [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]])
        sobel_x = np.array([[
            [-1, -2, -1],
            [-2, -4, -2],
            [-1, -2, -1]
        ], [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]])

        normalised_x = conv(normalised, sobel_x, stride=1, pad=1)
        normalised_y = conv(normalised, sobel_y, stride=1, pad=1)
        normalised_z = conv(normalised, sobel_z, stride=1, pad=1)
        # AR
        # normalised_z_min = conv(normalised, sobel_z, stride=1, pad=1, z_min=int(self.slices[2]*1.25/2))
        # water
        normalised_z_cut = conv(normalised, sobel_z, stride=1, pad=1, z_min=int(self.slices[2] * 1.07 / 2))

        normalised = np.sqrt(normalised_x**2 + normalised_y**2 + normalised_z**2)

        print("Sobel nonzero: ", np.nonzero(normalised))
        with open(os.path.join(path, f'sobel_profile_XYZ_{self.scaling}.npy'), 'wb') as f:
            np.save(f, normalised)
            np.save(f, self.scaling)

        normalised = np.sqrt(normalised_y ** 2 + normalised_x ** 2)
        with open(os.path.join(path, f'sobel_profile_XY_{self.scaling}.npy'), 'wb') as f:
            np.save(f, normalised)
            np.save(f, self.scaling)

        normalised = np.sqrt(normalised_y ** 2 + normalised_x ** 2 + normalised_z_cut ** 2)
        with open(os.path.join(path, f'sobel_profile_XYZ_min_{self.scaling}.npy'), 'wb') as f:
            np.save(f, normalised)
            np.save(f, self.scaling)

        normalised = np.sqrt(normalised_z_cut ** 2)
        with open(os.path.join(path, f'sobel_profile_Z_min_{self.scaling}.npy'), 'wb') as f:
            np.save(f, normalised)
            np.save(f, self.scaling)

    def process_profile_g_of_r(self, path, averaged=True):
        """
            The method is used to calculate radial distribution function from mass center of profile.
            :param path: path to the folder where profile is stored.
            :param averaged: defines whether the profile was already averaged with the respect of number of frames
            or not.
        """
        if not averaged:
            avg_prof = self.profile / self.num_of_averaging
        else:
            avg_prof = self.profile

        center = self.slices/2
        g_of_r = np.zeros(np.max(self.slices))

        for x in range(self.slices[0]):
            for y in range(self.slices[1]):
                for z in range(self.slices[2]):
                    dr = np.round(np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2))
                    dr = dr.astype(int)
                    g_of_r[dr] += avg_prof[x, y, z]

        N = avg_prof[np.nonzero(avg_prof)].sum()
        V = avg_prof[np.nonzero(avg_prof)].shape[0]

        for dist, layer in enumerate(g_of_r):
            g_of_r[dist] = layer / (4 * np.pi * (dist+0.5)**2) * V / N

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.plot(g_of_r[0:100], 'b-', markersize=2)
        plt.axhline(y=1, color='r', linestyle='-')
        plt.savefig(os.path.join(path, f'g(r)_full.png'), bbox_inches='tight')
        fig.tight_layout()
        plt.close()

        print(g_of_r)
        df = pd.DataFrame(columns=['r', 'g'])
        for r, g in enumerate(g_of_r):
            df.loc[len(df.index)] = [r+0.5, g]
        df.to_csv(os.path.join(path, f"GofR_full.csv"))

    def process_profile_2d(self, path, add_name="", averaged=True):
        """
            The method is used to calculate 2d density heatmap (in R and Z variables). R is x^2+y^2, Z is Z.
            Hence, the voxel profile is averaged over the rotation around the rotation axis.
            :param path: path to the folder where profile is stored.
            :param add_name: additional name for 2d heatmap. the default name is "Rz_profile_.png".
            :param averaged: defines whether the profile was already averaged with the respect of number of frames
            or not.
        """

        if not averaged:
            avg_prof = self.profile / self.num_of_averaging
        else:
            avg_prof = self.profile

        center = self.slices/2
        profile_rz = np.zeros((np.max(self.slices[0:2]), self.slices[2]))
        for x in range(self.slices[0]):
            for y in range(self.slices[1]):
                for z in range(self.slices[2]):
                    dr = np.round(np.sqrt((x - center[0])**2 + (y - center[1])**2))
                    dr = dr.astype(int)
                    profile_rz[dr, z] += avg_prof[x, y, z]

        N = avg_prof[np.nonzero(avg_prof)].sum()
        V = avg_prof[np.nonzero(avg_prof)].shape[0]

        print("avg in drop: ", N, "avg non zero in profile:", V)

        for R in range(profile_rz.shape[0]):
            profile_rz[R, :] = profile_rz[R, :] / (2 * np.pi * (R+0.5)) * V / N

        rho_max = np.max(profile_rz)

        with open(os.path.join(path, f'2d_border_profile_{self.scaling}.npy'), 'wb') as f:
            np.save(f, profile_rz)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        fig.tight_layout()

        ax.set_xlabel('R')
        ax.set_ylabel('Z')

        ax.pcolormesh(profile_rz.T, cmap='YlOrRd', vmin=0, vmax=rho_max)
        plt.savefig(os.path.join(path, f'Rz_profile_{add_name}.png'), bbox_inches='tight')
        # plt.show()
        plt.close()

    def fit_profile(self, path, add_name="int_fit", sobel=False, weigth_type="no"):
        add_name = add_name+"_"+weigth_type
        if sobel:
            average = self.profile[np.nonzero(self.profile)].mean()
            print(average)
            # water
            # self.profile = np.where(self.profile > 1.2 * average, self.profile, 0)
            # AR
            self.profile = np.where(self.profile > 2.0 * average, self.profile, 0)

            print("was in nonzero", self.profile[np.nonzero(self.profile)].shape)

            array = np.array(
                [np.nonzero(self.profile)[0], np.nonzero(self.profile)[1], np.nonzero(self.profile)[2]])
            array = array.T
            # water
            # clustering = DBSCAN(eps=1, min_samples=3, metric='euclidean', n_jobs=2).fit(array).labels_
            clustering = DBSCAN(eps=1, min_samples=3, metric='manhattan', n_jobs=2).fit(array).labels_

            counter = dict(Counter(clustering))
            biggest_ind = max(counter, key=counter.get)

            for i, ind in enumerate(array):
                if clustering[i] != biggest_ind:
                    self.profile[ind[0], ind[1], ind[2]] = 0

            print("now after cutting", self.profile[np.nonzero(self.profile)].shape)

        def sphereFit(spX, spY, spZ, weigth_vector=None):
            #   Assemble the A matrix
            spX = np.array(spX)
            spY = np.array(spY)
            spZ = np.array(spZ)
            A = np.zeros((len(spX), 4))
            A[:, 0] = spX * 2
            A[:, 1] = spY * 2
            A[:, 2] = spZ * 2
            A[:, 3] = 1

            #   Assemble the f matrix
            f = np.zeros((len(spX), 1))
            f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ)

            # add weigth
            if weigth_vector is not None:
                A = (weigth_vector * A.T).T
                f = f*weigth_vector[:, None]

            C, residules, rank, singval = np.linalg.lstsq(A, f)

            #   solve for the radius
            t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
            radius = np.sqrt(t)

            return radius, C[0], C[1], C[2], residules

        data = np.nonzero(self.profile)

        if weigth_type == "no":
            weigths_vector = None
        elif weigth_type == "softmax":
            weigths_vector = np.zeros((data[0].shape[0]))
            for ind in range(len(data[0])):
                weigths_vector[ind] = self.profile[data[0][ind], data[1][ind], data[2][ind]]
            norm = np.sum(np.exp(weigths_vector))
            weigths_vector = np.exp(weigths_vector)/norm
        elif weigth_type == "minmax":
            weigths_vector = np.zeros((data[0].shape[0]))
            for ind in range(len(data[0])):
                weigths_vector[ind] = self.profile[data[0][ind], data[1][ind], data[2][ind]]
            weigths_vector = (weigths_vector - np.min(weigths_vector)) / (np.max(weigths_vector) - np.min(weigths_vector))

        print(f"Doing {weigth_type} fitting")

        r, x0, y0, z0, res = sphereFit(data[0], data[1], data[2], weigths_vector)

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:20j]
        x = np.cos(u) * np.sin(v) * r
        y = np.sin(u) * np.sin(v) * r
        z = np.cos(v) * r
        x = x + x0
        y = y + y0
        z = z + z0

        fontsize=22

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
        fig.tight_layout()
        ax.voxels(self.profile, alpha=0.1)
        ax.plot_wireframe(x, y, z, color="r", linewidths=0.5, alpha=1)

        # label axes
        ax.set_xlabel('X', fontsize=fontsize, labelpad=15)
        ax.set_ylabel('Y', fontsize=fontsize, labelpad=15)
        ax.set_zlabel('Z', fontsize=fontsize, labelpad=5)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(18)

        # ortographic projection
        ax.set_proj_type('ortho')

        r = float(r)
        x0 = float(x0)
        y0 = float(y0)
        z0 = float(z0)

        angle = 90 - np.arcsin((np.min(data[2]) - z0) / r) * 180 / np.pi
        fit_text = f'R = {r:.2f}'
        angle_text = r"$Z_{min} = $"+f"{np.min(data[2])}" + r", $\alpha$ = "+f"{angle:.2f}"
        print(z0)
        ax.scatter([x0], [y0], [z0], color="r", alpha=1, linewidths=3, label=f"center: x = {x0:.1f}, y={y0:.1f}, z={z0:.2f}")
        plt.scatter([], [], [], color="w", alpha=0, label=fit_text)
        plt.scatter([], [], [], color="w", alpha=0, label=angle_text)
        plt.legend()

        ax.view_init(90, 0)
        plt.savefig(os.path.join(path, f"{add_name}_90_0.png"), format="png", bbox_inches="tight")

        ax.view_init(0, 90)
        plt.savefig(os.path.join(path, f"{add_name}_0_90.png"), format="png", bbox_inches="tight")

        ax.view_init(30, 30)
        plt.savefig(os.path.join(path, f"{add_name}_30_30.png"), format="png", bbox_inches="tight")

        ax.view_init(0, 0)
        ax.xaxis.set_ticklabels([])
        plt.savefig(os.path.join(path, f"{add_name}_0_0.png"), format="png", bbox_inches="tight")

        plt.close()

        print("stats:", res)

        return angle

    def find_angle_points_prof(self, path, add_name="distrib", r_cut=0.7, z_cut=0.2):
        """
            The method is used to calculate droplet contact angle using the density voxel profile. Two cut off
            parameters are used: r_cut and z_cut, for density in spot area anf in z-distribution respectively.
            The calculation of angle is implemented in the following way: the contact spot is determined as a set of
            molecules in the first density layer in droplet. Then the distribution in radial axis from center mass is
            calculated. The product of average density in 5-15 angstrom range and r_cut gives a threshold value for
            density. Ones the density is lower than threshold the spot area radius is determined. The same procedure
            is done for z density distribution.

            :param path: path to the folder where profile stored.
            :param add_name: additional name for spot area and z density profiles.
            :param r_cut: the cut-off threshold for density in spot area profile, all voxels with density lower than
            r_cut*"average density" are considered to be gas phase
            :param z_cut: the cut-off threshold for density in z-profile, all voxels with density lower than
            z_cut*"average density" are considered to be gas phase
        """

        lowest_disk = np.min(np.where(self.profile != 0)[2])
        highest_disk = lowest_disk+4

        profile_r = np.zeros(np.max(self.slices[0:2]))

        for x in range(self.slices[0]):
            for y in range(self.slices[1]):
                for z in range(lowest_disk, highest_disk):
                    dr = np.round(np.sqrt((x - self.slices[0]/2) ** 2 + (y - self.slices[1]/2) ** 2))
                    dr = dr.astype(int)
                    profile_r[dr] += self.profile[x, y, z]

        for R in range(profile_r.shape[0]):
            profile_r[R] = profile_r[R] / (2 * np.pi * (R + 0.5))

        profile_z = np.zeros(np.max(self.slices[2]))

        for z in range(self.slices[2]):
            for x in range(self.slices[0]):
                for y in range(self.slices[1]):
                    profile_z[z] += self.profile[x, y, z]
            if profile_z[z] != 0:
                z_layer_volume = np.sum(np.nonzero(self.profile[:, :, z]))
                profile_z[z] = profile_z[z] / z_layer_volume

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        fig.tight_layout()
        ax.set_xlabel('R')
        ax.plot(profile_r)
        plt.savefig(os.path.join(path, f'R_profile_disk_{add_name}.png'), bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        fig.tight_layout()
        ax.set_xlabel('Z')
        ax.set_ylabel(r'$\rho$')
        ax.plot(profile_z)
        plt.savefig(os.path.join(path, f'Z_profile_{add_name}.png'), bbox_inches='tight')
        plt.close()

        average_r = np.average(profile_r[5:15])

        spot_radius = 0

        for ind, density_in_r in enumerate(profile_r[5:], start=5):
            if density_in_r < r_cut * average_r:
                spot_radius = ind
                break

        average_z = np.average(profile_z[np.nonzero(profile_z)])
        profile_z = np.where(profile_z > z_cut * average_z, profile_z, 0)

        height = np.max(np.nonzero(profile_z)) - np.min(np.nonzero(profile_z))

        alpha = np.arcsin(2 * spot_radius * height / (spot_radius ** 2 + height ** 2)) * 180.0 / np.pi

        print(alpha)

        if height > ((height ** 2 + spot_radius ** 2) / (2 * height)):
            alpha = 180 - alpha

        df = pd.DataFrame(columns=["r_cut", "z_cut", "height", "spot_radius", "angle"])
        df.loc[len(df.index)] = [r_cut, z_cut, height, spot_radius, alpha]
        df.to_csv(os.path.join(path, f"angle.csv"))
        return height, spot_radius, alpha


class ProfileOnFloor(Profile):
    """
        Class is used to deal with the "profile" of system. The profile is treated as a density voxel map.
        The difference from Profile class is another distances calculations (pbc in Z direction in not used)
    """

    def __init__(self, slices=None):
        if slices is None:
            slices = [100, 100, 100]

        super().__init__(slices)

    def update_profile(self, droplet):
        """
            Calculate the profile of current frame (droplet or liquid phase).
            :param droplet: !!!DROPLET CAN ALSO BE LIQUID PHASE!!! droplet or liquid phase for profile to be calculated.
            Contains particle positions, simulation cells and other information.

            distances - are "distances" from particles to mass center with respect to the pbc (without Z direction),
            it can either be positive or negative, it is used to make profile "centered" to
            (spacing/2, spacing/2, spacing/2) voxel. Hence, the droplet is always in the center of profile.

            boxes - are the voxels the particle belongs to.
        """

        self.num_of_averaging += 1

        cell =  np.array(droplet.cell)

        distances = droplet.positions[:, :] - np.array([droplet.mass_center[0], droplet.mass_center[1], 0])

        distances[:, 0:-1] = np.where(distances[:, 0:-1] > 0.5 * cell[0:-1],
                                      distances[:, 0:-1] - cell[0:-1],
                                      distances[:, 0:-1])
        distances[:, 0:-1] = np.where(distances[:, 0:-1] < -1 * 0.5 * cell[0:-1],
                                      distances[:, 0:-1] + cell[0:-1],
                                      distances[:, 0:-1])

        boxes = np.round((distances/cell * self.slices)) + np.array([np.round(self.slices/2), np.round(self.slices/2), 0])
        boxes = boxes.astype(int)
        for box in boxes:
            self.profile[box[0], box[1], box[2]] += 1


class ProfileLiquidPhase(Profile):
    # TODO here is a ЖЕСТЬ with python indexes usage... but it just works...
    """
        Class is used to deal with the "profile" of system. The profile is treated as a density voxel map.
        The difference from Profile class is another boxes calculations, now the mass center is in (0,0,0) box. This
        moves the bubble to the center of the profile.
    """
    def __init__(self, slices):
        super().__init__(slices)

    def update_profile(self, droplet):
        self.num_of_averaging += 1

        cell = np.array(droplet.cell)

        distances = droplet.positions[:, :] - droplet.mass_center

        distances = np.where(distances > 0.5 * cell, distances - cell, distances)
        distances = np.where(distances < -1 * 0.5 * cell, distances + cell, distances)

        boxes = np.round((distances/cell * self.slices))
        boxes = boxes.astype(int)
        for box in boxes:
            self.profile[box[0], box[1], box[2]] += 1


def find_angle_profile_2d(path, name="2d_border_profile_1.npy", lower_shift=1, account_layers=8):
    """
        The method is used to calculate 2d density heatmap (in R and Z variables). R is x^2+y^2, Z is Z.
        Hence, the voxel profile is averaged over the rotation around the rotation axis.
        :param path: path to the folder where profile is stored.
        :param name:
        :param lower_shift:
        :param account_layers:
    """

    def linear(x, a, b):
        return a * x + b

    with open(os.path.join(path, name), 'rb') as f:
        profile_rz = np.load(f)

        r = []
        z = []

        lower_layer = np.min(np.where(profile_rz != 0)[1]) + lower_shift
        for layer in range(lower_layer, lower_layer+account_layers):
            r.append(np.argmax(profile_rz[:, layer])+0.5)
            z.append(layer+0.5)

        popt, pcov = curve_fit(linear, r, z)
        if popt[0]>0:
            r_grid = np.arange(((lower_layer - popt[1]) / popt[0]) - 0.5, ((lower_layer - popt[1]) / popt[0] + 5), 0.5)
        else:
            r_grid = np.arange(((lower_layer-popt[1])/popt[0]) - 5, ((lower_layer-popt[1])/popt[0]+0.5), 0.5)

        rho_max = np.max(profile_rz)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        fig.tight_layout()

        ax.set_xlabel('R')
        ax.set_ylabel('Z')

        ax.pcolormesh(profile_rz.T, cmap='YlOrRd', vmin=0, vmax=rho_max)
        ax.plot(r_grid, linear(r_grid, *popt), 'b-', label='fit: a=%f, b=%f' % tuple(popt))
        angle = atan(popt[0]) / np.pi * 180
        if angle < 0:
            angle = 180 + angle
        text_ang = f'angle = {angle:.3f}'
        plt.scatter([], [], color="w", alpha=0, label=text_ang)
        plt.legend()
        plt.savefig(os.path.join(path, f'{name}_aprox.png'), bbox_inches='tight')
        plt.close()

        return angle


def conv(x, kernel, stride=1, pad=1, z_min=0):
    """
        The standard convolution function. The resulting new_x array is a convolution of x with kernel. Stride and pad -
        are standard parameters. The only improvement is modification for ignoring part of profile (with z coordinates
        lower than z_min)
        :param x: the given 3-D array.
        :param kernel: the kernel for convolution in out case 3-D.
        :param stride: shift of kernel.
        :param pad: padding of given array x for correct convolution on the edges of profile.
        :param z_min: in some cases we don't want to have convolution in some region of profile, thus the z_min -
        minimal z coordinate for convolution to be calculated.
    """

    new_size = tuple([int(1 + np.floor((x.shape[0] + 2 * pad - kernel.shape[0]) / stride)),
                      int(1 + np.floor((x.shape[1] + 2 * pad - kernel.shape[1]) / stride)),
                      int(1 + np.floor((x.shape[2] + 2 * pad - kernel.shape[2]) / stride))])
    new_x = np.zeros(new_size)
    pad = tuple([[pad, pad], [pad, pad], [pad, pad]])
    x = np.pad(x, pad, mode='constant')

    for new_i, i in enumerate(range(0, x.shape[0] - kernel.shape[0] + 1, stride)):
        for new_j, j in enumerate(range(0, x.shape[1] - kernel.shape[1] + 1, stride)):
            for new_k, k in enumerate(range(0, x.shape[2] - kernel.shape[2] + 1, stride)):
                if k >= z_min:
                    new_x[new_i, new_j, new_k] = np.sum(x[i:i + kernel.shape[0],
                                                        j:j + kernel.shape[1],
                                                        k:k + kernel.shape[2]] * kernel)
                else:
                    new_x[new_i, new_j, new_k] = 0

    return new_x


if __name__ == '__main__':
    print(np.pi)
    print(np.arcsin(0.833)*180/np.pi)
    exit()