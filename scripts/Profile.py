from utils.utils import custom_distance

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import atan
import numpy as np
import os


class Profile:
    # TODO: spacing is not the best term for it's functional
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
        self.slices = slices.astype(int)*self.scaling

        self.profile = np.zeros((slices[0], slices[1], slices[2]))
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

        distances = np.where(distances > 0.5 * cell, distances - cell, distances)
        distances = np.where(distances < -1 * 0.5 * cell, distances + cell, distances)

        # boxes = np.round((distances/cell * self.slices)) + np.round(self.slices/2)
        boxes = np.floor((distances/cell * self.slices)) + np.floor(self.slices/2)
        boxes = boxes.astype(int)
        for box in boxes:
            self.profile[box[0], box[1], box[2]] += 1

    def save(self, path, name="profile"):
        """
            The method is used to save profiles to np file.
            :param path: path to the folder where to store profile.
            :param name: name for profile as a file.
        """
        with open(os.path.join(path, f'{name}_{self.scaling}.npy'), 'wb') as f:
            np.save(f, self.profile / self.num_of_averaging)

    def load(self, path, name):
        """
            The method is used to load profiles from np file.
            :param path: path to the folder where profile is stored.
            :param name: name of profile as a file.
        """
        with open(os.path.join(path, name), 'rb') as f:
            self.profile = np.load(f)
            if self.slices[0] != self.profile.shape[0] or self.slices[1] != self.profile.shape[1] or self.slices[2] != self.profile.shape[2]:
                print(self.profile.shape)
                self.slices = np.array(self.profile.shape).astype(int)
            else:
                print(self.profile.shape)

    def plot_profile(self, path, name="voxel"):
        """
            The method is used to save profiles as plots.
            :param path: path to the folder where to store profile.
            :param name: name for set of pictures.
        """
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
        fig.tight_layout()
        ax.voxels(self.profile, alpha=0.1)

        #label axes
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ortographic projection
        ax.set_proj_type('ortho')

        ax.view_init(0, 0)
        plt.savefig(os.path.join(path, f"{name}_0_0.png"), format="png", bbox_inches="tight")

        ax.view_init(90, 0)
        plt.savefig(os.path.join(path, f"{name}_90_0.png"), format="png", bbox_inches="tight")

        ax.view_init(0, 90)
        plt.savefig(os.path.join(path, f"{name}_0_90.png"), format="png", bbox_inches="tight")

        ax.view_init(30, 30)
        plt.savefig(os.path.join(path, f"{name}_30_30.png"), format="png", bbox_inches="tight")

        plt.close()

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        fig.tight_layout()

        ax.set_xlabel('X')
        ax.set_ylabel('Z')

        ax.pcolormesh(self.profile[:, int(self.slices[1]/2), :].T, cmap='YlOrRd')
        plt.savefig(os.path.join(path, f'xz_profile_{name}.png'), bbox_inches='tight')

        ax.set_xlabel('Y')
        ax.set_ylabel('Z')

        ax.pcolormesh(self.profile[int(self.slices[0] / 2), :, :].T, cmap='YlOrRd')
        plt.savefig(os.path.join(path, f'yz_profile_{name}.png'), bbox_inches='tight')
        # plt.show()
        plt.close()

    def process_profile_to_border(self, path, averaged=True):
        """
            The method is used to extract border of "object" (droplet of liquid phase) from profiles. Now the density
            approach is used, but the convolutions can be employed in the future. The threshold can be changed in
            the future.
            :param path: path to the folder where profile is stored.
            :param averaged: defines whether the profile was already averaged with the respect of number of frames
            or not.
        """
        if not averaged:
            avg_prof = self.profile / self.num_of_averaging
        else:
            avg_prof = self.profile

        average = avg_prof[np.nonzero(avg_prof)].mean()
        print(avg_prof[np.nonzero(avg_prof)].shape)

        border = np.where(avg_prof > 0.2 * average, avg_prof, 0)
        border = np.where(border < 0.5 * average, border, 0)

        with open(os.path.join(path, f'border_profile_{self.scaling}.npy'), 'wb') as f:
            np.save(f, border)


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
        ax.plot(g_of_r, 'b-', markersize=3)
        plt.savefig(os.path.join(path, f'g(r).png'), bbox_inches='tight')
        fig.tight_layout()
        # plt.show()
        plt.close()

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
                    dr = np.floor(np.sqrt((x - center[0])**2 + (y - center[1])**2))
                    dr = dr.astype(int)
                    profile_rz[dr, z] += avg_prof[x, y, z]

        N = avg_prof[np.nonzero(avg_prof)].sum()
        V = avg_prof[np.nonzero(avg_prof)].shape[0]

        print(N, V)

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

    def fit_profile(self, path, add_name="", averaged=True):
        def func(xy, a, b, c, d, e, f):
            x, y = xy
            return a + b * x + c * y + d * x ** 2 + e * y ** 2 + f * x * y


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
        # plt.show()
        plt.close()

        return angle



if __name__ == '__main__':
    find_angle_profile_2d("../results/1000H2O")
