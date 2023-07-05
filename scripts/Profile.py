from utils.utils import custom_distance

import matplotlib.pyplot as plt
import numpy as np
import os


class Profile:
    # TODO think about different spacing over axis
    def __init__(self, spacing=100):
        self.spacing = spacing
        self.profile = np.zeros((spacing, spacing, spacing))
        self.num_of_averaging = 0

    def update_profile(self, droplet):
        '''
        droplet can also be liquid phase...
        :return:
        '''
        self.num_of_averaging += 1

        cell = np.array(droplet.cell)

        distances = droplet.positions[:, :] - droplet.mass_center

        distances = np.where(distances > 0.5 * cell, distances - cell, distances)
        distances = np.where(distances < -1 * 0.5 * cell, distances + cell, distances)

        boxes = np.round((distances/cell * self.spacing)) + np.round(self.spacing/2)
        boxes = boxes.astype(int)
        for box in boxes:
            self.profile[box[0], box[1], box[2]] += 1

    def save(self, path):
        with open(os.path.join(path, f'profile_{self.spacing}.npy'), 'wb') as f:
            np.save(f, self.profile / self.num_of_averaging)

    def load(self, path, name):
        with open(os.path.join(path, name), 'rb') as f:
            self.profile = np.load(f)
            if self.spacing != self.profile.shape[0]:
                print(self.profile.shape)
                self.spacing = self.profile.shape[0]
            else:
                print(self.profile.shape)

    def plot_profile(self, path, name="voxel"):
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

    def process_profile_to_border(self, path, averaged=True):
        if not averaged:
            avg_prof = self.profile / self.num_of_averaging
        else:
            avg_prof = self.profile

        average = avg_prof[np.nonzero(avg_prof)].mean()
        print(avg_prof[np.nonzero(avg_prof)].shape)

        border = np.where(avg_prof > 0.2 * average, avg_prof, 0)
        border = np.where(border < 0.4 * average, border, 0)

        with open(os.path.join(path, f'border_profile_{self.spacing}.npy'), 'wb') as f:
            np.save(f, border)

    def process_profile_g_of_r(self, path, averaged=True):
        if not averaged:
            avg_prof = self.profile / self.num_of_averaging
        else:
            avg_prof = self.profile

        center = np.array([self.spacing/2, self.spacing/2, self.spacing/2])
        g_of_r = np.zeros(self.spacing)

        for x in range(self.spacing):
            for y in range(self.spacing):
                for z in range(self.spacing):
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
        plt.show()
        plt.close()


    def process_profile_2d(self, path, name="", averaged=True):
        if not averaged:
            avg_prof = self.profile / self.num_of_averaging
        else:
            avg_prof = self.profile

        center = np.array([self.spacing/2, self.spacing/2, self.spacing/2])
        profile_rz = np.zeros((self.spacing, self.spacing))
        for x in range(self.spacing):
            for y in range(self.spacing):
                for z in range(self.spacing):
                    dr = np.round(np.sqrt((x - center[0])**2 + (y - center[1])**2))
                    dr = dr.astype(int)
                    profile_rz[dr, z] += avg_prof[x, y, z]

        N = avg_prof[np.nonzero(avg_prof)].sum()
        V = avg_prof[np.nonzero(avg_prof)].shape[0]

        print(N, V)

        # for Z in range(profile_rz.shape[1]):
        #     for R in range(profile_rz.shape[0]):
        #         profile_rz[R, Z] = profile_rz[R, Z] / (2 * np.pi * (R + 0.5)) * V / N

        for R in range(profile_rz.shape[0]):
            profile_rz[R, :] = profile_rz[R, :] / (2 * np.pi * (R+0.5)) * V / N

        rho_max = np.max(profile_rz)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        fig.tight_layout()

        ax.set_xlabel('R')
        ax.set_ylabel('Z')

        ax.pcolormesh(profile_rz.T, cmap='YlOrRd', vmin=0, vmax=rho_max)
        plt.savefig(os.path.join(path, f'Rz_profile_{name}.png'), bbox_inches='tight')
        plt.show()
        plt.close()


class ProfileOnFloor(Profile):

    def __init__(self, spacing):
        super().__init__(spacing)

    def update_profile(self, droplet):
        '''

        :return:
        '''
        self.num_of_averaging += 1

        cell =  np.array(droplet.cell)

        distances = droplet.positions[:, :] - np.array([droplet.mass_center[0], droplet.mass_center[1], 0])

        distances[:, 0:-1] = np.where(distances[:, 0:-1] > 0.5 * cell[0:-1],
                                      distances[:, 0:-1] - cell[0:-1],
                                      distances[:, 0:-1])
        distances[:, 0:-1] = np.where(distances[:, 0:-1] < -1 * 0.5 * cell[0:-1],
                                      distances[:, 0:-1] + cell[0:-1],
                                      distances[:, 0:-1])

        boxes = np.round((distances/cell * self.spacing)) + np.array([np.round(self.spacing/2), np.round(self.spacing/2), 0])
        boxes = boxes.astype(int)
        for box in boxes:
            self.profile[box[0], box[1], box[2]] += 1

class ProfileLiquidPhase(Profile):
    # TODO here is a ЖЕСТЬ with python indexes usage... but it just works...
    def __init__(self, spacing):
        super().__init__(spacing)

    def update_profile(self, droplet):
        '''
        droplet can also be liquid phase...
        :return:
        '''
        self.num_of_averaging += 1

        cell = np.array(droplet.cell)

        distances = droplet.positions[:, :] - droplet.mass_center

        distances = np.where(distances > 0.5 * cell, distances - cell, distances)
        distances = np.where(distances < -1 * 0.5 * cell, distances + cell, distances)

        boxes = np.round((distances/cell * self.spacing))
        boxes = boxes.astype(int)
        for box in boxes:
            self.profile[box[0], box[1], box[2]] += 1
