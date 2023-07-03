import numpy as np
import matplotlib.pyplot as plt
import os


class Profile:
    # TODO think about different spacing over axis
    def __init__(self, spacing=100):
        self.spacing = spacing
        self.profile = np.zeros((spacing, spacing, spacing))
        self.num_of_averaging = 0

    def update_profile(self, droplet):
        '''

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
        with open(os.path.join(path, 'profile.npy'), 'wb') as f:
            np.save(f, self.profile / self.num_of_averaging)

    def load(self, path):
        with open(os.path.join(path, 'profile.npy'), 'rb') as f:
            self.profile = np.load(f)
            if self.spacing != self.profile.shape[0]:
                print(self.profile.shape)
                self.spacing = self.profile.shape[0]
            else:
                print(self.profile.shape)

    def plot_profile(self, path):
        print(self.num_of_averaging)
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
        ax.voxels(self.profile, alpha=0.1)

        #label axes
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.set_proj_type('ortho')

        ax.view_init(0, 0)
        fig.tight_layout()
        plt.savefig(os.path.join(path, "voxel_0_0.png"), format="png", bbox_inches="tight")

        ax.view_init(90, 0)
        fig.tight_layout()
        plt.savefig(os.path.join(path, "voxel_90_0.png"), format="png", bbox_inches="tight")

        ax.view_init(0, 90)
        fig.tight_layout()
        plt.savefig(os.path.join(path, "voxel_0_90.png"), format="png", bbox_inches="tight")

        ax.view_init(30, 30)
        fig.tight_layout()
        plt.savefig(os.path.join(path, "voxel_30_30.png"), format="png", bbox_inches="tight")

        plt.close()

    def process_profile_to_border(self, path):
        avg_prof = self.profile / self.num_of_averaging

        average = avg_prof[np.nonzero(avg_prof)].mean()
        print(avg_prof[np.nonzero(avg_prof)].shape)

        border = np.where(avg_prof > 0.4 * average, avg_prof, 0)
        border = np.where(border < 0.6 * average, border, 0)

        print(border[np.nonzero(border)].shape)

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
        ax.voxels(border, alpha=0.1)
        ax.set_proj_type('ortho')

        ax.view_init(90, 0)
        fig.tight_layout()
        plt.savefig(os.path.join(path, "border_90_0.png"), format="png", bbox_inches="tight")

        ax.view_init(0, 0)
        fig.tight_layout()
        plt.savefig(os.path.join(path, "border_0_0.png"), format="png", bbox_inches="tight")
        fig.tight_layout()

        ax.view_init(30, 30)
        fig.tight_layout()
        plt.savefig(os.path.join(path, "border_30_30.png"), format="png", bbox_inches="tight")

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



