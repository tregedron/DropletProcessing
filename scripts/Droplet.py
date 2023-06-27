import numpy as np
from utils.utils import custom_distance, custom_distance_without_Z_pbc, custom_distance_2d


class Droplet:

    def __init__(self, frame, indexes):
        self.indexes = indexes
        self.positions = frame.positions[indexes]
        self.size = len(self.positions)
        self.mass_center = None
        self.cell = frame.cell.lengths


    def calculate_mass_center(self):
        '''
        Calculate the mass center for given coordinates.
        :return: mass center position
        '''
        self.mass_center = np.zeros(3, dtype=float)
        theta = self.positions/self.cell * 2 * np.pi
        ksi = np.cos(theta)
        dzeta = np.sin(theta)
        ksi = np.average(ksi, axis=0)
        dzeta = np.average(dzeta, axis=0)
        theta = np.arctan2(-dzeta, -ksi) + np.pi
        self.mass_center = theta / (2*np.pi) * self.cell


class DropletOnFloor(Droplet):

    def __init__(self, frame, indexes):
        super().__init__(frame, indexes)
        self.height = 0
        self.floor = 0
        self.radius = 0
        self.alpha = 0
        self.averaged = 0

    def calc_h_r(self):
        self.floor = min(self.positions[:, 2])
        self.height = max(self.positions[:, 2]) - self.floor
        positions = self.positions[self.positions[:, 2] - self.floor < 3]
        mass_center_in_disc = np.array([self.mass_center[0], self.mass_center[1], self.floor])
        dr = custom_distance_2d(positions, self.cell, center=mass_center_in_disc)
        self.radius = np.sort(dr, axis=None)
        self.radius = np.mean(self.radius[-10:-1])

    def find_alpha(self):
        alpha = np.arcsin(2 * self.radius * self.height / (self.height**2 + self.radius**2))* 180.0 / np.pi
        if self.height < ((self.height**2 + self.radius**2) / (2 * self.height)):
            alpha = 180-alpha
        self.alpha = self.alpha + alpha
        self.averaged = self.averaged + 1


