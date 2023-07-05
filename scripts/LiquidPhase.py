import numpy as np
from utils.utils import custom_distance, custom_distance_without_Z_pbc, custom_distance_2d

class LiquidPhase:

    def __init__(self, frame):
        self.positions = frame.positions
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
