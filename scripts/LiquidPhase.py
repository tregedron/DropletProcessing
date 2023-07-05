import numpy as np
from utils.utils import custom_distance, custom_distance_without_Z_pbc, custom_distance_2d

class LiquidPhase:
    # TODO: maybe it should inherit from Droplet class
    # TODO: add and treat indexes
    """
        Class is some kind of frame class from chemfiles with additional methods. Stores the information about
        the liquid phase on the frame.
    """

    def __init__(self, frame):
        """
            :param frame: chemfiles frame, usually contains a topology, particle positions, simulation cell parameters.
        """
        self.positions = frame.positions
        self.size = len(self.positions)
        self.mass_center = None
        self.cell = frame.cell.lengths


    def calculate_mass_center(self):
        """
            Calculate the mass center for given coordinates in periodic boundary conditions. Idea of treating
            coordinates on the circle instead of the line is used.
            :return:
        """
        self.mass_center = np.zeros(3, dtype=float)
        theta = self.positions/self.cell * 2 * np.pi
        ksi = np.cos(theta)
        dzeta = np.sin(theta)
        ksi = np.average(ksi, axis=0)
        dzeta = np.average(dzeta, axis=0)
        theta = np.arctan2(-dzeta, -ksi) + np.pi
        self.mass_center = theta / (2*np.pi) * self.cell
