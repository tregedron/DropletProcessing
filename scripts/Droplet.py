import numpy as np
from utils.utils import custom_distance, custom_distance_without_Z_pbc, custom_distance_2d

mass_dictionary = {"O": 15.999, "C": 12.0096, "H": 1.00784, "A": 1}

class Droplet:
    """
        Class is used to store the droplet and some info about it.
    """

    def __init__(self, frame, indexes):
        """
            :param frame: chemfiles frame, usually contains a topology, particle positions, simulation cell parameters.
            :param indexes: list of indexes of atoms. Tells whis atoms belong to this droplet, can be obtained from
            clustering.
            self.size - size of the droplet
            self.mass_center - mass center of droplet
            self.cell - the simulation box sizes at the frame which the droplet belongs to.
        """
        self.indexes = indexes
        self.positions = frame.positions[indexes]
        self.size = len(self.positions)
        self.mass_center = None
        self.cell = frame.cell.lengths
        self.masses = np.array([mass_dictionary[frame.topology.atoms[i].name[0]] for i in self.indexes])


    def calculate_mass_center(self):
        """
            Calculate the mass center for given coordinates in periodic boundary conditions. Idea of treating
            coordinates on the circle instead of the line is used.
        """
        self.mass_center = np.zeros(3, dtype=float)
        theta = self.positions/self.cell * 2 * np.pi
        ksi = np.cos(theta)
        dzeta = np.sin(theta)
        ksi = np.average(ksi, weights=self.masses, axis=0)
        dzeta = np.average(dzeta, weights=self.masses, axis=0)
        theta = np.arctan2(-dzeta, -ksi) + np.pi
        self.mass_center = theta / (2*np.pi) * self.cell



class DropletOnFloor(Droplet):
    # TODO: deal with the averaging or remove averaging from here.
    """
            Class is used to store the droplet on the surface and some info about it. Some methods to determine
            droplet-surface contact angle are implemented  (based on geometry, the contact angle is calculated
            in aproximation of spherical droplet).
    """

    def __init__(self, frame, indexes):
        """
            :param frame: chemfiles frame, usually contains a topology, particle positions, simulation cell parameters.
            :param indexes: list of indexes of atoms. Tells whis atoms belong to this droplet, can be obtained from
            clustering.
            self.height - the height of droplet in z direction.
            self.floor - the lowest point of droplet in z direction.
            self.radius - the radius of droplet-surface contact area. Contact area is considered to be circle.
            self.alpha - contact angle of the droplet.
            self.averaged - number of frames where droplet existed. Not sure that this is necessary in current
            realisation.
        """
        super().__init__(frame, indexes)
        self.height = 0
        self.floor = 0
        self.radius = 0
        self.alpha = 0
        self.averaged = 0

    def calc_h_r(self):
        # TODO implement the "true" radius of droplet calculations.
        """
            The method is used to determine the geometric parameters of droplet: height, lowest point (floor),
            and contact area radius.
        """
        self.floor = min(self.positions[:, 2])
        self.height = max(self.positions[:, 2]) - self.floor
        positions = self.positions[self.positions[:, 2] - self.floor < 3]
        mass_center_in_disc = np.array([self.mass_center[0], self.mass_center[1], self.floor])
        dr = custom_distance_2d(positions, self.cell, center=mass_center_in_disc)
        self.radius = np.sort(dr, axis=None)
        self.radius = np.mean(self.radius[-10:-1])

    def find_alpha(self):
        """
            The method is used to determine the contact angle of droplet. The geometry of droplet is used.
        """
        alpha = np.arcsin(2 * self.radius * self.height / (self.height**2 + self.radius**2)) * 180.0 / np.pi
        if self.height < ((self.height**2 + self.radius**2) / (2 * self.height)):
            alpha = 180-alpha
        self.alpha = self.alpha + alpha
        self.averaged = self.averaged + 1




