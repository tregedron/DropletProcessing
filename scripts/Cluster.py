from sklearn.cluster import DBSCAN
from utils.utils import custom_distance, custom_distance_without_Z_pbc

class Cluster:
    #TODO change distance functions in different PBC
    """
    Class is used for clustering on the frame. It takes a frame and some clustering parameters, and store
    the clustering results
    """

    def __init__(self, frame, selection=None, cutoff=5, neighbours=10):
        """
            :param frame: chemfiles frame, usually contains a topology, particle positions, simulation cell parameters.
            :param selection: chemfiles selection, contains some kind of indexes of atoms. Can be used to narrow down
            the number of atoms in clustering.
            :param cutoff: cut off distance for clustering algorithm, provided in Angstroms.
            :param neighbours: number of neighbours in cut off sphere for algorithm (DBSCAN) to expand cluster from
            this point. If 1 is set the DBSCAN is hierarchical clustering. But the noise is not marked as noise (-1).
        """
        self.frame = frame
        self.cutoff = cutoff
        self.neighbours = neighbours
        self.selection = selection
        self.positions = []
        self.clustering = []
        self.dict_residue_to_selected_atom = {}

    def create_positions(self):
        """
            The function makes a selection before clustering. selection.evaluate method from chemfiles is used.
            If the selection is not provided (None) all atoms on the frame will be clustered. The clustering with
            selection can be useful, for example, in case of clustering of micelles, when only oxygen atoms are
            used in clustering.
        """
        if self.selection is None:
            self.positions = self.frame.positions
        else:
            list_selection_ids = self.selection.evaluate(self.frame)
            for ind, O_ind in enumerate(list_selection_ids):
                res_id = self.frame.topology.residue_for_atom(O_ind).id
                if res_id not in self.dict_residue_to_selected_atom.keys():
                    self.dict_residue_to_selected_atom[res_id] = []
                self.dict_residue_to_selected_atom[res_id].append(ind)

            self.positions = self.frame.positions[list_selection_ids]

    def expand_clustering_to_full_frame(self):
        """
            The function expands clustering to the unaccounted atoms belonging to residues, if the selection
            was used.
        """
        frame_labels = []
        for key in self.dict_residue_to_selected_atom.keys():
            for ind in self.frame.topology.residues[key-1].atoms:
                frame_labels.append(self.clustering[self.dict_residue_to_selected_atom[key][0]])
        self.clustering = frame_labels

    def cluster_frame(self, pbc_z=True):
        """
            The function performs clustering of frame. self.create_positions() selects atoms to be clustered,
            then the distance matrix in pbc (custom_distance_matrix) for selected atoms is created. There are two
            different cases of pbc with or without z component (pbc in xyz or in xy only). Then the clustering
            is performed via DBSCAN algorithm based on the distance matrix, eps cutoff and min samples in neighbours are
            declared in initialisation.
            :param pbc_z: boolean variable. Declares whether to use pbc xyz or pbc xy only. True - xyz, False - xy.
        """
        self.create_positions()
        if pbc_z:
            custom_distance_matrix = custom_distance(self.positions, self.frame.cell.lengths)
        else:
            custom_distance_matrix = custom_distance_without_Z_pbc(self.positions, self.frame.cell.lengths)
        self.clustering = DBSCAN(eps=self.cutoff, min_samples=self.neighbours, metric='precomputed', n_jobs=2).fit(custom_distance_matrix).labels_

        if self.selection is not None:
            self.expand_clustering_to_full_frame()

