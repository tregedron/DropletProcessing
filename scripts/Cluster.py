from sklearn.cluster import DBSCAN
from utils.utils import custom_distance

class Cluster:

    def __init__(self, frame, selection=None, cutoff=5, neighbours=1):
        self.frame = frame
        self.cutoff = cutoff
        self.neighbours = neighbours
        self.selection = selection
        self.positions = []
        self.clustering = []
        self.dict_residue_to_selected_atom = {}

    def create_positions(self):
        if self.selection is None:
            self.positions = self.frame.positions
        else:
            # dict_residue_to_selected_atom contains dictionary: {res_id : o_ids}
            list_selection_ids = self.selection.evaluate(self.frame)
            for ind, O_ind in enumerate(list_selection_ids):
                res_id = self.frame.topology.residue_for_atom(O_ind).id
                if res_id not in self.dict_residue_to_selected_atom.keys():
                    self.dict_residue_to_selected_atom[res_id] = []
                self.dict_residue_to_selected_atom[res_id].append(ind)

            self.positions = self.frame.positions[list_selection_ids]

    def expand_clustering_to_full_frame(self):
        frame_labels = []
        for key in self.dict_residue_to_selected_atom.keys():
            for ind in self.frame.topology.residues[key-1].atoms:
                frame_labels.append(self.clustering[self.dict_residue_to_selected_atom[key][0]])
        self.clustering = frame_labels

    def cluster_frame(self):
        self.create_positions()
        custom_distance_matrix = custom_distance(self.positions, self.frame.cell.lengths)
        self.clustering = DBSCAN(eps=self.cutoff, min_samples=self.neighbours, metric='precomputed', n_jobs=2).fit(custom_distance_matrix).labels_

        if self.selection is not None:
            self.expand_clustering_to_full_frame()

