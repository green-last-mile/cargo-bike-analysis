"""
HexagonalDataset.

This dataset is used to train a hexagonal encoder model.
As defined in GeoVex paper[1].

References:
    [1] https://openreview.net/forum?id=7bvWopYY1H
"""
from typing import List
import numpy as np
from srai.h3 import get_local_ij_index
from srai.embedders.geovex.dataset import HexagonalDataset
import torch
from tqdm import tqdm



class HexagonalDatasetPatch(HexagonalDataset):  # type: ignore
    """
    Dataset for the hexagonal encoder model.

    It works by returning a 3d tensor of hexagonal regions. The tensor is a cube with the target
    hexagonal region in the center, and the rings of neighbors around surrounding it.
    """

    def __init__(
        self,
        target_indexes: List[str],
        data,
        neighbourhood,
        neighbor_k_ring: int = 6,
        **kwargs,
    ):
        # self._assert_k_ring_correct(neighbor_k_ring)
        # self._assert_h3_neighbourhood(neighbourhood)
        # store the desired k
        self._k: int = neighbor_k_ring
        # number of columns in the dataset
        self._N: int = data.shape[1]
        # store the list of valid h3 indices (have all the neighbors in the dataset)
        self._valid_cells = []
        # store the data as a torch tensor
        self._data_torch = torch.Tensor(data.to_numpy(dtype=np.float32))
        # iterate over the data and build the valid h3 indices
        self._invalid_cells, self._valid_cells = self._seperate_valid_invalid_cells(
            target_indexes, data, neighbourhood, neighbor_k_ring, set(data.index)
        )

    def _seperate_valid_invalid_cells(
        self,
        target_indexes: List[str],
        data,
        neighbourhood,
        neighbor_k_ring: int,
        all_indices: set[str],
    ) -> tuple[set[str], ]:
        invalid_h3s = set()
        valid_h3s = []

        for h3_index in tqdm(target_indexes, total=len(target_indexes)):
            neighbors = neighbourhood.get_neighbours_up_to_distance(
                h3_index, neighbor_k_ring, include_center=False, unchecked=True
            )
            # check if all the neighbors are in the dataset
            if len(neighbors.intersection(all_indices)) == len(neighbors):
                # add the h3_index to the valid h3 indices, with the ring of neighbors
                valid_h3s.append(
                    (
                        h3_index,
                        data.index.get_loc(h3_index),
                        [
                            # get the index of the h3 in the dataset
                            (data.index.get_loc(_h), get_local_ij_index(h3_index, _h))
                            for _h in neighbors
                        ],
                    )
                )
            else:
                # some of the neighbors are not in the dataset, add the h3_index to the invalid h3s
                invalid_h3s.add(h3_index)
        return invalid_h3s, valid_h3s

   