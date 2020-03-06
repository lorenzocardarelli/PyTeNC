from numpy import dot

class Chain:
    """This class represents Things

    :param color: how the thing should look like
    """

    def __init__(s):
        """Tells you the color of the thing.

        :returns: Color string
        :rtype: str or unicode
        """
        s.LATTICE_UNIT_CELL_SIZE = 1
        s.lattice_keys = {
            'site': None,
            'nearest_n': ['two', None],
            'next_nearest_n': ['three', None]
            }

class Bipartite_Chain:
    def __init__(s):
        s.LATTICE_UNIT_CELL_SIZE = 2
        s.lattice_keys = {
            'site': None,
            'first_site': 0,
            'second_site': 1,
            'edge': ['two', None],
            # 'two': nearest-neigbours operators
            # [0,1]: the leftmost term acts on the sites of the bipartition sublattice [0,1]
            'nearest_n_sublattice_0': ['two', 0],
            'nearest_n_sublattice_1': ['two', 1],
            'next_nn_sublattice_0': ['three', 0],
            'next_nn_sublattice_1': ['three', 1],
            'second_nnn_sublattice_0': ['four', 0],
            'second_nnn_sublattice_1': ['four', 1],
            }

class Ladder:
    def __init__(s):
        s.LATTICE_UNIT_CELL_SIZE = 2
        s.lattice_keys = {
            'bottom_site': 0,
            'top_site': 1,
            'rung': ['two', 0],
            'bottom_edge': ['three', 0],
            'top_edge': ['three', 1]
            }

class Triangular_Ladder:
    def __init__(s):
        s.LATTICE_UNIT_CELL_SIZE = 2
        s.lattice_keys = {
            'bottom_site':0,
            'top_site':1,
            'rung_a': ['two',0],
            'rung_b': ['two',1],
            'bottom_edge': ['three',0],
            'top_edge': ['three',1]
            }

#               1 -- te -- 1
#             .  .       .
#           a      b   .
#         .          .
#        0 -- be -- 0

class Bipartite_Ladder:
    def __init__(s):
        s.LATTICE_UNIT_CELL_SIZE = 4
        s.lattice_keys = {
            'first_bot_site':0,
            'first_top_site':1,
            'second_bot_site':2,
            'second_top_site':3,
            'first_rung': ['two',0],
            'second_rung': ['two',2],
            'first_bot_edge': ['three',0],
            'first_top_edge': ['three',1],
            'second_bot_edge': ['three',2],
            'second_top_edge': ['three',3],
            'first_SouthWest_NorthEast_diagonal': ['four',0],
            'second_SouthWest_NorthEast_diagonal': ['four',2]
            }

