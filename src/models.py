from copy import deepcopy

from numpy import int16 as npint16
from numpy import int64 as npint64

from numpy import abs as npabs
from numpy import append as npappend
from numpy import array as nparray
from numpy import array_equal
from numpy import atleast_2d
from numpy import concatenate
from numpy import copy as npcopy
from numpy import count_nonzero
from numpy import diag
from numpy import dot
from numpy import empty
from numpy import eye
from numpy import kron
from numpy import mod
from numpy import nonzero
from numpy import pi
from numpy import real_if_close
from numpy import save as npsave
from numpy import zeros

from scipy.linalg import expm
from scipy import sparse

from sys import exit as sysexit

from numpy import set_printoptions
set_printoptions( threshold = 1000 )

from operators import (
                        Angular_Momentum_Operators,
                        Particle_Operators,
                        Pauli_Operators,
                        Boson_Operators,
#                        Fermion_Operators
                      )

#==============================================================================
#
# Note: the operators must be in the form of a list because other functions
# refer to them as such - even if there's no bipartition or whatsoever.
#
#==============================================================================
#
# REMARK: the operators class (say, Angular_Momentum) need to be initialized as
# first. With that, the class inherits the attributes s.d, s.ONE and s.P_list.
# Then one can initialize the class Model, which uses s.ONE to assign the
# identity matrix attribute.
#
#==============================================================================


class Model:

    def __init__(s):
        s.model_operators = {}

    def make_opt(s, key, op):
        s.model_operators[ key ] = [ op ]



class Spin_Model(Angular_Momentum_Operators, Model):

    def __init__(s, Spin ):
        Angular_Momentum_Operators.__init__(s, Spin)
        Model.__init__(s)
        s.build_opts( )

    def build_opts(s):
        s.make_opt( 'Sz', s.LZ )
        s.make_opt( 'Sx', s.LX )
        s.make_opt( 'Sy', s.LY )
        s.make_opt( 'Sp', s.LP )
        s.make_opt( 'Sm', s.LM )



class Ising_Pauli_Model(Pauli_Operators, Model):

    def __init__(s):
        Pauli_Operators.__init__(s)
        Model.__init__(s)
        s.build_opts( )

    def build_opts(s):
        s.make_opt( 'sigmaZ', s.sigmaZ )
        s.make_opt( 'sigmaX', s.sigmaX )
        s.make_opt( 'sigy', s.sigY )
        s.make_opt( 'sigp', s.sigP )
        s.make_opt( 'sigm', s.sigM )



class Bose_Hubbard_Model(Boson_Operators, Model):

    def __init__(s, n ):
        Boson_Operators.__init__(s, n)
        Model.__init__(s)
        s.build_opts( )

    def build_opts(s):
        s.make_opt('b',      s.b)
        s.make_opt('b_dag',  s.b_dag)
        s.make_opt('n',      s.n )
        s.make_opt('n(n-1)', s.n2n )



class Spin_Matter_Model(Model):

    def __init__(s, gauge_symmetry, number_legs_ladder, boundary_conditions):
        s.ONE = 0 # We need ONE to be defined for initializing Model. Afterwards it gets overwritten.
        Model.__init__(s)
        s.number_legs_ladder = number_legs_ladder
        s.number_internal_rungs = number_legs_ladder - 1
        if boundary_conditions == 'OBC':
            s.number_active_rungs = number_legs_ladder - 1
        elif boundary_conditions == 'PBC':
            s.number_active_rungs = number_legs_ladder

        # =====================================================================
        # Operators on the fermionic space - mass.
        # =====================================================================
        po = Particle_Operators( )
        s.idm =     po.ONE     # Identity matrix.
        s.null =    0*s.idm    # Null matrix.
        s.psi_dag = po.psi_dag # Creation operator.
        s.psi =     po.psi     # Destruction operator.
        s.n =       po.n       # Particle number, decreasingly sorted.

        # Symmetry transformations.
        s.phi = po.phi # Particle-Hole inversion.

        # =====================================================================
        # Operators on the bosonic space - links.
        # =====================================================================
        amo = Angular_Momentum_Operators(0.5)
        s.Sp = amo.LP # Raising operator, U_dag.
        s.Sm = amo.LM # Lowering operator, U.
        s.Sz = amo.LZ # E_z.
        s.Sz2 = amo.LZ_sqrd  # E_z^2.
        s.P = {'up': amo.P_list[0], 'down': amo.P_list[1]}

        # Symmetry transformations.
        s.Sx = 2*amo.LX
        s.pi_rotation_x = amo.pi_rotation_x
        s.pi_rotation_z = amo.pi_rotation_z

        # =====================================================================
        # Generate the projectors onto gauge-invariant space and with those
        # make the operators set.
        # =====================================================================
        s.get_dictionary_identity_operators_on_links_and_vertices_of_rung( gauge_symmetry )
        s.set_gauge_symmetry( gauge_symmetry, boundary_conditions )
        s.make_gauge_invariant_operators( gauge_symmetry, boundary_conditions )
        s.make_operators_for_links_alignment( )
        s.display_rung_states_space( gauge_symmetry, boundary_conditions )


    def get_dictionary_identity_operators_on_links_and_vertices_of_rung(s, gauge_symmetry):
        """Define a 'template' dictionary of identity operators.

        The dictionary is a self-object. It gets deep-copied when definining an
        operator that acts on the Hilbert space of a rung (not projected on the
        gauge invariant subspace).

        Parameters
        ----------
        None

        Yields
        -------
        dict_identity_operators_on_links_and_vertices_of_rung : dict
            A dictionary of lists of identity matrices in sparse.csr_matrix format.

        """
        if gauge_symmetry == 'quantum_link':
            vertex_idm = eye(2)
        elif gauge_symmetry == 'spin_ice':
            vertex_idm = eye(1)

        link_idm = eye(2)

        s.dict_identity_operators_on_links_and_vertices_of_rung = { }
        s.dict_identity_operators_on_links_and_vertices_of_rung['vertex'] =     []
        s.dict_identity_operators_on_links_and_vertices_of_rung['link_left'] =  []
        s.dict_identity_operators_on_links_and_vertices_of_rung['link_right'] = []
        s.dict_identity_operators_on_links_and_vertices_of_rung['link_rung'] =  []

        for i in range(s.number_legs_ladder):
            s.dict_identity_operators_on_links_and_vertices_of_rung['vertex'].append(   vertex_idm )
            s.dict_identity_operators_on_links_and_vertices_of_rung['link_left'].append(  link_idm )
            s.dict_identity_operators_on_links_and_vertices_of_rung['link_right'].append( link_idm )
            s.dict_identity_operators_on_links_and_vertices_of_rung['link_rung'].append(  link_idm )

        s.dict_identity_operators_on_links_and_vertices_of_rung['link_rung'].append( s.idm )


    def set_gauge_symmetry(s, gauge_symmetry, boundary_conditions):
        """Set the gauge symmetry and get the projectors onto the invariant Hilbert space.

        Parameters
        ----------
        gauge_symmetry : string
            Specifies the wished gauge symmetry: 'quantum_link', 'spin_ice'.
        boundary_conditions : string
            Specifies the wished boundary conditions: 'OBC', 'PBC'.

        Yields
        -------
        gis_projector : list[ csr_matrix, shape (d, D) ]
            The list contains 2 projectors if the system is bipartite (QLM), 1 otherwise (SpinIce).
            d = dimension rung gauge invariant Hilbert space.
            D = dimension rung Hilbert space, 2^( #links + #vertices ).

        """

        s.gis_projector = [ ]

        # Specify the charge density according to the wished model.
        if gauge_symmetry == 'quantum_link':
            s.translational_invariance_length = 2

            # rung 0: antiparticle at the bottom.
            charge_density_set = [ s.n-s.idm, s.n ]
            # Bottom/top links boundary conditions.
            if boundary_conditions == 'OBC':
                # This is the only configuration which is topological in 2-legs ladder..
                link_rung_boundary_conditions = [ -1, (-1)**(s.number_legs_ladder+1) ]
            # If the boundary conditions are periodic, the list of values on the boundary is obviously empty.
            elif boundary_conditions == 'PBC':
                link_rung_boundary_conditions = [ ]

            set_boundary_conditions = ( boundary_conditions, link_rung_boundary_conditions )
            s.gis_projector.append( s.get_matrix_projector_rung_gauge_invariant_Hilbert_space( charge_density_set, set_boundary_conditions ) )

            # rung 1: particle at the bottom.
            charge_density_set = [ s.n, s.n-s.idm ]
            # Bottom/top links boundary conditions.
            if boundary_conditions == 'OBC':
                # This is the only configuration which is topological in 2-legs ladder..
                link_rung_boundary_conditions = [ 1, (-1)**s.number_legs_ladder ]
            # If the boundary conditions are periodic, the list of values on the boundary is obviously empty.
            elif boundary_conditions == 'PBC':
                link_rung_boundary_conditions = [ ]

            set_boundary_conditions = ( boundary_conditions, link_rung_boundary_conditions )
            s.gis_projector.append( s.get_matrix_projector_rung_gauge_invariant_Hilbert_space( charge_density_set, set_boundary_conditions ) )

        elif gauge_symmetry == 'spin_ice':
            s.translational_invariance_length = 1

            # Charge density null everywhere: ice rule.
            #charge_density_set = [ s.null ]
            charge_density_set = [ nparray([[0.]]) ]
            # Bottom/top links boundary conditions.
            if boundary_conditions == 'OBC':
                # This is to be set, now it's random.
                link_rung_boundary_conditions = [ -1, -1 ]
                set_boundary_conditions = ( boundary_conditions, link_rung_boundary_conditions )
            s.gis_projector.append( s.get_matrix_projector_rung_gauge_invariant_Hilbert_space( charge_density_set, set_boundary_conditions ) )


    def get_matrix_projector_rung_gauge_invariant_Hilbert_space(s, charge_density_set, set_boundary_conditions):
        """Make the rectangular matrix to project onto the gauge invariant Hilbert space.

        0) According to the parameters, set the boundary conditions and charge density operators.
        1) Build the Gauss law operators: Sz on the links, charge density on the vertices.
        2) Get the projectors onto the rung gauge invariant Hilbert space.

        Parameters
        ----------
        charge_density_set : list
        set_boundary_conditions : tuple( string, list )
            string = specifies the wished boundary conditions: 'OBC', 'PBC'.
            list = contains the fixed values at the boundaries.

        Returns
        -------
        gauge_invariant_space_projector : csr_matrix, shape (d, D)
            d = dimension rung gauge invariant Hilbert space.
            D = dimension rung Hilbert space, 2^( #links + #vertices ).

        """

        # 0) Set the boundary conditions and charge density operators according to the input parameters.
        link_rung_operator = [ s.Sz for _ in range(s.number_legs_ladder + 1) ]
        charge_density_operator = [ charge_density_set[ mod(_, len(charge_density_set)) ] for _ in range(s.number_legs_ladder) ]

        # 1) Build the Gauss law operators: Sz on the links, charge density on the vertices.
        dict_Gauss_law_operators_rung_Hilbert_space = { }
        dict_Gauss_law_operators_rung_Hilbert_space['charge_density'] = [ _ for _ in range(s.number_legs_ladder) ]
        dict_Gauss_law_operators_rung_Hilbert_space['sz_link_left'] =   [ _ for _ in range(s.number_legs_ladder) ]
        dict_Gauss_law_operators_rung_Hilbert_space['sz_link_right'] =  [ _ for _ in range(s.number_legs_ladder) ]
        dict_Gauss_law_operators_rung_Hilbert_space['sz_link_rung'] =   [ _ for _ in range(s.number_legs_ladder+1) ]

        for leg_index in range(s.number_legs_ladder):
            list_operators_on_rung = [('vertex', leg_index, charge_density_operator[leg_index])]
            dict_Gauss_law_operators_rung_Hilbert_space['charge_density'][leg_index] = s.make_operator_rung_full_Hilbert_space( list_operators_on_rung )

            list_operators_on_rung = [('link_left', leg_index, s.Sz)]
            dict_Gauss_law_operators_rung_Hilbert_space['sz_link_left'][leg_index] = s.make_operator_rung_full_Hilbert_space( list_operators_on_rung )

            list_operators_on_rung = [('link_right', leg_index, s.Sz)]
            dict_Gauss_law_operators_rung_Hilbert_space['sz_link_right'][leg_index] = s.make_operator_rung_full_Hilbert_space( list_operators_on_rung )

        for leg_index in range( s.number_legs_ladder + 1 ):
            list_operators_on_rung = [('link_rung', leg_index, link_rung_operator[leg_index])]
            dict_Gauss_law_operators_rung_Hilbert_space['sz_link_rung'][leg_index] = s.make_operator_rung_full_Hilbert_space( list_operators_on_rung )

        # 2) Get the projector onto the rung gauge invariant Hilbert space.
        gauge_invariant_space_projector = \
            s.make_matrix_projector_onto_rung_gauge_invariant_Hilbert_space( dict_Gauss_law_operators_rung_Hilbert_space, set_boundary_conditions )

        return gauge_invariant_space_projector


    def make_operator_rung_full_Hilbert_space(s, list_operators_on_rung):
        """Build a matrix operator acting on the Hilbert space of a rung.

        The operator construction requires a ordered tensor product of the operators
        acting on the links and vertices of the rung. The order is the following,
        reflected in the 'for' loops: 'vertex', 'link_left', 'link_right', 'link_rung'.

        Parameters
        ----------
        list_operators_on_rung : list[ tuples( position_on_rung, leg_or_rung_index, operator ) ]

        Returns
        -------
        rung_Hilbert_space_operator : csr_matrix
            An operator acting on the rung Hilbert space.

        """
        dict_operators_on_links_and_vertices_of_rung = deepcopy(s.dict_identity_operators_on_links_and_vertices_of_rung)

        for my_tuple in list_operators_on_rung:
            position_on_rung =  my_tuple[0]
            leg_or_rung_index = my_tuple[1]
            operator =          my_tuple[2]
            dict_operators_on_links_and_vertices_of_rung[position_on_rung][leg_or_rung_index] = operator

        rung_Hilbert_space_operator = s.kron_all_legs_and_rungs_operators( dict_operators_on_links_and_vertices_of_rung )

        return rung_Hilbert_space_operator


    def kron_all_legs_and_rungs_operators(s, dict_operators_on_links_and_vertices_of_rung):
        """

        """
        rung_Hilbert_space_operator = sparse.csr_matrix( nparray([1]) )

        for operator in dict_operators_on_links_and_vertices_of_rung['vertex']:
            rung_Hilbert_space_operator = sparse.csr_matrix( sparse.kron( rung_Hilbert_space_operator, operator ) )
            rung_Hilbert_space_operator.eliminate_zeros( )

        for operator in dict_operators_on_links_and_vertices_of_rung['link_left']:
            rung_Hilbert_space_operator = sparse.csr_matrix( sparse.kron( rung_Hilbert_space_operator, operator ) )
            rung_Hilbert_space_operator.eliminate_zeros( )

        for operator in dict_operators_on_links_and_vertices_of_rung['link_right']:
            rung_Hilbert_space_operator = sparse.csr_matrix( sparse.kron( rung_Hilbert_space_operator, operator ) )
            rung_Hilbert_space_operator.eliminate_zeros( )

        for operator in dict_operators_on_links_and_vertices_of_rung['link_rung']:
            rung_Hilbert_space_operator = sparse.csr_matrix( sparse.kron( rung_Hilbert_space_operator, operator ) )
            rung_Hilbert_space_operator.eliminate_zeros( )

        return rung_Hilbert_space_operator


    def make_matrix_projector_onto_rung_gauge_invariant_Hilbert_space(s, dict_Gauss_law_operators_rung_Hilbert_space, set_boundary_conditions):
        """Get the matrix projector onto the rung gauge invariant Hilbert space.

        The null entries in the diagonal correspond to gauge invariant states.

        Parameters
        ----------
        dict_Gauss_law_operators_rung_Hilbert_space : dict
            The dictionary of Gauss law operators on the rung Hilbert space.
        set_boundary_conditions : tuple( string, list )
            string = specifies the wished boundary conditions: 'OBC', 'PBC'.
            list = contains the fixed values at the boundaries.

        Returns
        -------
        gauge_invariant_space_projector : csr_matrix, shape (d, D)
            The matrix projector, shape (D, d), where:
                d = dimension rung gauge invariant Hilbert space
                D = dimension rung Hilbert space, 2^( #links + #vertices )

        """

        gauss_law_eigenvalues = zeros( dict_Gauss_law_operators_rung_Hilbert_space['charge_density'][0].shape[0] )

        for leg_index in range(s.number_legs_ladder):
            charge    = dict_Gauss_law_operators_rung_Hilbert_space['charge_density'][leg_index]
            sz_right  = dict_Gauss_law_operators_rung_Hilbert_space['sz_link_right'][leg_index]
            sz_bottom = dict_Gauss_law_operators_rung_Hilbert_space['sz_link_rung'][leg_index]
            sz_left   = dict_Gauss_law_operators_rung_Hilbert_space['sz_link_left'][leg_index]
            sz_top    = dict_Gauss_law_operators_rung_Hilbert_space['sz_link_rung'][leg_index+1]
            gauss_law_eigenvalues += ( s.get_gauss_operator(charge, sz_right, sz_bottom, sz_left, sz_top) ).diagonal( )

        # Projector (partial-trace) over fixed boundary rung links.
        if set_boundary_conditions[0] == 'OBC':
            lowest_link =    set_boundary_conditions[1][0]
            uppermost_link = set_boundary_conditions[1][1]

            external_rungs_boundary_conditions = \
                abs( dict_Gauss_law_operators_rung_Hilbert_space['sz_link_rung'][0].diagonal( ) - lowest_link*0.5 )

            external_rungs_boundary_conditions += \
                abs( dict_Gauss_law_operators_rung_Hilbert_space['sz_link_rung'][-1].diagonal( ) - uppermost_link*0.5 )

        elif set_boundary_conditions[0] == 'PBC':
            external_rungs_boundary_conditions = \
                abs(   dict_Gauss_law_operators_rung_Hilbert_space['sz_link_rung'][0].diagonal( ) \
                     - dict_Gauss_law_operators_rung_Hilbert_space['sz_link_rung'][-1].diagonal( ) )

        # Make the projector onto gauge-invariant states.
        gauge_invariant_space_projector = gauss_law_eigenvalues + external_rungs_boundary_conditions

        # Gauge invariant (physical) states with required boundary conditions correspond
        # to '0' in the array gauge_invariant_space_projector. Slice out only those rows.
        # To do that, first set all non-0s to a finite number != 1, then set the 0s to 1.
        gauge_invariant_space_projector[ gauge_invariant_space_projector != 0 ] = 2
        gauge_invariant_space_projector[ gauge_invariant_space_projector == 0 ] = 1

        # sparse.diags( ) returns a matrix of class <class 'scipy.sparse.dia.dia_matrix'>.
        # Therefore, a new conversion with sparse.csr_matrix( ) is needed.
        gauge_invariant_space_projector = sparse.csr_matrix( sparse.diags( gauge_invariant_space_projector ) )[ gauge_invariant_space_projector==1 ]

        return gauge_invariant_space_projector


    def get_gauss_operator(s, charge, sz_right, sz_bottom, sz_left, sz_top):
        """Define the Gauss law matrix operator on a cross.

        The abs( ) function is needed to fix positive the eigenvalues of unphysical
        states and make sure that they do not compensate and become 0 (physical) by
        summing Gauss law operators on different legs.

        Parameters
        ----------
        charge :   nparray
        sz_right : nparray
        sz_bot :   nparray
        sz_left :  nparray
        sz_top :   nparray
            Matrix operators, acting on a vertex-4links cross.

        Returns
        -------
        gauss_operator : nparray
            The absolute value of the Gauss law matrix operator acting on a cross.

        """

        gauss_operator = abs( charge - (sz_right + sz_bottom - sz_left - sz_top) )

        return gauss_operator


    def make_gauge_invariant_operators(s, gauge_symmetry, boundary_conditions):
        """Build Hamiltonian operators in the gauge invariant Hilbert space.

        Use the function project_and_store. The operators are stored in model_operators.

        Parameters
        ----------
        boundary_conditions : string
            Specifies the wished boundary conditions: 'OBC', 'PBC'.

        Yields
        -------
        model_operators : dict
            The dict items are list of length s.translational_invariance_length.

        """
        # Identity matrix.
        s.project_and_store('idm')
        s.d = s.model_operators['idm'][0].shape[0]

        # Null matrix.
        s.model_operators['null_int64'] = sparse.csr_matrix( zeros( [s.d, s.d], dtype=npint64 ) )
        s.model_operators['null'] = sparse.csr_matrix( zeros( [s.d, s.d] ) )

        if gauge_symmetry == 'quantum_link':

            # Local particle density.
            for leg_index in range(s.number_legs_ladder):
                s.project_and_store('n_%d' %leg_index, [('vertex', leg_index, s.n)])

            # Sum rung particle density.
            s.model_operators['n'] = [ ]
            for ti_cell_index in range( s.translational_invariance_length ):
                n_tot = s.model_operators['null']
                for leg_index in range(s.number_legs_ladder):
                    n_tot += s.model_operators['n_%d' %leg_index][ti_cell_index]
                s.model_operators['n'].append( n_tot )

            # Intra-rung hopping.
            for leg_index in range(s.number_internal_rungs):
                leg_p_index = leg_index + 1
                top_rung_index_pre_projection = leg_index + 1
                s.project_and_store('rung_%d_hop_down' %leg_index, [('vertex', leg_index, s.psi_dag), ('vertex', leg_p_index, s.psi),     ('link_rung', top_rung_index_pre_projection, s.Sm)])
                s.project_and_store('rung_%d_hop_up'   %leg_index, [('vertex', leg_index, s.psi),     ('vertex', leg_p_index, s.psi_dag), ('link_rung', top_rung_index_pre_projection, s.Sp)])

            if boundary_conditions == 'PBC':
                top_leg_index = s.number_legs_ladder - 1
                top_rung_index_pre_projection = s.number_legs_ladder
                s.project_and_store('rung_%d_hop_down' %top_leg_index, [('vertex', top_leg_index, s.psi_dag), ('vertex', 0, s.psi),     ('link_rung', 0, s.Sm), ('link_rung', top_rung_index_pre_projection, s.Sm)])
                s.project_and_store('rung_%d_hop_up'   %top_leg_index, [('vertex', top_leg_index, s.psi),     ('vertex', 0, s.psi_dag), ('link_rung', 0, s.Sp), ('link_rung', top_rung_index_pre_projection, s.Sp)])

            # Inter-rungs hopping.
            for leg_index in range(s.number_legs_ladder):
                s.project_and_store('hop_off_leg_%d_towards_right' %leg_index, [('vertex', leg_index, s.psi),     ('link_right', leg_index, s.Sm)])
                s.project_and_store('hop_off_leg_%d_towards_left'  %leg_index, [('vertex', leg_index, s.psi),     ('link_left',  leg_index, s.Sp)])
                s.project_and_store('hop_in_leg_%d_from_right'     %leg_index, [('vertex', leg_index, s.psi_dag), ('link_right', leg_index, s.Sp)])
                s.project_and_store('hop_in_leg_%d_from_left'      %leg_index, [('vertex', leg_index, s.psi_dag), ('link_left',  leg_index, s.Sm)])

        # Electric field legs ...
        for leg_index in range(s.number_legs_ladder):
            s.project_and_store('sz_left_%d'  %leg_index, [('link_left',  leg_index, s.Sz)])
            s.project_and_store('sz_right_%d' %leg_index, [('link_right', leg_index, s.Sz)])
        # ... and rungs.
        for rung_index_post_projection in range(s.number_active_rungs):
            rung_index_pre_projection = rung_index_post_projection + 1
            s.project_and_store('sz_rung_%d' %rung_index_post_projection, [('link_rung', rung_index_pre_projection, s.Sz)])

        # Rung order parameters: sum rung electric field.
        s.model_operators['magnetization_rung'] = [ ]
        for ti_cell_index in range( s.translational_invariance_length ):
            magnetization_rung = s.model_operators['null']
            for rung_index_post_projection in range(s.number_active_rungs):
                magnetization_rung += s.model_operators['sz_rung_%d' %rung_index_post_projection][ti_cell_index]
            s.model_operators['magnetization_rung'].append( magnetization_rung / s.number_active_rungs )

        # Particle density imbalance.
        s.model_operators['particle_density_imbalance'] = [ ]
        for ti_cell_index in range( s.translational_invariance_length ):
            particle_density_imbalance = s.model_operators['null']
            for leg_index in range(s.number_legs_ladder):
                particle_density_imbalance += (-1)**leg_index*(-1)**ti_cell_index*s.model_operators['n_%d' %leg_index][ti_cell_index]
            s.model_operators['particle_density_imbalance'].append( particle_density_imbalance )

        # Horizontal gauge field sz.
        for leg_index in range(s.number_legs_ladder):
            s.model_operators['horizontal_gauge_field_sz_leg_%d' %leg_index] = [ ]
            for ti_cell_index in range( s.translational_invariance_length ):
                s.model_operators['horizontal_gauge_field_sz_leg_%d' %leg_index].append(
                    s.model_operators['sz_right_%d' %leg_index][ti_cell_index] )

        # Particle density.
        for leg_index in range(s.number_legs_ladder):
            s.model_operators['particle_density_leg_%d' %leg_index] = [ ]
            for ti_cell_index in range( s.translational_invariance_length ):
                s.model_operators['particle_density_leg_%d' %leg_index].append(
                    s.model_operators['n_%d' %leg_index][ti_cell_index] )

        # Vertical gauge field sz.
        for rung_index_post_projection in range(s.number_active_rungs):
            s.model_operators['vertical_gauge_field_sz_rung_%d' %rung_index_post_projection] = [ ]
            for ti_cell_index in range( s.translational_invariance_length ):
                s.model_operators['vertical_gauge_field_sz_rung_%d' %rung_index_post_projection].append(
                    s.model_operators['sz_rung_%d' %rung_index_post_projection][ti_cell_index] )

        # Ring exchange terms.
        for leg_index in range(s.number_internal_rungs):
            s.project_and_store('ring_exchange_clock_left_rung_%d' %leg_index, [
                ('link_right', leg_index,   s.Sm),
                ('link_rung',  leg_index+1, s.Sm),
                ('link_right', leg_index+1, s.Sp) ]
            )
            s.project_and_store('ring_exchange_clock_right_rung_%d' %leg_index, [
                ('link_left', leg_index,   s.Sm),
                ('link_rung', leg_index+1, s.Sp),
                ('link_left', leg_index+1, s.Sp) ]
            )
            s.project_and_store('ring_exchange_anticlock_left_rung_%d' %leg_index, [
                ('link_right', leg_index,   s.Sp),
                ('link_rung',  leg_index+1, s.Sp),
                ('link_right', leg_index+1, s.Sm) ]
            )
            s.project_and_store('ring_exchange_anticlock_right_rung_%d' %leg_index, [
                ('link_left', leg_index,   s.Sp),
                ('link_rung', leg_index+1, s.Sm),
                ('link_left', leg_index+1, s.Sm) ]
            )
        if boundary_conditions == 'PBC':
            top_leg_index = s.number_legs_ladder - 1
            top_rung_index = s.number_active_rungs
            s.project_and_store('ring_exchange_clock_left_rung_%d' %top_leg_index, [
                ('link_right', top_leg_index,  s.Sm),
                ('link_rung',  top_rung_index, s.Sm),
                ('link_rung',  0, s.Sm),
                ('link_right', 0, s.Sp) ]
            )
            s.project_and_store('ring_exchange_clock_right_rung_%d' %top_leg_index, [
                ('link_left', top_leg_index,  s.Sm),
                ('link_rung', top_rung_index, s.Sp),
                ('link_rung', 0, s.Sp),
                ('link_left', 0, s.Sp) ]
            )
            s.project_and_store('ring_exchange_anticlock_left_rung_%d' %top_leg_index, [
                ('link_right', top_leg_index,  s.Sp),
                ('link_rung',  top_rung_index, s.Sp),
                ('link_rung',  0, s.Sp),
                ('link_right', 0, s.Sm) ]
            )
            s.project_and_store('ring_exchange_anticlock_right_rung_%d' %top_leg_index, [
                ('link_left', top_leg_index,  s.Sp),
                ('link_rung', top_rung_index, s.Sm),
                ('link_rung', 0, s.Sm),
                ('link_left', 0, s.Sm) ]
            )

        # Flippable plaquette counter term.
        for rung_index in range(s.number_active_rungs):

            s.model_operators['flippable_clock_left_rung_%d' %rung_index] = [ ]
            s.model_operators['flippable_clock_right_rung_%d' %rung_index] = [ ]
            s.model_operators['flippable_anticlock_left_rung_%d' %rung_index] = [ ]
            s.model_operators['flippable_anticlock_right_rung_%d' %rung_index] = [ ]

            for ti_cell_index in range( s.translational_invariance_length ):

                s.model_operators['flippable_anticlock_left_rung_%d' %rung_index].append(
                    s.model_operators['ring_exchange_anticlock_left_rung_%d' %rung_index][ti_cell_index] * \
                    s.model_operators['ring_exchange_clock_left_rung_%d' %rung_index][ti_cell_index]
                    )
                s.model_operators['flippable_anticlock_right_rung_%d' %rung_index].append(
                    s.model_operators['ring_exchange_anticlock_right_rung_%d' %rung_index][ti_cell_index] * \
                    s.model_operators['ring_exchange_clock_right_rung_%d' %rung_index][ti_cell_index]
                    )

                s.model_operators['flippable_clock_left_rung_%d' %rung_index].append(
                    s.model_operators['ring_exchange_clock_left_rung_%d' %rung_index][ti_cell_index] * \
                    s.model_operators['ring_exchange_anticlock_left_rung_%d' %rung_index][ti_cell_index]
                    )
                s.model_operators['flippable_clock_right_rung_%d' %rung_index].append(
                    s.model_operators['ring_exchange_clock_right_rung_%d' %rung_index][ti_cell_index] * \
                    s.model_operators['ring_exchange_anticlock_right_rung_%d' %rung_index][ti_cell_index]
                    )

        # RokhsarKivelson term.
        for rung_index in range(s.number_active_rungs):

            s.model_operators['RokhsarKivelson_left_rung_%d' %rung_index] = [ ]
            s.model_operators['RokhsarKivelson_right_rung_%d' %rung_index] = [ ]

            for ti_cell_index in range( s.translational_invariance_length ):

                s.model_operators['RokhsarKivelson_left_rung_%d' %rung_index].append(
                    s.model_operators['flippable_clock_left_rung_%d' %rung_index][ti_cell_index] + \
                    s.model_operators['flippable_anticlock_left_rung_%d' %rung_index][ti_cell_index]
                    )
                s.model_operators['RokhsarKivelson_right_rung_%d' %rung_index].append(
                    s.model_operators['flippable_clock_right_rung_%d' %rung_index][ti_cell_index] + \
                    s.model_operators['flippable_anticlock_right_rung_%d' %rung_index][ti_cell_index]
                    )

        # Legs order parameters.
        for leg_index in range(s.number_legs_ladder):
            """

            """
            s.model_operators['magnetization_leg_%d' %leg_index] = \
                [ s.model_operators['sz_left_%d' %leg_index][_] for _ in [0,1] ]
                #[ s.model_operators['sz_right_%d' %leg_index][_] for _ in [0,1] ]

            s.model_operators['string_order_leg_%d' %leg_index] =  \
                [ s.model_operators['sz_right_%d' %leg_index][_] + \
                  s.model_operators['sz_left_%d'  %leg_index][_] for _ in [0,1] ]

            s.model_operators['parity_order_leg_%d' %leg_index] = \
                [ sparse.csr_matrix( real_if_close( expm( 1.j*pi*s.model_operators['string_order_leg_%d' %leg_index][_].toarray( )))) \
                    for _ in [0,1] ]

        # Symmetry transformations.
        # s.project_and_store('particlehole_and_spinx',
        #                     a01_particle_1   = s.phi,
        #                     a02_particle_2   = s.phi,
        #                     a03_link_left_1  = s.sigmaX,
        #                     a04_link_left_2  = s.sigmaX,
        #                     a05_link_right_1 = s.sigmaX,
        #                     a06_link_right_2 = s.sigmaX,
        #                     a07_link_rung_1  = s.sigmaX)

        # s.project_and_store('signinversion_and_spinz',
        #                     a01_particle_1   = -s.idm,
        #                     a02_particle_2   = -s.idm,
        #                     a03_link_left_1  = s.sigmaZ,
        #                     a04_link_left_2  = s.sigmaZ,
        #                     a05_link_right_1 = s.sigmaZ,
        #                     a06_link_right_2 = s.sigmaZ,
        #                     a07_link_rung_1  = -s.sigmaZ )

        # s.project_and_store('particlehole_and_pix',
        #                     a01_particle_1   = s.phi,
        #                     a02_particle_2   = s.phi,
        #                     a03_link_left_1  = s.pi_rotation_x,
        #                     a04_link_left_2  = s.pi_rotation_x,
        #                     a05_link_right_1 = s.pi_rotation_x,
        #                     a06_link_right_2 = s.pi_rotation_x,
        #                     a07_link_rung_1  = s.pi_rotation_x)

        # s.project_and_store('signinversion_and_piz',
        #                     a01_particle_1   = -s.idm,
        #                     a02_particle_2   = -s.idm,
        #                     a03_link_left_1  = s.pi_rotation_z,
        #                     a04_link_left_2  = s.pi_rotation_z,
        #                     a05_link_right_1 = s.pi_rotation_z,
        #                     a06_link_right_2 = s.pi_rotation_z,
        #                     a07_link_rung_1  = -s.pi_rotation_z)

    def project_and_store(s, key, list_operators_on_rung=[]):
        """Project the Hamiltonian operator onto the gauge invariant Hilbert space.

        Call the function make_operator_rung_full_Hilbert_space to do the projection.

        Parameters
        ----------
        key : string
            The name of the Hamiltonian operator.

        list_operators_on_rung : list
            The list of tuples specifying which local operator acts on which vexter/link
            of the rung.

        Yields
        -------
        model_operators : dict
            The dict items are list of length s.translational_invariance_length.

        """
        s.model_operators[ key ] = \
            [ s.gis_projector[_].dot( s.make_operator_rung_full_Hilbert_space( list_operators_on_rung ).dot( s.gis_projector[_].T ) )
              for _ in range( s.translational_invariance_length ) ]


    def make_operators_for_links_alignment(s):
        """Defines the quantum numbers associated to the lateral links.

        The Hilbert space on a rung is redundant because the set of lateral links are
        shared with nearest-neighboring rungs. This can be exploited to enforce
        Abelian-like symmetries (strict overlap between the sets of two adjacent rungs).

        Map the spins on the left and right side of the rung to a univocal value, multiplying
        by an increasing order of magnitude the Sz operators on the links, bottom up.
        Ex: a rung has links on the right side bottom = 0.5, centre = -0.5, top = -0.5
        ---> 0.5*2 - 0.5*20 - 0.5*200 = 1 - 10 - 100 = - 89

        Parameters
        ----------
        None

        Yields
        -------
        model_operators['links_set_left/right']

        """
        opt_name = 'links_set_left'
        s.model_operators[ opt_name ] = []
        for ti_cell_index in range( s.translational_invariance_length ):
            sum_weighted_links = s.model_operators['null_int64']
            for leg_index in range(s.number_legs_ladder):
                sum_weighted_links -= ( s.model_operators['sz_left_%d' %leg_index][ti_cell_index]*2*10**leg_index ).astype( npint64 )
            s.model_operators[ opt_name ].append( sum_weighted_links )

        opt_name = 'links_set_right'
        s.model_operators[ opt_name ] = []
        for ti_cell_index in range( s.translational_invariance_length ):
            sum_weighted_links = s.model_operators['null_int64']
            for leg_index in range(s.number_legs_ladder):
                sum_weighted_links += ( s.model_operators['sz_right_%d' %leg_index][ti_cell_index]*2*10**leg_index ).astype( npint64 )
            s.model_operators[ opt_name ].append( sum_weighted_links )


    # =====================================================================================
    # Display the rung states space, printing sorted expectation values of sites and links.
    # =====================================================================================
    def display_rung_states_space(s, gauge_symmetry, boundary_conditions):
        """Description.

        Verbose.

        Parameters
        ----------
        param : type
            Description.

        Returns
        -------
        param : type
            Description.

        """
        for ti_cell_index in range( s.translational_invariance_length ):
            s.printb("\nSite:", ti_cell_index)

            string = ''

            if gauge_symmetry == 'quantum_link':
                for leg_index in range(s.number_legs_ladder):
                    string += 'n_%d\t' %leg_index
                string += '\t'

            for leg_index in range(s.number_legs_ladder):
                string += 'sz_left_%d\t' %leg_index

            for leg_index in range(s.number_legs_ladder):
                string += 'sz_right_%d\t' %leg_index

            for rung_index in range(s.number_active_rungs):
                string += 'sz_rung_%d\t' %rung_index

            s.printb( 'state\t', string )

            for i in range(s.d):
                string = ''

                if gauge_symmetry == 'quantum_link':
                    for leg_index in range(s.number_legs_ladder):
                        string += '%+2d\t\t' %(s.model_operators['n_%d' %leg_index][ti_cell_index][i,i])

                for leg_index in range(s.number_legs_ladder):
                    string += '%+2.1f\t\t' %(s.model_operators['sz_left_%d' %leg_index][ti_cell_index][i,i])

                for leg_index in range(s.number_legs_ladder):
                    string += '%+2.1f\t\t' %(s.model_operators['sz_right_%d' %leg_index][ti_cell_index][i,i])

                for rung_index in range(s.number_active_rungs):
                    string += '%+2.1f\t\t' %(s.model_operators['sz_rung_%d' %rung_index][ti_cell_index][i,i])

                s.printb( '%d\t' %i, string )


"""
Number of states for OBCs/PBCs, for 2, 3 and 4 legs:
OBC - 18, 90, 468
PBC - 28, 132, 712

SPT open boundary conditions, namely the (unique) ones for each rung size
which shows at least SPT-like string/parity order: upup for 2 and 4 legs
and updown for 3 legs.

boundary_conditions == 'upup': # 18 states, 66 states, 468 states
    bot_even = -1 # pointing upwards
    top_even = -1 # pointing upwards
    bot_odd =   1 # pointing downwards
    top_odd =   1 # pointing downwards

boundary_conditions == 'downdown': # 10 states, 66 states, 244 states
    bot_even =  1 # pointing downwards
    top_even =  1 # pointing downwards
    bot_odd =  -1 # pointing upwards
    top_odd =  -1 # pointing upwards

boundary_conditions == 'updown': # 12 states, 90 states, 336 states
    bot_even = -1 # pointing upwards
    top_even =  1 # pointing downwards
    bot_odd =   1 # pointing downwards
    top_odd =  -1 # pointing upwards

boundary_conditions == 'downup': # 12 states, 46 states, 336 states
    bot_even =  1 # pointing downwards
    top_even = -1 # pointing upwards
    bot_odd =  -1 # pointing upwards
    top_odd =   1 # pointing downwards
"""




