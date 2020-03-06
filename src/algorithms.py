from slicing import cython_extract, cython_place_back, cython_extract_from_vector_to_matrix # cython_place_back_from_matrix_to_vector
from numpy import set_printoptions
from numpy import inf
float_formatter = lambda x: '%+.1E' % x if x!=0 else '%8d'%0
set_printoptions(sign=' ', formatter={'float_kind':float_formatter}, threshold=inf, linewidth=680, suppress=True )

from copy import copy
# from copy import deepcopy
# from datetime import datetime
from fractions import Fraction
from glob import glob
from numpy import abs as npabs
from numpy import allclose
from numpy import all as npall
from numpy import append as npappend
from numpy import arange
# from numpy import argmax
from numpy import argsort
from numpy import argwhere
from numpy import array as nparray
from numpy import array_equal
#from numpy import asarray
from numpy import asscalar
from numpy import atleast_2d
from numpy import conj
from numpy import concatenate
#from numpy import count_nonzero
from numpy import delete
from numpy import diag
#from numpy import diagonal
from numpy import divide
from numpy import divmod as npdivmod
from numpy import dot
from numpy import empty
from numpy import eye
# from numpy import flip
from numpy import insert
from numpy import intersect1d
# from numpy import isclose
#from numpy import isin
from numpy import issubdtype
from numpy import lcm as nplcm
from numpy import linspace
#from numpy import load as npload
from numpy import log
# from numpy import kron
from numpy import mean
#from numpy import max as npmax
# from numpy import min as npmin
from numpy import mod
from numpy import nonzero
from numpy import ones
# from numpy import pi
from numpy import prod
#from numpy import real_if_close
from numpy import reshape
from numpy import save as npsave
#from numpy import searchsorted
from numpy import setdiff1d
from numpy import sort
from numpy import sqrt
from numpy import squeeze
from numpy import sum as npsum
from numpy import tensordot as tdot
from numpy import transpose
from numpy import union1d
from numpy import unique
# from numpy import var
from numpy import where
from numpy import zeros
# from numpy import int as npint
from numpy import int64 as npint64
from numpy import float64 as npfloat64
from numpy import uint64 as npuint64
from numpy import ndarray
from numpy.linalg import norm
from numpy.linalg import solve
from numpy.random import choice
from numpy.random import permutation
from numpy.random import rand
from numpy.random import randint
from numpy.random import shuffle
from os import remove as osremove
from os import rename as osrename
from os import getpid as osgetpid
from pickle import dump, load, HIGHEST_PROTOCOL
from psutil import Process
from random import choices
from re import sub
from scipy.linalg import eigh_tridiagonal as eigtdm
# from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.io import savemat
from string import ascii_uppercase
from string import digits
from subprocess import call
from sys import exit as sysexit
from sys import exit as tmpsysexit
#from sys import platform # to use in case of errors while running on Windows
from warnings import showwarning
from time import sleep
from time import time

import scipy.sparse.linalg.eigen.arpack as arp

"""
LEGEND

Gamma[i, a, b ]       has axes (physical, virtual_left, virtual_right)
Lambda[a, b ]         has axes (virtual_left, virtual_right)
theta[a, i, j, b ]    has axes (virtual_left, physical_left, physical_right, virtual_right)
H_mpo[v, w, i*, i ]   has axes (MPO_left, MPO_right, physical_bot, physical_top)
bond_Hamiltonian[i*, j*, i, j ] has axes (bottom_left, bottom_right, top_left, top_right)
ltm[a, a*, v ]         has axes ( )
rtm[b, b*, w ]         has axes ( )

V   is the   MPO virtual dimension
d   is the   physical dimension

Infinite DMRG legend:   Gamma[ 0 ]*Lambda[ 0 ]*Gamma[ 1 ]*Lambda[ 1 ]
Finite DMRG legend:     Lambda[ 0 ]*Gamma[ 1 ]*Lambda[ 1 ]* ... *Lambda[L-1 ]*Gamma[L-1 ]*Lambda[L ]

The charge indices acquire positive sign when outgoing and viceversa.
The physical charges are obetained and stored in their absolute value. So are
the virtual charges, obtained as outgoing vectors (from left to right), hence
with positive sign in front.

The equation for the charge conservation reads:

\sum( signed_vectors ) = 0 --->
\sum( - abs(ingoing_vectors) + abs(outgoing_vectors) = 0 --->
\sum( abs(outgoing_vectors) ) = \sum( abs(ingoing_vectors) )
\sum( as_stored_outgoing_vectors ) = \sum( as_stored_ingoing_vectors )

==============================================================================


==============================================================================

get_standard_self_attributes_info

single_site_expectation_value_list( s.model_operators[name_onsite_opt ])
  -->  expectation_value_site_i
calculate_average_mpo_chain_expectation_value( )

==============================================================================

. declare_charge_indices

. ground_state_Heff_abelian
    . update_abelian_matvec_and_svd_objects
        . get_matvec_fused_legs_charges
            . generate_list_charges
            . generate_list_charges x 12
            . generate_list_charges
        . get_central_charges_sets
        . update_svd_charge_indices
            . generate_list_charges
            . generate_list_charges
    . lanczos_algorithm
        .two_sites_extract_valid_entries

==============================================================================

. list_cen_charges = [ ]
    - Contains LATTICE_UNIT_CELL_SIZE lists. Each list contains four nparrays returned by
      generate_list_charges, one for each matvec step. It may be only one.
. list_row_charges = [{ } ]
. list_col_charges = [{ } ]

. valid_central_charges = [ ]
. dict_cen_charges = [{ }, { }, { }, { } ]
. dict_row_charges = [{ }, { }, { }, { } ]
. dict_col_charges = [{ }, { }, { }, { } ]

REMARK: in steps 1 and 2, only list_row_charges depend on the virtual charges,
        hence it needs not to be updated. One could test whether the virtual
        charges have been changed: in case not, get_matvec_fused_legs_charges can be
        skipped.

==============================================================================

!!! The name of the timer corresponds to the function profiled. !!!

. ITERATION_TIMER: from the beginning to the end, including lowest-gs search,
                    updates, physical quantities analysis, print.
    . LOWEST_EV_TIMER

==============================================================================
"""






# ============================================================================================================================================================
# Class with general methods and attributes for both (i)DMRG and (i)TEBD.
# ============================================================================================================================================================
class General_Attributes_Methods:
    """
    A class used to represent an Animal

    ...

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(s):
        """
        Parameters
        ----------
        name : str
            The name of the animal
        sound : str
            The sound the animal makes
        num_legs : int, optional
            The number of legs the animal (default is 4)
        """
        s.initialize_MPS_objects( )


    def initialize_MPS_objects(s, get_mpo_blocks=True):
        """Gets and prints the spreadsheet's header columns

        Parameters
        ----------
        file_loc : str
            The file location of the spreadsheet
        print_cols : bool, optional
            A flag used to print the columns to the console (default is
            False)

        Returns
        -------
        list
            a list of strings used that are the header columns
        """
        s.initialize_reloadable_objects( )
        s.initialize_basic_objects( )
        s.initialize_run_objects( )

        # Prepare the Hamiltonian as an MPO and a bond operator.
        s.printb('Making the needed matrix product operators.' )
        s.open_model_operators( )
        s.make_matrix_product_operators( )

        # Generate the indices for a charge-wise matmul.
        if s.ABELIAN_SYMMETRIES:
            s.initialize_physical_charges( )

        # Load or initialize the state: Lambda, Gamma, ltm, rtm...
        if s.initial_state_filename == '':
            s.internal_streamline_current_step_tag = 'iDMRG'
            s.initialize_state( )
        else:
            s.load_initial_state( )

        if s.ABELIAN_SYMMETRIES:
            s.initialize_mpo_charges( )
            # Extract the blocks of the Hamiltonian mpo for the matvec and to update the transfer matrices.
            if get_mpo_blocks:
                s.get_matrix_product_operator_blocks_and_indices_per_charges_set( )

        # In case a previous state with size L is to be used as a starting
        # point to search the ground state of a chain with larger size.
        s.prepare_transfer_matrices_warmup_sweep_or_inspection( )

        # Set the pivot at the centre of the chain.
        s.update_indices( )


# =============================================================================
# Quantum state and mpos initialization, including Abelian symmetries.
# =============================================================================
    def initialize_state(s):
        """
        Initialize the self objects:
          - s.gamma_list = [ nparray, nparray ]
          - s.lambda_list = [ nparray, nparray, nparray ]
        Define the length of the initial state: for iDMRG, the length is 2, for DMRG is REQUIRED_CHAIN_LENGTH.
        """
        s.make_pure_initial_state( )
        s.set_new_state_ticket( )


    def set_new_state_ticket(s):
        """Associate a ticket to the current state.

        If the state is reloaded from a data file on the same class object, it
        is convenient to flag it with a new ticket. Objects that depend on the
        state - for instance, transfer matrices - have a ticket too. The tickets
        mismatch indicates if the objects need to be reloaded.
        The ticket is an integer between 0 and 1.E9.

        """
        s.state_ticket = randint(1.E9)


    def make_pure_initial_state(s):
        """Initialize the self objects:
          - s.gamma_list = [ nparray, nparray ]
          - s.lambda_list = [ nparray, nparray, nparray ]
        """
        s.gamma_list = [ ]
        s.lambda_list = [ nparray([1.]) for _ in range(s.INITIAL_STATE_LENGTH+1) ]
        s.update_indices( pivot_index=1 )
        LISM = len(s.INITIAL_STATE_MATRIX)

        # If not defined, get the unit cell configuration for the initial state.
        if s.ABELIAN_SYMMETRIES:

            if LISM == 2:
                # If a configuration is defined by the user, that has priority ...
                s.check_twosites_state_charge_manifold( )

            elif LISM == 0:
                # ... otherwise, pick a random product state from the proper total charge manifold.
                if s.INITIAL_STATE_LENGTH == 2:
                    s.get_abelian_initial_state_matrix_twosites( )
                elif s.INITIAL_STATE_LENGTH == s.REQUIRED_CHAIN_LENGTH:
                    s.get_abelian_initial_state_matrix_nsites( )
                s.printb('\nThe initial product state is a random one in the manifold with the specified total charge:\n%s.' %s.INITIAL_STATE_MATRIX )

        else:
            if LISM == 0:
                s.INITIAL_STATE_MATRIX = list(randint(s.d, size=s.INITIAL_STATE_LENGTH))

        LISM = len(s.INITIAL_STATE_MATRIX)

        # Fill gamma_list with the configuration of states defined in s.INITIAL_STATE_MATRIX.
        for i in range( s.INITIAL_STATE_LENGTH ):
            s.gamma_list.append( zeros([s.d, 1, 1]) )
            s.gamma_list[i][ s.INITIAL_STATE_MATRIX[ mod(i,LISM) ], 0, 0 ] = 1

        # Fill the arrays of initial virtual charges.
        if s.ABELIAN_SYMMETRIES:

            # The virtual charges, for the finite chain grown from iDMRG.
            s.virtual_charges_normalized = { }

            if s.INITIAL_STATE_LENGTH == 2:

                for symmetry in s.LIST_SYMMETRIES_NAMES:

                    left_index =  mod(0, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE)
                    right_index = mod(1, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE)
                    left_state =  s.INITIAL_STATE_MATRIX[ 0 ]
                    right_state = s.INITIAL_STATE_MATRIX[ 1 ]

                    if symmetry == 'links_alignment':

                        zero_charge = nparray([ 0 ], dtype=npint64)
                        virtual_charge_left = - nparray( [ s.physical_charges_normalized['links_set_left'][ left_index ][ left_state ] ], dtype=npint64 )
                        virtual_charge_centre = nparray( [ s.physical_charges_normalized['links_set_right'][ left_index ][ left_state ] ], dtype=npint64 )
                        virtual_charge_right =  nparray( [ s.physical_charges_normalized['links_set_right'][ right_index ][ right_state ] ], dtype=npint64 )

                        s.virtual_charges_normalized['links_alignment'] = [ ]
                        s.virtual_charges_normalized['links_alignment'].append( virtual_charge_left )
                        s.virtual_charges_normalized['links_alignment'].append( virtual_charge_centre )
                        s.virtual_charges_normalized['links_alignment'].append( virtual_charge_right )

                    else:

                        physical_charges_normalized = s.physical_charges_normalized[ symmetry ]
                        incoming_virtual_charge = 0
                        incoming_physical_charge = physical_charges_normalized[ 0 ][ s.INITIAL_STATE_MATRIX[ 0 ] ]

                        zero_charge = nparray([ 0 ], dtype=npint64)
                        virtual_charge_centre = nparray([ incoming_physical_charge + incoming_virtual_charge ], dtype=npint64)

                        s.virtual_charges_normalized[ symmetry ] = [ ]
                        s.virtual_charges_normalized[ symmetry ].append( zero_charge )
                        s.virtual_charges_normalized[ symmetry ].append( virtual_charge_centre )
                        s.virtual_charges_normalized[ symmetry ].append( zero_charge )

            elif s.INITIAL_STATE_LENGTH == s.REQUIRED_CHAIN_LENGTH:

                for symmetry in s.LIST_SYMMETRIES_NAMES:

                    physical_charges_normalized = s.physical_charges_normalized[ symmetry ]
                    s.virtual_charges_normalized[ symmetry ] = [ nparray([ 0 ], dtype=npint64) ]
                    for _ in range(s.REQUIRED_CHAIN_LENGTH):

                        incoming_physical_charge = physical_charges_normalized[ 0 ][ s.INITIAL_STATE_MATRIX[ mod(_,LISM) ] ]
                        incoming_virtual_charge = s.virtual_charges_normalized[ symmetry ][_][0]
                        outgoing_virtual_charge = nparray([ incoming_physical_charge + incoming_virtual_charge ], dtype=npint64)
                        s.virtual_charges_normalized[ symmetry ].append( outgoing_virtual_charge )

            else:
                print('INITIAL_STATE_LENGTH is neither ==2 nor ==REQUIRED_CHAIN_LENGTH.')
                sysexit(0)


    def check_twosites_state_charge_manifold(s):

        # In case of abelian symmetry, make sure that the user-defined product
        # state is in the right charge manifold.
        left =  mod( 0, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE )
        right = mod( 1, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE )

        s.desired_charge_respected = True

        for symmetry in s.LIST_SYMMETRIES_NAMES:

            if symmetry == 'links_alignment':
                average_charge_per_site = 0

                # Internal links alignment.
                physical_charges = [ s.physical_charges['links_set_right'][ left ], s.physical_charges['links_set_left'][ right ] ]
                s.check_initial_state_charge( symmetry, physical_charges, average_charge_per_site )

                # External links alignment.
                physical_charges = [ s.physical_charges['links_set_left'][ left ], s.physical_charges['links_set_right'][ right ] ]
                s.check_initial_state_charge( symmetry, physical_charges, average_charge_per_site )

            else:
                physical_charges = s.physical_charges[ symmetry ]
                average_charge_per_site = s.AVERAGE_CHARGE_PER_SITE[ symmetry ]
                s.check_initial_state_charge( symmetry, physical_charges, average_charge_per_site )

        if not s.desired_charge_respected:
            sysexit('The user-defined initial product state does not have the desired two-sites charge.')
            sysexit(0)


    def check_initial_state_charge(s, symmetry, physical_charges, average_charge_per_site ):
        """

        """
        total_charge = 0
        left =  mod(0, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE)
        right = mod(1, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE)
        for i in range(2):
            state = s.INITIAL_STATE_MATRIX[ i ]
            index = mod(i, len( physical_charges ) )
            total_charge += physical_charges[index ][ state ]

        if total_charge == int(2*average_charge_per_site ):
            s.printb('\nThe total charge of the two sites user-defined initial product state for the operator %s is %d.' %(symmetry, average_charge_per_site*2))

        else:
            s.printb('Two-sites physical charge:', average_charge_per_site*2)
            s.printb('State site 0:', s.INITIAL_STATE_MATRIX[ 0 ], 'Physical charge:', symmetry, physical_charges[ left ][ s.INITIAL_STATE_MATRIX[ 0 ] ])
            s.printb('State site 1:', s.INITIAL_STATE_MATRIX[ 1 ], 'Physical charge:', symmetry, physical_charges[ right ][ s.INITIAL_STATE_MATRIX[ 1 ] ]);
            s.desired_charge_respected = False

        s.printb('State site 0:', s.INITIAL_STATE_MATRIX[ 0 ], 'Physical charge:', symmetry, physical_charges[ left ][ s.INITIAL_STATE_MATRIX[ 0 ] ])
        s.printb('State site 1:', s.INITIAL_STATE_MATRIX[ 1 ], 'Physical charge:', symmetry, physical_charges[ right ][ s.INITIAL_STATE_MATRIX[ 1 ] ]);


    def ias_get_charges_and_populations_matrices(s):
        num_syms = len(s.LIST_SYMMETRIES_NAMES)
        Cmat = zeros(2*num_syms, npint64) # matrix charges low and high
        Pmat = zeros(2*num_syms, npint64) # matrix charges populations

        for i in range(num_syms):
            symmetry = s.LIST_SYMMETRIES_NAMES[i]
            index = 2*i

            AVG_CHARGE = s.TOTAL_CHARGE[ symmetry ] / s.REQUIRED_CHAIN_LENGTH
            unique_physical_charges = unique( s.physical_charges[ symmetry ][0] )

            CL = unique_physical_charges[unique_physical_charges <= AVG_CHARGE].max( )
            CH = unique_physical_charges[unique_physical_charges >= AVG_CHARGE].min( )
            Cmat[index] = CL
            Cmat[index + 1] = CH

            if CL == CH:
                PL = PH = s.REQUIRED_CHAIN_LENGTH
            else:
                M = nparray( [[CL, CH], [1, 1]] )
                V = nparray( [s.TOTAL_CHARGE[ symmetry ], s.REQUIRED_CHAIN_LENGTH] )
                PL, PH = solve(M, V).astype( npint64 )

            Pmat[index] = PL
            Pmat[index + 1] = PH

        return Cmat, Pmat


    def get_physical_charges_matrix(s):
        num_syms = len(s.LIST_SYMMETRIES_NAMES)
        pcm = zeros([num_syms, s.d], npint64) # physical charges matrix
        for sym_index in range(num_syms):
            symmetry = s.LIST_SYMMETRIES_NAMES[sym_index]
            pcm[sym_index] = s.physical_charges[symmetry][0]
        return pcm


    def get_unique_valid_pcm(s, pcm, Cmat):
        pcm_mask = zeros( pcm.shape, npint64 )
        num_syms = pcm.shape[0]
        for i in range(num_syms):
            pcm_mask[i][where( (pcm[i] == Cmat[2*i]) | (pcm[i] == Cmat[2*i+1]) )[0]] = 1
        # The unique, valid columns.
        pcm_indices = where(pcm_mask.prod(axis=0))[0]
        uvpcm = unique(pcm[:, pcm_indices], axis=1)
        return uvpcm


    def get_uvpcm_indices(s, uvpcm, pcm):
        uvpcm_ind = []
        cols = uvpcm.shape[1]
        for i in range(cols):
            ucol = uvpcm[:,i]
            ind = (pcm.T - ucol).prod(axis=1)
            uvpcm_ind.append(where(ind==0)[0])
        return uvpcm_ind


    def get_linear_system_uvpcm(s, uvpcm):
        num_syms = len(s.LIST_SYMMETRIES_NAMES)
        lin_sys = zeros( [2*num_syms, uvpcm.shape[1]], npint64 )
        for i in range(num_syms):
            uvs = unique( uvpcm[i] )
            index = 2*i
            lin_sys[index][ uvpcm[i] == uvs[0] ] = 1
            if len(uvs) == 2:
                lin_sys[index+1][ uvpcm[i] == uvs[1] ] = 1
        return lin_sys


    def solve_linear_system(s, lin_sys, Pmat):
        # Create a joint matrix ab, where a*x = b.
        ab = concatenate( (lin_sys, atleast_2d(Pmat).T), axis=1 )
        # Remove null rows associated to symmetries with commensurate total charge.
        not_null_rows = where( lin_sys.any(axis=1) )[0]
        ab = ab[not_null_rows, :]
        # If everything is correct, there should be no linearly dependent ab row
        # at this stage. Check if this is the case and if not raise a warning.
        if unique( ab, axis=0 ).shape[0] != ab.shape[0]:
            print('Warning! The linear system should have no linearly dependent rows. Check this.')
            sysexit(0)
        # If lin_sys is not square (or ab is not [N,N+1]), discard some rows or cols.
        delta = ab.shape[1] - ab.shape[0] - 1
        if delta > 0:
            # Discard delta cols of a
            # DANGER: it may be that some fundamental charges sets are discarded.
            a = ab[:, :-1]
            b = ab[:, -1]
            a = a[:, :-delta]
        elif delta < 0:
            # Discard delta rows
            ab = ab[:-delta, :]
            a = ab[:, :-1]
            b = ab[:, -1]
        else:
            a = ab[:, :-1]
            b = ab[:, -1]
        return solve(a, b).astype( npint64 )


    def get_shuffles_valid_Abelian_chain_product_state(s, population, uvpcm_ind):
        rdp = empty(0, npint64) # random degeneracy population
        for i in range(len(population)):
            pop = population[i]
            appendix = choice( uvpcm_ind[i], size=pop )
            rdp = npappend(rdp, appendix)
        s.INITIAL_STATE_MATRIX = permutation( rdp )


    def get_abelian_initial_state_matrix_nsites(s):
        """For each symmetry, identify the average charge per site, which may
        be fractional - in case the total charge is incommensurate with the
        chain length. If fractional, identify the closest charges CL and CH below
        and above the average charge and extract the associated degeneracy indices.

        Calculate the indices x and y, such that
        : CL*x + CH*y = TOTAL_CHARGE
        : x + y = REQUIRED_CHAIN_LENGTH
        : largest x possible

        Iterate for each symmetry. At the end, get the intersection of the

        Note: for the sake of simplification, at the moment the algorithms only
        work is s.PHYSICAL_CHARGES_UNIT_CELL_SIZE==1, namely if the system is
        translational invariant (at least with respect to the conserved charges).
        Translation symmetry breaking can be implemented, if needed. In case, the
        elements to turn into lists are
            pcm, uvpcm_ind

        and a set of cycles
            for pc_index in range(s.PHYSICAL_CHARGES_UNIT_CELL_SIZE):
        need to added on most of the functions.

        """
        # Get the list including nearest-to-average lower and higher charges, for all Abelian symmetries.
        Cmat, Pmat = s.ias_get_charges_and_populations_matrices( )

        # Get the matrix of physical charges, as rows, for all Abelian symmetries.
        pcm = s.get_physical_charges_matrix( )

        # Using pcm and Cmat, get the matrix of unique columns (charges might have degeneracies) with valid charges.
        uvpcm = s.get_unique_valid_pcm( pcm, Cmat )
        # Get the relative degeneracy indices of the columns of uvpcm on pcm.
        uvpcm_ind = s.get_uvpcm_indices( uvpcm, pcm )

        # Using uvpcm, get the linear system that, solved, yields the multiple Abelian symmetries population distribution.
        lin_sys = s.get_linear_system_uvpcm( uvpcm )

        # Solve the linear system, get the population of the uvpcm columns.
        population = s.solve_linear_system( lin_sys, Pmat )

        # Some charges sets may have been discarded from uvpcm. Reduce uvpcm_ind accordingly.
        uvpcm_ind = [uvpcm_ind[_] for _ in range(len(population))]

        # Eventually, we can use uvpcm_ind and population to build a random chain state with the right total charge.
        s.get_shuffles_valid_Abelian_chain_product_state( population, uvpcm_ind )


    def get_abelian_initial_state_matrix_twosites(s):
        """

        """
        two_sites_charges_matrix = zeros([ s.d, s.d ])
        left =  mod(0, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE)
        right = mod(1, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE)

        for symmetry in s.LIST_SYMMETRIES_NAMES:

            if symmetry == 'links_alignment':
                # Internal links alignment.
                two_sites_total_charge = 0
                physical_charges = [ s.physical_charges['links_set_right'][ left ], s.physical_charges['links_set_left'][ right ] ]
                two_sites_charges_matrix = s.make_two_sites_charges_matrix( physical_charges, two_sites_charges_matrix, two_sites_total_charge )

                # External links alignment.
                two_sites_total_charge = 0
                physical_charges = [ s.physical_charges['links_set_left'][ left ], s.physical_charges['links_set_right'][ right ] ]
                two_sites_charges_matrix = s.make_two_sites_charges_matrix( physical_charges, two_sites_charges_matrix, two_sites_total_charge )

            else:
                two_sites_total_charge = int( 2*s.AVERAGE_CHARGE_PER_SITE[ symmetry ] )
                physical_charges = [ s.physical_charges[ symmetry ][ mod(_, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE) ] for _ in range(2) ]
                two_sites_charges_matrix = s.make_two_sites_charges_matrix( physical_charges, two_sites_charges_matrix, two_sites_total_charge )

        good_states = where( two_sites_charges_matrix == 0 )
        number_good_link_states = len(good_states[ 0 ])
        rnd = randint( number_good_link_states )

        s.INITIAL_STATE_MATRIX = [ good_states[0][rnd], good_states[1][rnd] ]


    def make_two_sites_charges_matrix(s, physical_charges, two_sites_charges_matrix, two_sites_total_charge ):

        mat = abs( physical_charges[ 0 ][ :, None ] + physical_charges[ 1 ][ None, : ] - two_sites_total_charge )
        two_sites_charges_matrix += mat

        return two_sites_charges_matrix


    def complicated_function_to_solve_diophantine_equation_with_strictly_positive_solutions(s):
        """
        Description ...
        """
        # Assume that there is a charge==0 and all the others are positive.
        wherearezeros = where( s.physical_charges==0 )[ 0 ]

        # Do the while until a correct population is obtained.
        the_population_obtained_was_not_acceptable = True
        while the_population_obtained_was_not_acceptable:
            total_charge_left = s.TOTAL_CHARGE

            # helper needs to contain all integers from 0 to s.d, except for physical_charges==0.
            helper = setdiff1d( linspace(0, s.d-1, s.d, dtype=npint64), wherearezeros)
            population = zeros( s.d, dtype=npint64 )

            # Do the while until all the charges are populated (succefsully or not) or a good population is obtained.
            filling_the_population_vector = True
            while filling_the_population_vector:

                # Extract and delete the index from helper.
                to_remove = randint(len(helper) )  # from 1 to max, eclude 0
                i = helper[to_remove ]

                # Get a random integer == population of the i-th charge.
                helper = delete(helper, to_remove )
                highest_local_population_possible = int(total_charge_left/s.quellatotale[ i ])
                if len(helper)==0 or highest_local_population_possible==0:
                    locpop = highest_local_population_possible
                    population[ i ] = locpop
                else:
                    locpop = randint(1, high=highest_local_population_possible+1)
                    population[ i ] = locpop

                # total_charge_left >= population*charge.
                total_charge_left -= int(npsum( s.quellatotale[ i ]*population[ i ]) ) # this was integer if CHARGES were too
                if total_charge_left == 0:

                    # The total charge has been hit.
                    if npsum(population) == s.REQUIRED_CHAIN_LENGTH:

                        # A number of states == length of the chain have been populated.
                        filling_the_population_vector = False
                        the_population_obtained_was_not_acceptable = False
                    elif npsum(population) < s.REQUIRED_CHAIN_LENGTH:

                        # Either we fill the remaining states to be populated with charge 0
                        i = choice(wherearezeros)
                        population[ i ] = s.REQUIRED_CHAIN_LENGTH - npsum(population)
                        filling_the_population_vector = False
                        the_population_obtained_was_not_acceptable = False

                elif len(helper) == 0 or (len(helper) == 1 and s.quellatotale[helper[ 0 ] ] == 0):

                    # Either we (unsuccesfully, otherwise total_charge_left would be 0) finished populating
                    # the local states, or the only state left has charge null, hence it can't top the total
                    # charge. In both case, just try again.
                    filling_the_population_vector = False
                    showwarning('The population obtained was not in the right charge sector. Do it again.', category = UserWarning, filename = '', lineno = 0)

        # A couple of tests.
        if npsum(population) != s.REQUIRED_CHAIN_LENGTH:
            showwarning('The pure state initial configuration specifies a number of local states different from the chain length.', category = UserWarning, filename = '', lineno = 0); (0)

        elif npsum(population*s.quellatotale ) != s.TOTAL_CHARGE:
            sysexit('The pure state initial configuration specifies a total charge not correct.')

        good_configuration = nparray([i for i in range(s.d) for _ in range(population[ i ]) ])
        shuffle(good_configuration)
        return good_configuration


    def initialize_physical_charges(s):
        """
        Initialize the self objects (dictionaries):
          - s.physical_charges[ symmetry ]
          - s.physical_charges_normalized[ symmetry ]
        """
        s.physical_charges = { }
        s.physical_charges_normalized = { }
        s.PHYSICAL_CHARGES_UNIT_CELL_SIZE = len( s.model_operators[ s.LIST_SYMMETRIC_OPERATORS_NAMES[0] ] )

        # In case the system is bipartite but the Hilbert space is translationally invariant over one site.
        s.reset_physycal_charges_unit_cell_size( )

        for symmetry in s.LIST_SYMMETRIC_OPERATORS_NAMES:
            s.initialize_physical_charges_commuting_operator( symmetry )


    def reset_physycal_charges_unit_cell_size(s):
        # The s.PHYSICAL_CHARGES_UNIT_CELL_SIZE is initially set to 1.
        max_phch_ucs = 0

        # If the commuting operator is translational invariant over more than one site but the operators are the same, then reduce the tr.inv. to 1 site.
        for symmetry in s.LIST_SYMMETRIC_OPERATORS_NAMES:

            average_charge_per_site = s.AVERAGE_CHARGE_PER_SITE[ symmetry ]
            commuting_operator = [ _.toarray( ).astype( npint64 ) for _ in s.model_operators[ symmetry ] ]

            if len( commuting_operator ) == 2:
                if array_equal( diag( commuting_operator[ 0 ] ), diag( commuting_operator[ 1 ] ) ):
                    max_phch_ucs = max( max_phch_ucs, 1 )
                else:
                    max_phch_ucs = max( max_phch_ucs, 2 )
            else:
                max_phch_ucs = s.PHYSICAL_CHARGES_UNIT_CELL_SIZE

        s.PHYSICAL_CHARGES_UNIT_CELL_SIZE = max_phch_ucs


    def initialize_physical_charges_commuting_operator(s, symmetry):
        """
        Make the charge index list for the physical leg (and the virtual as well, trivial).
        """
        commuting_operator = [ _.toarray( ).astype( npint64 ) for _ in s.model_operators[ symmetry ] ]

        # s.map_physical_charges( ): extract the diagonal of the single site operator
        # corresponding to the conserved quantity operator, map it to integers and
        # offset it to the total charge. Return it as s.physical_charges_normalized.
        # It is assumed that the onsite commuting_operator has only integer eigenvalues.
        s.physical_charges_normalized[ symmetry ] = [ s.map_physical_charges( commuting_operator[_], symmetry ) for _ in range( s.PHYSICAL_CHARGES_UNIT_CELL_SIZE ) ]
        s.physical_charges[ symmetry ] = [ diag( commuting_operator[_] ) for _ in range( s.PHYSICAL_CHARGES_UNIT_CELL_SIZE ) ]


    def map_physical_charges(s, commuting_operator, symmetry):
        """
        Offset the eigenvalues to the average charge per site and...
        Ex: -3/2, -1/2, 1/2, 3/2  --> -2, -1, 0, 1 with charge = 1/2
        Ex: 0, 1, 2, 3  --> -1.5, -0.5, 0.5, 1.5 with charge = 1.5
        """
        pc = diag(commuting_operator) # physical_charges

        if s.INITIAL_STATE_LENGTH == 2:
            average_charge_per_site = s.AVERAGE_CHARGE_PER_SITE[ symmetry ]
            charges = diag(commuting_operator)
            charges_offset = charges - average_charge_per_site
            if not issubdtype(charges_offset.dtype, npint64):
                # Find the set of deltas between the charges.
                charges_offset_with_zero = npappend( charges_offset, 0 )
                set_charges_delta = unique(npabs((charges_offset_with_zero[:, None ] - charges_offset_with_zero[None, : ]).flatten( ) ) )
                # Find the smallest element (besides the 0) of set_charges_delta.
                unit = set_charges_delta[ 1 ]
                if unit != 1:
                    charges_offset = charges_offset / unit
            physical_charges_normalized = charges_offset.astype( npint64 )

        elif s.INITIAL_STATE_LENGTH == s.REQUIRED_CHAIN_LENGTH:
            # If the commuting operator diagonal is not made of integers, normalize it.
            if pc.dtype != npint64:
                print('The conserved quantity operator has not integer diagonal elements. Stop the simulation here and check, the code is not tested.')
                sysexit(0)
                # Possible sources of error: Fraction reads 1.5 as 1.50000001 or so
                lcm = lcm.reduce( [Fraction(str(_)).denominator for _ in pc] )
                pc = (lcm*pc).astype(npint64)
                
            # Define the average charge per site as a Fraction object, for cleaner operations.
            N = s.TOTAL_CHARGE[ symmetry ]
            L = s.REQUIRED_CHAIN_LENGTH
            NL = Fraction(N, L)
            # Define the array of offset physical charges.
            frac_pc_offset = [(Fraction(_) - NL) for _ in pc]
            # Obtain the lowest common multiplier that normalizes the physical charges.
            lcm = nplcm.reduce([_.denominator for _ in frac_pc_offset])
            # Define the normalized physical charges, multiplying pc by the lcm.
            physical_charges_normalized = nparray([(lcm*_).numerator for _ in frac_pc_offset]).astype(npint64)

        return physical_charges_normalized


    def initialize_mpo_charges(s):
        """
        # Initialize the self object (dictionary):
        #   - mpo_charges[ s.mpo_operator_tag ][ symmetry ]
        """
        s.mpo_charges = { }
        for s.mpo_operator_tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:
            s.mpo_charges[ s.mpo_operator_tag ] = { }
            for symmetry in s.LIST_SYMMETRIES_NAMES:
                s.initialize_mpo_charges_commuting_operator( symmetry )


    def initialize_mpo_charges_commuting_operator(s, symmetry):
        """
        Define the following self objects:
            - s.mpo_charges

        Let us remove '_CHARGES_UNIT_CELL_SIZE' from the variables, for readability.
        If LATTICE==1, then necessarily also PHYSICAL==1.
        If LATTICE==2, then PHYSICAL==1 (QLM_TwoLegs) or ==2 (QLM_ThreeLegs).

        If LATTICE==1, then MPO==1 (assuming that the lattice size refers to
        both possible bipartitions of the Hilbert space (QLM) or to a bipartite
        for of the Hamiltonian, hence of the MPO.
        If LATTICE==2, then MPO==1 (QLM_TwoLegs and QLM_ThreeLegs) or MPO==2.
        REMARK! MPO==1 refers to the charges of the first row of H_MPO.
        Essentially, if the bipartition regards the Hilbert space but not the
        Hamiltonian (as for QLM_TwoLegs and QLM_ThreeLegs), then MPO==1.

        Hence: ALWAYS USE mod(xxx, RELATIVE_CHARGES_UNIT_CELL_SIZE). All this
        specific _CHARGES_UNIT_CELL_SIZE might sound pointless because the
        memory required overall is negligible, but it is a matter of principle,
        clarity and also to avoid redundant identical matrices.

        The relevant indices are those on colums. The row indices correspond
        necessarily to the column indices of the previous MPO.
        The matmul here is from the left, hence it maps a 'state' on the
        row to one on the column index.
        The first row index we enter with is the identity matrix and it
        takes us to all columns indices. 0:identity, 1:c_dag, 2:c, 3:mu.
        It has quantum number 0 and takes us to the column index charge.
        The second (and other) row index enters with any charge index and
        takes us out exclusively with charge=0, being the last column the
        only one with non-zero entries. Again, the only indices that matter
        are those on the column axes.
        The last row enters with 0 and for the same reason as above exits
        with 0 as well.

        Add an array of zeros with dimension = number of MPO columns.
        """
        number_mpo_columns = s.matrix_product_operator[ s.mpo_operator_tag ][ 0 ].shape[ 1 ]
        s.mpo_charges[ s.mpo_operator_tag ][ symmetry ] = zeros(number_mpo_columns, dtype=npint64)


        """
        We define physical_charges_matrix as the difference between the
        exit and the entry charge: if col=4\charge=-1 and row=2\charge=0,
        then physical_charges_matrix[2,4 ]=1.
        The operator in s.matrix_product_operator['hamiltonian'][ i ][ 0, x ] should
        have non-zero entries only in positions which share charge delta.
        """
        physical_charge_index = 0
        if symmetry == 'links_alignment':
            """
            The convention on the direction of the mpo charges vectors, is such
            that we refer to the column axis to set the mpo virtual charges.
            By construction of the mpo, the operators on the first row (we
            assume only nearest neighbors interaction/hopping for simplicity,
            but the concept can be extended) connect the site upon they act
            with the site on its right: hence, for the purpose of QLM alignment,
            the mpo virtual leg refers to the right-hand side set of links.

            In fact, if we set pcn_symmetry_tag = 'links_set_left', then
            s.mpo_charges['hamiltonian'][ symmetry ] = [ 0, 0, 0 .. ], the mpo first row leaves
            the left set of links untouched.
            """
            pcn_symmetry_tag = 'links_set_right'
        else:
            pcn_symmetry_tag = symmetry

        pcn = s.physical_charges_normalized[ pcn_symmetry_tag ][ physical_charge_index ]
        physical_charges_matrix = pcn[:, None] - pcn[None, :]

        # The first and last charge indices ought to have charge = 0, by definition.
        for x in range(1, number_mpo_columns - 1):
            """
            The other charges (the intermediate steps, in the automaton
            scheme ) are determined as follows. Every on-site operator
            in the Hamiltonian uniquely changes the charge by a certain
            unit: for instance, the density operator n changes it by a
            unit of 0, the annihilation operator c by a unit -1. That
            means, all and only the non-zero entries of the physical
            charges matrix associated to the operator share the same unit.
            By virtue of that, we identify that unit.
            """
            first_row_MPO = s.matrix_product_operator[ s.mpo_operator_tag ][ 0 ][ 0, x ]
            nonzero_entries_position = where( first_row_MPO )
            nonzero_entries = physical_charges_matrix[ nonzero_entries_position ]
            physical_charges_column_x = unique( nonzero_entries ).astype( npint64 )

            # It should always be only one number.
            s.mpo_charges[ s.mpo_operator_tag ][ symmetry ][ x ] = physical_charges_column_x



    # =============================================================================
    # Hamiltonian initialization, including Abelian symmetries.
    # =============================================================================
    def add_row_HMPO(s, i):
        """
        insert(a, b, c, axis=0) ---> insert in the matrix a,
        between rows b-1 and b, a row of of the form c*ones.
        """
        s.matrix_product_operator['hamiltonian'][ i ] = insert( s.matrix_product_operator['hamiltonian'][ i ], 1, 0, axis = 0)


    def add_col_HMPO(s, i):
        """insert(a, b, c, axis=1) ---> insert in the matrix a,
        between columns b-1 and b, a column of of the form c*ones.
        """
        s.matrix_product_operator['hamiltonian'][ i ] = insert( s.matrix_product_operator['hamiltonian'][ i ], 1, 0, axis = 1)


    def extend_HMPO(s):
        for i in range( s.LATTICE_UNIT_CELL_SIZE):
            # insert(a, b, c, axis=0) ---> insert in the matrix a,
            # between rows b-1 and b, a row of of the form c*ones.
            s.matrix_product_operator['hamiltonian'][ i ] = insert( s.matrix_product_operator['hamiltonian'][ i ], 1, 0, axis = 0)
            # insert(a, b, c, axis=1) ---> insert in the matrix a,
            # between columns b-1 and b, a column of of the form c*ones.
            s.matrix_product_operator['hamiltonian'][ i ] = insert( s.matrix_product_operator['hamiltonian'][ i ], 1, 0, axis = 1)


    def extend2_HMPO(s):
        for i in range( s.LATTICE_UNIT_CELL_SIZE ):
            # insert 2 rows (axis=0) of 0s in MPO_lists_dict['hamiltonian'][ i ], between sec-to-last and last rows (-1)
            s.matrix_product_operator['hamiltonian'][ i ] = insert(insert( s.matrix_product_operator['hamiltonian'][ i ], 1, 0, axis=0), 1, 0, axis=0)
            # insert 2 rows (axis=0) of 0s in MPO_lists_dict['hamiltonian'][ i ], between sec-to-last and last cols (-1)
            s.matrix_product_operator['hamiltonian'][ i ] = insert(insert( s.matrix_product_operator['hamiltonian'][ i ], 1, 0, axis=1), 1, 0, axis=1)


    def extend3_HMPO(s):
        for i in range( s.LATTICE_UNIT_CELL_SIZE):
            # insert 3 rows (axis=0) of 0s in MPO_lists_dict['hamiltonian'][ i ], between sec-to-last and last rows (-1)
            s.matrix_product_operator['hamiltonian'][ i ] = insert(insert(insert( s.matrix_product_operator['hamiltonian'][ i ], 1, 0, axis=0), 1, 0, axis=0), 1, 0, axis=0)
            # insert 3 rows (axis=0) of 0s in MPO_lists_dict['hamiltonian'][ i ], between sec-to-last and last cols (-1)
            s.matrix_product_operator['hamiltonian'][ i ] = insert(insert(insert( s.matrix_product_operator['hamiltonian'][ i ], 1, 0, axis=1), 1, 0, axis=1), 1, 0, axis=1)


    def get_dtype(s):
        """Changes the Hamiltonian type to complex, if any of the operators are such.

        """
        s.H_dtype = npfloat64
        """
        #for o in s.site_opts:
        #    if o[ 1 ].dtype == npcomplex:
        #        s.H_dtype = npcomplex
        #for o in s.two_sites_operators:
        #    if o[ 1 ].dtype == npcomplex:
        #        s.H_dtype = npcomplex
        #for o in s.matrix_product_operator['hamiltonian']:
        #    if o[ 1 ].dtype == npcomplex or o[2 ].dtype == npcomplex:
        #        s.H_dtype = npcomplex
        """


    def initialize_identity_MPO_lists_dict(s):
        """
        Initiate the Hamiltonian MPO cell: 1 if translationally invariant, 2 if bipartite...
        """
        s.matrix_product_operator['hamiltonian'] = [ ]
        s.get_dtype( )

        for i in range( s.LATTICE_UNIT_CELL_SIZE):

            s.matrix_product_operator['hamiltonian'].append(zeros((2,2,s.d,s.d), dtype=s.H_dtype ) )
            s.matrix_product_operator['hamiltonian'][ i ][ 0,0 ] = eye(s.d)
            s.matrix_product_operator['hamiltonian'][ i ][ 1,1 ] = eye(s.d)


    def make_hamiltonian_matrix_product_operator(s):
        """
        Generates the translationally invariant MPO Hamiltonian block.
        ( 1 c n )
        ( 0 0 c+)
        ( 0 0 1 )
        """
        s.initialize_identity_MPO_lists_dict( )

        #-----------------------------------------------------------------------

        # On-site operators.
        for opt_span, o1 in s.site_opts:

            if opt_span == None:
                for i in range( s.LATTICE_UNIT_CELL_SIZE):
                    s.matrix_product_operator['hamiltonian'][ i ][ 0,1 ] += o1
            else:
                s.matrix_product_operator['hamiltonian'][ opt_span ][ 0,1 ] += o1

        #-----------------------------------------------------------------------

        # Multiple-sites operators.
        for opt_span, o1, o2 in s.two_sites_operators:
            range_two_sites_opt = opt_span[ 0 ]
            leftmost_site_index = opt_span[ 1 ]

            # Nearest-neighboring-sites operators.
            if range_two_sites_opt == 'two':
                # Homogeneous lattice/term.
                if leftmost_site_index == None:
                    s.extend_HMPO( )
                    for leftmost_site_index in range(s.LATTICE_UNIT_CELL_SIZE):
                        s.matrix_product_operator['hamiltonian'][ mod(leftmost_site_index, s.LATTICE_UNIT_CELL_SIZE) ][ 0, 1 ] = o1
                        s.matrix_product_operator['hamiltonian'][ mod(leftmost_site_index + 1, s.LATTICE_UNIT_CELL_SIZE) ][ 1, -1 ] = o2
                # Inhomogeneous lattice/term.
                else:
                    index_0 = leftmost_site_index
                    index_1 = mod( leftmost_site_index + 1, s.LATTICE_UNIT_CELL_SIZE )
                    #
                    s.add_col_HMPO( index_0 )
                    s.matrix_product_operator['hamiltonian'][ index_0 ][ 0, 1 ] = o1
                    #
                    s.add_row_HMPO( index_1 )
                    s.matrix_product_operator['hamiltonian'][ index_1 ][ 1, -1 ] = o2

            # Next-nearest-neighboring-sites operators.
            elif range_two_sites_opt == 'three':
                # Homogeneous lattice/term.
                if leftmost_site_index == None:
                    s.extend2_HMPO( )
                    for leftmost_site_index in range(s.LATTICE_UNIT_CELL_SIZE):
                        s.matrix_product_operator['hamiltonian'][ mod(leftmost_site_index,     s.LATTICE_UNIT_CELL_SIZE) ][ 0, 1 ] = o1
                        s.matrix_product_operator['hamiltonian'][ mod(leftmost_site_index + 1, s.LATTICE_UNIT_CELL_SIZE) ][ 1, 2 ] = eye(s.d)
                        s.matrix_product_operator['hamiltonian'][ mod(leftmost_site_index + 2, s.LATTICE_UNIT_CELL_SIZE) ][ 2, -1 ] = o2
                # Inhomogeneous lattice/term.
                else:
                    index_0 = leftmost_site_index
                    index_1 = mod( leftmost_site_index + 1, s.LATTICE_UNIT_CELL_SIZE )
                    index_2 = mod( leftmost_site_index + 2, s.LATTICE_UNIT_CELL_SIZE )
                    #
                    s.add_col_HMPO( index_0 )
                    s.matrix_product_operator['hamiltonian'][ index_0 ][ 0, 1 ] = o1
                    #
                    s.add_row_HMPO( index_1 )
                    s.add_col_HMPO( index_1 )
                    s.matrix_product_operator['hamiltonian'][ index_1 ][ 1, 1 ] = eye(s.d)
                    #
                    s.add_row_HMPO( index_2 )
                    s.matrix_product_operator['hamiltonian'][ index_2 ][ 1, -1 ] = o2

            # Second-next-nearest-neighboring-sites operators - four sites apart.
            elif range_two_sites_opt == 'four':
                # Homogeneous lattice/term.
                if leftmost_site_index == None:
                    s.extend3_HMPO( )
                    for leftmost_site_index in range(s.LATTICE_UNIT_CELL_SIZE):
                        s.matrix_product_operator['hamiltonian'][ mod(leftmost_site_index,     s.LATTICE_UNIT_CELL_SIZE) ][ 0, 1 ] = o1
                        s.matrix_product_operator['hamiltonian'][ mod(leftmost_site_index + 1, s.LATTICE_UNIT_CELL_SIZE) ][ 1, 2 ] = eye(s.d)
                        s.matrix_product_operator['hamiltonian'][ mod(leftmost_site_index + 2, s.LATTICE_UNIT_CELL_SIZE) ][ 2, 3 ] = eye(s.d)
                        s.matrix_product_operator['hamiltonian'][ mod(leftmost_site_index + 3, s.LATTICE_UNIT_CELL_SIZE) ][ 3, -1 ] = o2
                # Inhomogeneous lattice/term.
                else:
                    index_0 = leftmost_site_index
                    index_1 = mod( leftmost_site_index + 1, s.LATTICE_UNIT_CELL_SIZE )
                    index_2 = mod( leftmost_site_index + 2, s.LATTICE_UNIT_CELL_SIZE )
                    index_3 = mod( leftmost_site_index + 3, s.LATTICE_UNIT_CELL_SIZE )
                    #
                    s.add_col_HMPO( index_0 )
                    s.matrix_product_operator['hamiltonian'][ index_0 ][ 0, 1 ] = o1
                    #
                    s.add_row_HMPO( index_1 )
                    s.add_col_HMPO( index_1 )
                    s.matrix_product_operator['hamiltonian'][ index_1 ][ 1, 1 ] = eye(s.d)
                    #
                    s.add_row_HMPO( index_2 )
                    s.add_col_HMPO( index_2 )
                    s.matrix_product_operator['hamiltonian'][ index_2 ][ 1, 1 ] = eye(s.d)
                    #
                    s.add_row_HMPO( index_3 )
                    s.matrix_product_operator['hamiltonian'][ index_3 ][ 1, -1 ] = o2

        #-----------------------------------------------------------------------

        s.V = s.matrix_product_operator['hamiltonian'][ 0 ].shape[ 0 ]
        # Add this leg dimension to the list of virtual legs dimensions.
        s.mpo_virtual_leg_dimension['hamiltonian'] = s.V


    # =============================================================================
    # Quantum state interrogation and information output.
    # =============================================================================
    def first_interrogation(s):
        """
        The relevance of info_tags is the following: it determinses the order
        by which the data is sorted and output. The alternatives would be
        something like .keys( ), on which one has low degree of control.
        """
        if s.main_streamline_current_step_tag == 'iDMRG_step':

            s.info_tags =  ['length',       'chi',       'double_deg'] + s.DATA_COLUMNS_TAG + ['t_elapsed',       'Lanczos']
            s.info_type =  {'length': '%d', 'chi': '%d', 'double_deg': '%s',                   't_elapsed': '%s', 'Lanczos': '%s' }
            HEADLINE = [ ]
            for tag in s.info_tags:
                if '_mid' in tag:
                    tag += '\t\t'
                HEADLINE.append( tag )
            s.printb( HEADLINE )

        elif s.main_streamline_current_step_tag == 'DMRG_sweep':

            s.info_tags =  ['sweeps',       'step',       'direction',       'chi',       'double_deg'] + s.DATA_COLUMNS_TAG + ['t_elapsed',       'Lanczos']
            s.info_type =  {'sweeps': '%d', 'step': '%d', 'direction': '%s', 'chi': '%d', 'double_deg': '%s',                   't_elapsed': '%s', 'Lanczos': '%s' }
            HEADLINE = [ ]
            for tag in s.info_tags:
                if '_mid' in tag:
                    tag += '\t\t'
                HEADLINE.append( tag )
            s.printb( HEADLINE )

        else:

            s.print_only_post_run('Here, specify first_interrogation' ); sysexit( 0 )

        # Attach the to type dictionary all items coming from the user-specified sought-after physical quantities.
        s.info_type.update(s.INFOSTREAM_OPERATORS_SUMMED_OVER_ALL_SITES)
        s.info_type.update(s.SELF_ATTRIBUTES)
        central_pair = { }
        for key in s.INFOSTREAM_OPERATORS_ACTING_ON_CENTRAL_SITES:
            central_pair[key+'_mid'] = '%s'
        s.info_type.update(central_pair)

        # Do the same with the values.
        s.info_value = dict.fromkeys(s.info_type, 0)

        # First interrogation.
        s.interrogate( )


    def get_standard_self_attributes_info(s):

        if s.main_streamline_current_step_tag == 'iDMRG_step':
            # During the warmup part.
            s.info_value['length'] = len( s.gamma_list )
            s.info_value['chi'] =    len( s.lambda_list[ s.lambda_index_centre ])

        elif s.main_streamline_current_step_tag == 'DMRG_sweep':
            # During the sweep part.
            s.info_value['sweeps'] =    s.sweeps_counter
            s.info_value['step'] =      s.sweep_index
            s.info_value['direction'] = s.sweep_direction
            s.info_value['chi'] =       len( s.lambda_list[ s.lambda_index_centre ])

        # Check if a Lanczos minimization was performed.
        s.info_value['Lanczos'] = s.using_lanczos

        # Check if the Schimdt spectrum at the centre of the chain is doubly
        # degenerate: 1 yes, 0 no, 2 boh.
        x = sort( s.lambda_list[ s.lambda_index_centre ] )
        parity = mod(len(x), 2)

        if parity == 0 and len(x)>1:

            odd_elements =  x[::2 ]
            even_elements = x[ 1::2 ]

            if allclose( odd_elements, even_elements, rtol=1e-02, atol=1e-04):
                s.entanglement_spectrum_double_degeneracy = 'yes'

            else:
                s.entanglement_spectrum_double_degeneracy = 'no'

        else:
            s.entanglement_spectrum_double_degeneracy = 'odd'

        s.info_value['double_deg'] = s.entanglement_spectrum_double_degeneracy


    def get_requested_self_attributes_info(s):
        # Append the tag of some additional user-specified properties to display.
        for tag in s.SELF_ATTRIBUTES_TAGS:
            s.info_value[ tag ] = getattr(s, tag)


    def interrogate(s):

        timer = time( )
        blanks = 6

        s.get_standard_self_attributes_info( )
        s.get_requested_expectation_value_info( )
        s.get_requested_self_attributes_info( )

        # Append the time elapsed during the iteration.
        s.info_value['t_elapsed'] = s.time_conversion( time( ) - s.iteration_time_elapsed )

        # Collect the data into a string.
        string = ''
        for tag in s.info_tags:
            string += ' '*blanks + s.info_type[ tag ] % s.info_value[ tag ]
        string = string[4: ]

        s.latest_stdout_info_string = string

        # Print it.
        s.printb( string )
        s.INTERROG_TIMER = time( ) - timer


    def get_requested_expectation_value_info(s):
        """
        Calculate the onsite/two_sites/mpo expectation values.

        """
        # Average of the expectation value of an operator over the entire chain.
        for s.mpo_operator_tag in s.NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_ALL_SITES:
            s.info_value[ s.mpo_operator_tag ] = s.calculate_average_mpo_chain_expectation_value( )

        # Average of the expectation value of an operator over the central pair of sites.
        for s.mpo_operator_tag in s.NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_CENTRAL_SITES:
            s.info_value[ s.mpo_operator_tag + '_mid'] = '  '.join('%.3f' % _ for _ in s.expectation_value_central_pair_MPO( ) )


    # =========================================================================
    # Functions to display information.
    # =========================================================================
    def print_only_post_run(s, *args, blanks=4):
        if s.POST_RUN_INSPECTION:
            string = s.get_string_from_args( args, blanks )
            print( string )


    def printb_also_post_run(s, msg):
        s.printb(msg)
        if s.POST_RUN_INSPECTION:
            print( msg )


    def printb(s, *args, blanks=4):
        if not s.POST_RUN_INSPECTION:
            string = s.get_string_from_args( args, blanks )
            print( string )

            if s.LOCAL_RUN:
                s.print_txt( string )

            else:
                s.list_stdout_strings.append( string )
                # Print to file every s.STDOUT_FLUSH_TIME_INTERVAL.
                if (time( ) - s.STDOUT_FLUSH_TIMER)/60 > s.STDOUT_FLUSH_TIME_INTERVAL:
                    s.print_multiple_lines( s.list_stdout_strings)
                    s.list_stdout_strings = [ ]
                    s.STDOUT_FLUSH_TIMER = time( )


    def get_string_from_args(s, args, blanks):
        string = ''
        for arg in args:
            if isinstance(arg, list):
                for _ in arg:
                    string += str(_) + str(' '*blanks)
            else:
                string += str(arg) + str(' '*blanks)
        return string


    def print_all_ranks(s, *args):
        """
        Function that prints to file, appending element-wise strings contained in the list argument.
        """
        string = '\t'.join(str(_) for _ in args)
        with open( s.STATE_DATAFILE_PATH + '.txt', 'a') as my_file:
            my_file.write(string + '\n')


    def print_txt(s, *args):
        """
        Function that prints to file, appending element-wise strings contained in the list argument.
        """
        string = '\t'.join(str(_) for _ in args)
        with open( s.STATE_DATAFILE_PATH + '.txt', 'a') as my_file:
            my_file.write(string + '\n')


    def print_multiple_lines(s, string_list):
        """
        Function that prints to file, appending element-wise strings contained in the list argument.
        """
        with open( s.STATE_DATAFILE_PATH + '.txt', 'a' ) as my_file:
            for _ in string_list:
                my_file.write(_ + '\n')


    #==========================================================================
    # Correlations and expectation values.
    #==========================================================================
    def open_model_operators(s):
        """

        """
        s.nparray_operator = { }
        for operator_tag in s.NAMES_NORMAL_MATRIX_OPERATORS_FOR_CORRELATIONS_AND_LOCAL_EXPECTATION_VALUES:
            s.nparray_operator[ operator_tag ] = [ _.toarray( ) for _ in s.model_operators[ operator_tag ] ]


    def make_matrix_product_operators(s):
        """

        """
        s.matrix_product_operator = { }

        for s.mpo_operator_tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:

            if s.mpo_operator_tag == 'hamiltonian':
                s.make_hamiltonian_matrix_product_operator( )

            elif 'align' in s.mpo_operator_tag:
                tag = s.mpo_operator_tag[-4: ]
                s.make_hmpo_alignment(tag)

            else:
                s.make_mpo_onsite_operator( )


    def make_hmpo_alignment(s, location):
        """
        Generates the translationally invariant MPO Hamiltonian block.
        [[ 1, szl,  0  ]
         [ 0,  0 , szr ]
         [ 0,  0 ,  1  ] ]
        Initialize the on-site MPO.
        """
        HMPO_name = 'links_alignment' + location

        s.matrix_product_operator[HMPO_name ] = [zeros((3, 3, s.d, s.d), dtype=s.H_dtype ) for _ in range( s.LATTICE_UNIT_CELL_SIZE) ]
        sz_right = [_.toarray( ) for _ in s.model_operators['sz_right' + location ] ]
        sz_left =  [_.toarray( ) for _ in s.model_operators['sz_left' + location ] ]

        for i in range( s.LATTICE_UNIT_CELL_SIZE ):
            s.matrix_product_operator[HMPO_name ][ i ][0, 0] = eye(s.d)
            s.matrix_product_operator[HMPO_name ][ i ][2, 2] = eye(s.d)
            s.matrix_product_operator[HMPO_name ][ i ][0, 1] = sz_right[ i ]
            s.matrix_product_operator[HMPO_name ][ i ][1, 2] = sz_left[ i ]


    def make_mpo_onsite_operator(s):
        my_opt = [ _.toarray( ) for _ in s.model_operators[ s.mpo_operator_tag ] ]
        # Initialize the on-site MPO.
        s.matrix_product_operator[ s.mpo_operator_tag ] = [zeros((2, 2, s.d, s.d), dtype = s.H_dtype ) for _ in range( s.LATTICE_UNIT_CELL_SIZE) ]

        for i in range( s.LATTICE_UNIT_CELL_SIZE):
            s.matrix_product_operator[ s.mpo_operator_tag ][ i ][0, 0] = eye(s.d)
            s.matrix_product_operator[ s.mpo_operator_tag ][ i ][1, 1] = eye(s.d)
            s.matrix_product_operator[ s.mpo_operator_tag ][ i ][0, 1] = my_opt[ i ]


    def calculate_average_mpo_chain_expectation_value(s):
        """

        """
        try:
            x = s.calculate_mpo_chain_expectation_value( )
        except:
            x = 0

        # This is a rather poor criterion to identify and separate an on-site operator from a multiple-sites one.
        if s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ] > 2:
            if s.mpo_operator_tag == 'hamiltonian':
                x /= len( s.gamma_list )
                s.energy = x

                if hasattr(s, 'lowest_eigenvalue_lanczos') and s.energy != 0 and s.lowest_eigenvalue_lanczos != 0:
                    relative_error_krylov_eigenvalue = abs( ( s.energy - s.lowest_eigenvalue_lanczos ) / min( s.energy, s.lowest_eigenvalue_lanczos ) )
                    s.rek_value = relative_error_krylov_eigenvalue
                    # Reminder: in the info string, we do not print the lowest Lanczos, but s.energy.
                    del s.lowest_eigenvalue_lanczos
            else:
                x /= len( s.gamma_list ) - 1
                s.printb('what? the MPO leg is bigger than 2 but it is not the hamiltonian', x); sysexit( 0 )
        else:
            x /= len( s.gamma_list )

        return x


    def expectation_value_central_pair_MPO(s):

        V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ]
        ltm = zeros(V_mpo); ltm[ 0 ] = 1
        rtm = zeros(V_mpo); rtm[- 1 ] = 1

        # Left mpo
        mpo_left = squeeze( tdot( ltm, s.matrix_product_operator[ s.mpo_operator_tag ][ s.mpo_index_left ], axes=(0,0) ) )
        opt_left = squeeze( tdot( rtm, mpo_left, axes=(0,0) ) )

        # Right mpo
        mpo_right = squeeze( tdot( s.matrix_product_operator[ s.mpo_operator_tag ][ s.mpo_index_right ], rtm, axes=(1,0) ) )
        opt_right = squeeze( tdot( ltm, mpo_right, axes=(0,0) ) )

        # Only left.
        lgl = tdot( s.gamma_list[ s.gamma_index_left ], diag( s.lambda_list[ s.lambda_index_centre ] ), axes=(2,0) )
        x = tdot( opt_left, lgl, axes=(1, 0) )
        x = tdot( x, conj( lgl ), axes=( [ 0,1,2 ], [ 0,1,2 ] ) )
        left_value = asscalar( x )

        # Only right.
        lgl = tdot( s.gamma_list[ s.gamma_index_right ], diag( s.lambda_list[ s.lambda_index_centre ] ), axes=(1,0) )
        x = tdot( opt_right, lgl, axes=(1, 0) )
        x = tdot( x, conj( lgl ), axes=( [ 0,1,2 ], [ 0,1,2 ] ) )
        right_value = asscalar( x )

        # Both.
        both = [ left_value, right_value ]

        if s.main_streamline_current_step_tag == 'iDMRG_step':

            if s.warmup_step > 0:

                index = int( ( len( s.list_one_site_expectation_values[ s.mpo_operator_tag ] ) + 1 ) / 2 )
                s.list_one_site_expectation_values[ s.mpo_operator_tag ].insert(index, left_value ) # Insert gamma_left_lg in the center ..

                index = int( ( len( s.list_one_site_expectation_values[ s.mpo_operator_tag ] ) + 1 ) / 2 )
                s.list_one_site_expectation_values[ s.mpo_operator_tag ].insert(index, right_value ) # .. and gamma_right_gl to its right.

        elif s.main_streamline_current_step_tag == 'DMRG_sweep':

            s.list_one_site_expectation_values[ s.mpo_operator_tag ][ s.gamma_index_left ] =  left_value
            s.list_one_site_expectation_values[ s.mpo_operator_tag ][ s.gamma_index_right ] = right_value

        return both


    # =========================================================================
    # mpo contractions
    # =========================================================================
    def calculate_mpo_chain_expectation_value(s):

        if s.warmup_step == 0:
            ltm = s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ 0 ]
            rtm = s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ 0 ]
            l = diag( s.lambda_list[ 1 ])
            lg =   s.gamma_list[ 0 ]
            gl =   s.gamma_list[ 1 ]
            mpo1 = s.matrix_product_operator[ s.mpo_operator_tag ][ 0 ]
            mpo2 = s.matrix_product_operator[ s.mpo_operator_tag ][ mod(1, s.LATTICE_UNIT_CELL_SIZE) ]
            expv = s.contract_mpo_ltm_lg_l_gl_rtm(ltm, lg, l, gl, rtm, mpo1, mpo2)

        else:
            ltm_index = s.ltm_index_internal
            rtm_index = s.rtm_index_internal
            l =   diag( s.lambda_list[ s.lambda_index_centre ] )
            ltm = s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ ltm_index ]
            rtm = s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ rtm_index ]
            expv = s.contract_mpo_ltm_l_rtm( ltm, l, rtm )

        return expv


    def contract_mpo_ltm_lg_l_gl_rtm(s, ltm, lg, l, gl, rtm, mpo1, mpo2):
        expv = tdot( ltm, lg, axes = (0,1) )
        expv = tdot( expv, conj(lg), axes = (0,1) )
        expv = tdot( expv, mpo1, axes = ([ 1,0,3 ],[3,0,2 ]) )
        expv = tdot( expv, l, axes = (0,0) )
        expv = tdot( expv, l, axes = (0,0) )
        expv = transpose( expv, (1,2,0) )
        expv = tdot( expv, gl, axes = (0,1) )
        expv = tdot( expv, conj(gl), axes = (0,1) )
        expv = tdot( expv, mpo2, axes = ([ 1,0,3 ],[3,0,2 ]) )
        expv = asscalar( tdot( expv, rtm, axes = ([ 0,1,2 ],[ 0,1,2 ]) ) )
        return expv


    def contract_mpo_ltm_l_rtm(s, ltm, l, rtm):
        if len(l.shape) == 1: l = diag(l)
        ltm_l = tdot( ltm, l, axes=(0,0) )
        l_rtm = tdot( rtm, l, axes=(1,0) )
        expv = asscalar( tdot( ltm_l, l_rtm, axes=([ 0,1,2 ],[2,1,0 ]) ) )
        return expv

    # ================================ Not Used ================================

    def contract_mpo_ltm_lg_l_rtm(s, ltm, lg, l, rtm, mpo):
        lgl = tdot( lg, l, axes=(2,0) )
        expv = s.contract_mpo_ltm_lgl_rtm(ltm, lgl, rtm, mpo)
        return expv


    def contract_mpo_ltm_l_gl_rtm(s, ltm, l, gl, rtm, mpo):
        lgl = tdot( l, gl, axes=(1,1) )
        lgl = transpose(lgl, (1,0,2) )
        expv = s.contract_mpo_ltm_lgl_rtm(ltm, lgl, rtm, mpo)
        return expv


    def contract_mpo_ltm_lgl_rtm(s, ltm, lgl, rtm, mpo):
        expv = tdot( ltm, lgl, axes = (0,1) )
        expv = tdot( expv, conj(lgl), axes = (0,1) )
        expv = tdot( expv, mpo, axes = ([ 1,0,3 ],[3,0,2 ]) )
        expv = asscalar( tdot( expv, rtm, axes = ([ 0,1,2 ],[ 0,1,2 ]) ) )
        return expv


    def single_site_expectation_value_hmpo_list(s):
        expv_list = [ ]
        for i in range(len( s.gamma_list ) ):
            expv_list.append( s.expectation_value_site_i_MPO( i ) )
        return expv_list


    # =========================================================================
    # opt contractions
    # =========================================================================
    def contract_opt_ltm_l_rtm(s, ltm, l, rtm):
        ltm_l = tdot( ltm, l, axes=(1,0))
        l_rtm = tdot( l, rtm, axes=(1,0))
        expv = asscalar( tdot( ltm_l, l_rtm, axes=([ 0,1 ],[ 0,1 ])))
        return expv


    def contract_opt_lg_opt_rtm(s, lgamma, opt, rtm):
        expv = tdot( opt, lgamma, axes=(1,0))
        expv = tdot( expv, conj( lgamma ), axes=([ 0,1 ],[ 0,1 ]))
        expv = asscalar( squeeze( tdot( expv, rtm, axes=([ 0,1 ],[ 0,1 ]))))
        return expv


    def contract_opt_gl_opt_ltm(s, ltm, opt, gammal):
        expv = tdot( opt, gammal, axes=(1,0))
        expv = tdot( expv, conj( gammal ), axes=([ 0,2 ],[ 0,2 ]))
        expv = asscalar( squeeze( tdot( expv, ltm, axes=([ 0,1 ],[ 0,1 ]))))
        return expv


    def contract_opt_lg_lg_opt1_opt2_rtm(s, gamma_left, gamma_right, opt1, opt2, rtm):
        expv = tdot( opt1, gamma_left, axes=(1,0))
        expv = tdot( expv, conj(gamma_left), axes=([ 0,1 ],[ 0,1 ]))
        expv = tdot( expv, gamma_right, axes=(0,1))
        expv = tdot( expv, opt2, axes=(1,1))
        expv = tdot( expv, conj(gamma_right), axes=([2,0],[0,1]))
        expv = asscalar( squeeze( tdot( expv, rtm, axes=([ 0,1 ],[ 0,1 ]))))
        return expv


    def contract_opt_lg_l_gl_opt1_opt2(s, gamma_left, gamma_right, opt1, opt2, lambda_schmeigv):
        expv = tdot( opt1, gamma_left, axes=(1,0))
        expv = tdot( expv, conj(gamma_left), axes=([ 0,1 ],[ 0,1 ]))
        expv = tdot( expv, lambda_schmeigv, axes=(1,1))
        lgr = tdot( lambda_schmeigv, gamma_right, axes=(1,1))
        expv = tdot( expv, lgr, axes=(0,0))
        expv = tdot( expv, opt2, axes=(1,1))
        expv = asscalar( squeeze( tdot( expv, conj(gamma_right), axes=([2,0,1],[0,1,2]))))
        return expv


    def contract_opt_gl_gl_opt1_opt2_ltm(s, gamma_left, gamma_right, opt1, opt2, ltm):
        expv = tdot( ltm, gamma_left, axes=(1,1))
        expv = tdot( expv, opt1, axes=(1,1))
        expv = tdot( expv, conj(gamma_left), axes=([0,2],[1,0]))
        expv = tdot( expv, gamma_right, axes=(0,1))
        expv = tdot( expv, opt2, axes=(1,1))
        expv = asscalar( squeeze( tdot( expv, conj(gamma_right), axes=([0,2,1],[1,0,2]))))
        return expv


    # =========================================================================
    # Entanglement-related functions
    # =========================================================================
    def delta_EE(s):
        """
        Return the difference between entanglement entropies.
        """
        s.BEE0 = npsum(-2*s.lambda_list[ 0 ]**2*log( s.lambda_list[ 0 ]) )
        s.BEE1 = npsum(-2*s.lambda_list[ 1 ]**2*log( s.lambda_list[ 1 ]) )
        return npabs( s.BEE0-s.BEE1)


    def energy_variance(s):
        h2 = s.nearest_neighbor_correlation_bond_operator( s.two_sites_Hamiltonian_fourlegs_tensor_squared[ 1 ])
        h = s.nearest_neighbor_correlation_bond_operator( s.two_sites_Hamiltonian_fourlegs_tensor[ 1 ])
        return npabs(h2-h**2)


    def entanglement_entropy(s):
        """
        # Return the bipartite entanglement entropy.
        """
        s.EE0 = npsum(-2*s.lambda_list[ 0 ]**2*log( s.lambda_list[ 0 ]) )
        s.EE1 = npsum(-2*s.lambda_list[ 1 ]**2*log( s.lambda_list[ 1 ]) )
        return mean([ s.EE0,s.EE1 ])


    def chain_centre_schmidt_values(s):
        return sort( s.lambda_list[ s.current_chain_centre_index ])


    def entanglement_gap(s):
        chi=len( s.lambda_list[ 0 ])
        mask=mod(arange(chi),2)*2-1
        egap=npsum( s.lambda_list[ 0 ]*mask)
        return egap


    def schmidt_norm(s):
        return sum( s.lambda_list[ 0 ]**2)


    # =============================================================================
    # Data and quantum state storage / load.
    # =============================================================================
    def store_state_as_pkl(s):
        """
        # First backup the .pkl file to overwrite, by renaming it.
        """
        temp_filename = s.STATE_DATAFILE_PATH + '_temp' + '.pkl'
        killer_filename = s.STATE_DATAFILE_PATH + '_kill' + '.txt'

        try:
            osrename( s.STATE_DATAFILE_PATH + '.pkl', temp_filename )
        except FileNotFoundError:
            pass

        # Then store the new .pkl file.
        with open( s.STATE_DATAFILE_PATH + '.pkl', 'wb' ) as my_file:
            for variable_name in s.MPS_STATE_VARIABLES:
                dump( s.__dict__[ variable_name ], my_file, HIGHEST_PROTOCOL)
        try:
            osremove( temp_filename )
        except FileNotFoundError:
            pass

        # Remove all the states that share the same name as the current one (it only works once ).
        older_states = glob( sub('_\d\d\d\d\d', '', s.STATE_DATAFILE_PATH ) + '*.pkl' )
        for state in older_states:
            if state != s.STATE_DATAFILE_PATH + '.pkl':
                osremove( state )
                s.printb('The datafile %s is being removed.' %state)

        # This is to stop the run in a controlled manner through external print.
        # To do that, create a copy of the data output .txt file in the data
        # folder, with a string '_kill' appended at the end.
        # The following tries to delete the file. As long as the file is not
        # there, an exception is thrown and the next line - exit( ) - is not executed.
        try:
            osremove( killer_filename ); sysexit("Run stopped after user's external print.")
        except FileNotFoundError:
            pass


    def get_csr_shape(s, sparse_tensor):

        tensor_shape = sparse_tensor.shape
        row_dim = prod( tensor_shape[ :-1 ] )
        col_dim = tensor_shape[ -1 ]

        return row_dim, col_dim


    def convert_sparse_tensor_to_csr_dense_matrix(s, sparse_tensor):

        shape =            s.get_csr_shape( sparse_tensor )
        sparse_matrix =    reshape( sparse_tensor, shape )
        csr_dense_matrix = csr_matrix( sparse_matrix )

        return csr_dense_matrix


    def compress_central_sparse_gammas_to_dense_csr_gammas(s):

        if s.main_streamline_current_step_tag == 'iDMRG_step':

            if not s.warmup_step == 0:

                s.gamma_list[ s.gamma_index_left - 1 ]  = s.convert_sparse_tensor_to_csr_dense_matrix( s.gamma_list[ s.gamma_index_left - 1 ]  )
                s.gamma_list[ s.gamma_index_right + 1 ] = s.convert_sparse_tensor_to_csr_dense_matrix( s.gamma_list[ s.gamma_index_right + 1 ] )

        elif s.main_streamline_current_step_tag == 'DMRG_sweep':

            s.gamma_list[ s.gamma_index_left - 1 ]  = s.convert_sparse_tensor_to_csr_dense_matrix( s.gamma_list[ s.gamma_index_left - 1 ]  )
            s.gamma_list[ s.gamma_index_right + 1 ] = s.convert_sparse_tensor_to_csr_dense_matrix( s.gamma_list[ s.gamma_index_right + 1 ] )

        elif s.main_streamline_current_step_tag == 'post_run_analysis':

            s.print_only_post_run('Here, specify: compress_central_sparse_gammas_to_dense_csr_gammas' ); sysexit( 0 )


    def load_initial_state(s):

        s.printb('\nLoading the initial state: ', s.initial_state_filename, '\n')
        with open( s.initial_state_filename, 'rb') as my_file:
            for variable_name in s.MPS_STATE_VARIABLES:
                s.__dict__[ variable_name ] = load( my_file )
                # if variable_name in ['gamma_list']:
                #     sparse_matrices_list = load(my_file )
                #     s.__dict__[ variable_name ] = s.convert_list_csr_matrices_to_sparse_tensors( variable_name, sparse_matrices_list )
                # else:
                #     s.__dict__[ variable_name ] = load(my_file )

        s.current_chain_centre_index = s.get_current_chain_centre_index( )
        s.set_new_state_ticket( )


    def convert_list_csr_matrices_to_sparse_tensors(s, variable_name, sparse_matrices_list):
        """
        Function to define the shape of the tensor to retrieve.
        """

        dense_matrices_list = [ ]

        for csr_dense_matrix in sparse_matrices_list:

            shape = s.get_tensor_shape(variable_name, csr_dense_matrix)
            dense_matrix = csr_dense_matrix.toarray( )
            dense_matrix = reshape(dense_matrix, shape )
            dense_matrices_list.append( dense_matrix )

        return dense_matrices_list


    def get_tensor_shape(s, variable_name, csr_dense_matrix):

        if variable_name == 'gamma':

            shape = ( s.d, int(csr_dense_matrix.shape[ 0 ] / s.d), -1)

        elif variable_name in ['ltm','rtm']:

            row = csr_dense_matrix.shape[ 0 ]
            shape = ( int(sqrt(row) ), int(sqrt(row) ), -1)

        return shape


    def store_npy(s, arrays_list):
        """
        Function that stores some final elements to .npy file.
        """
        npsave( s.STATE_DATAFILE_PATH + '.npy', arrays_list)


    def store_for_matlab(s, filename, key, array):
        savemat(filename+'.mat',{key:array})


    def load_from_matlab(s, filename, key):
        x = load(filename+'.mat')[key ]
        return x


    def RAM(s):
        info = Process(osgetpid( ) ).memory_info( )
        mem = info[ 0 ]/1.E9
        vmem = info[ 1 ]/1.E9
        return nparray([mem, vmem ])


    # =============================================================================
    # Miscellaneous.
    # =============================================================================
    def time_benchmarking(s):
        s.printb( str(' '*4).join(_ for _ in s.TIMERS_TAGS_SHORT) )
        percentage = [ 100*s.__getattribute__(_)/s.ITERATION_TIMER for _ in s.TIMERS_TAGS ]; percentage.append(sum(percentage )-100)
        s.printb( str(' '*11).join('%d' %_ for _ in percentage ) )
        s.printb( str(' '*4).join( s.time_conversion( s.__getattribute__(_) ) for _ in s.TIMERS_TAGS), '\n' )
        s.reset_timers( )


    def time_conversion(s, t):
        if t < 1.e-9:
            string = '0 s'
            return string
        elif 1.e-9 < t < 1.e-6:
            string = '%.2f'%(t*1.e9)+' ns'
            return string
        elif 1.e-6 < t < 1.e-3:
            string = '%.2f'%(t*1.e6)+' μs'
            return string
        elif 1.e-3 < t < 1:
            string = '%.2f'%(t*1.e3)+' ms'
            return string
        elif 1 < t < 60:
            string = '%.2f'%t+' secs'
            return string
        elif 60 <= t < 3600:
            string = '%.2f'%(t/60)+' mins'
            return string
        elif 3600 <= t < 24*3600:
            string = '%.2f'%(t/3600)+' hours'
            return string


    def display_memory_size_main_elements(s):
        """
        Display the RAM consumed by the main self objects.
        """
        s.RAM_threshold = 1

        s.printb('\n************** These are the dictionaries exceeding %d Mb of RAM:' %s.RAM_threshold)
        dicts_mem = s.return_RAM_self_dictionaries( )

        s.printb('************** These are the lists exceeding %d Mb of RAM:' %s.RAM_threshold)
        lists_mem = s.return_RAM_self_lists( )

        s.printb('************** These are the arrays exceeding %d Mb of RAM:' %s.RAM_threshold)
        array_mem = s.return_RAM_self_ndarrays( )

        unit = 1E6 # 1E3 = Kb, 1E6 = Mb
        unit_name = 'Mb'
        mem_tot = int((lists_mem + array_mem + dicts_mem)/unit)
        array_mem = int(array_mem/unit)
        lists_mem = int(lists_mem/unit)
        dicts_mem = int(dicts_mem/unit)

        s.printb('************** Tot. memory estimated:\n', mem_tot, unit_name, 'lists', lists_mem, 'arrays', array_mem, 'dicts', dicts_mem, '\n')


    def return_RAM_self_dictionaries(s):
        keys_dicts = [ ]

        for variable_name in s.__dict__.keys( ):
            if type( s.__dict__[ variable_name ] ) is dict:
                keys_dicts.append( variable_name )

        mem_tot = 0

        for variable_name in keys_dicts:

            mem_var = s.return_RAM_dict( s.__dict__[ variable_name ] )
            s.print_RAM_if_above_threshold( variable_name, mem_var )
            mem_tot += mem_var

        return mem_tot


    def return_RAM_self_lists(s):
        keys_lists = [ ]

        for variable_name in s.__dict__.keys( ):
            if type( s.__dict__[ variable_name ]) is list:
                keys_lists.append( variable_name )

        mem_tot = 0

        for variable_name in keys_lists:

            mem_var = s.return_RAM_list( s.__dict__[ variable_name ] )
            s.print_RAM_if_above_threshold( variable_name, mem_var )
            mem_tot += mem_var

        return mem_tot


    def return_RAM_self_ndarrays(s):
        keys_ndarrays = [ ]

        for variable_name in s.__dict__.keys( ):
            if type( s.__dict__[ variable_name ] ) is ndarray:
                keys_ndarrays.append( variable_name )

        mem_tot = 0

        for variable_name in keys_ndarrays:

            mem_var = s.__dict__[ variable_name ].nbytes
            s.print_RAM_if_above_threshold( variable_name, mem_var )
            mem_tot += mem_var

        return mem_tot


    def print_RAM_if_above_threshold(s, variable_name, mem_var):
        mem_var = int(mem_var/1E6)

        if mem_var > s.RAM_threshold:
            s.printb(variable_name, mem_var, 'Mb')


    def return_RAM_dict(s, my_dict):
        mem_tot = 0

        for item_key, item in my_dict.items( ):

            if type( item ) is dict:
                mem_tot += s.return_RAM_dict( item )

            elif type( item ) is list:
                mem_tot += s.return_RAM_list( item )

            elif type( item ) is ndarray:
                mem_tot += item.nbytes

            elif type( item ) is csr_matrix:
                mem_tot += item.data.nbytes + item.indptr.nbytes + item.indices.nbytes

            elif type( item ) is float:
                mem_tot += 8

            elif type( item ) is npfloat64:
                mem_tot += 8

            elif type( item ) is int:
                mem_tot += 4

            elif type( item ) in [ bool, str, type(None ), tuple ]:
                pass

            else:
                s.printb('In return_RAM_dict a %s was found' %str(type(item)) )

        return mem_tot


    def return_RAM_list(s, my_list):
        mem_tot = 0

        for item in my_list:

            if type( item ) is dict:
                mem_tot += s.return_RAM_dict( item )

            elif type( item ) is list:
                mem_tot += s.return_RAM_list( item )

            elif type( item ) is ndarray:
                mem_tot += item.nbytes

            elif type( item ) is csr_matrix:
                mem_tot += item.data.nbytes + item.indptr.nbytes + item.indices.nbytes

            elif type( item ) is float:
                mem_tot += 8

            elif item.dtype in [npfloat64, npint64 ]:
                mem_tot += 8

            elif type( item ) is int:
                mem_tot += 4

            elif type( item ) in [ bool, str, type(None ), tuple ]:
                pass

            else:
                s.printb('In return_RAM_list a %s was found' %str(type(item)) )

        return mem_tot


    def display_memory_stored_elements(s):
        unit = 1E6 # 1E3 = Kb, 1E6 = Mb
        unit_name = 'Mb'

        for variable_name in ['gamma_list']:
            my_list = s.__dict__[ variable_name ]
            mem_tot = 0
            size_tot = 0

            for array in my_list:
                mem_tot += array.nbytes
                size_tot += prod(array.shape )
                numpytype = array.dtype

            mem_tot = int(mem_tot/unit)
            s.printb(variable_name, mem_tot, unit_name, size_tot, numpytype )


    # =========================================================================
    # Data analyses.
    # =========================================================================
    def get_entanglement_spectrum(s):

        x = -2*log( s.lambda_list[ s.current_mixed_canonical_form_centre_index] )
        # The entanglement spectrum must be a column vector shape ( CHI,1 ).
        entanglement_spectrum = reshape(x, (len(x), 1))

        return entanglement_spectrum


# ============================================================================================================================================================
# Class with methods and attributes for (in)finite DMRG.
# ============================================================================================================================================================
class DMRG(General_Attributes_Methods):
    """
    A class used to represent an Animal

    ...

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(s, initial_state_filename):
        """
        Parameters
        ----------
        name : str
            The name of the animal
        sound : str
            The sound the animal makes
        num_legs : int, optional
            The number of legs the animal (default is 4)
        """
        s.initial_state_filename = initial_state_filename
        General_Attributes_Methods.__init__(s)



# =============================================================================
# Initialization of run elements
# =============================================================================
    def initialize_reloadable_objects(s):
        """
        The following variables are stored in the data file (not the master).
        Here, they are initialized for a warmup step.
        If a data file is found, these variables will be updated with their
        corresponding value stored in the .pkl data file.
        """
        # This variables tells whether the mixed-canonical form has its centre at the centre of the chain.
        s.centered_mixed_canonical_form = True
        s.sweep_direction = 'C'
        s.warmup_step = 0
        s.main_streamline_current_step_tag = 'iDMRG_step'
        s.internal_streamline_current_step_tag = 'iDMRG_step'
        s.sweep_index = 0


    def initialize_basic_objects(s):
        """The dimension of the virtual leg of the MPOs.

        """
        s.mpo_virtual_leg_dimension = {
                                       'n': 2,
                                       'idm': 2,
                                       'magnetization_leg_0': 2,
                                       'magnetization_leg_1': 2,
                                       'magnetization_leg_2': 2,
                                       'magnetization_leg_3': 2,
                                       'magnetization_rung': 2,
                                       'particle_density_imbalance': 2,
                                       }

        for tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:
            if not tag in list(s.mpo_virtual_leg_dimension.keys()) + ['hamiltonian']:
                print("\n\nThe following matrix product operators are required: ", s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS)
                print("The virtual leg dimension of the operator %s was not specified. Please specify it, in algorithms.py." %tag); sysexit(0)

        s.lowest_eigenvalue_lanczos = 0


    def initialize_run_objects(s):
        # This index tells the location of the centre of the mixde-canonical form, which
        # corresponds to the chain centre during the warmup and changes during the sweep.
        s.current_chain_centre_index = 1
        s.sweeps_counter = 0
        s.sweep_unit_step = 1
        s.krylov_iterator = 0
        s.using_lanczos = True
        s.tensor_contractions_tag = 'matvec'
        s.latest_stdout_info_string = ''
        s.rek_value = 0
        s.rek_vector = 0

        # The expectation values.
        try:
            s.list_one_site_expectation_values = { }
            for tag in s.NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_CENTRAL_SITES:
                s.list_one_site_expectation_values[ tag ] = [ 0 for _ in range( s.REQUIRED_CHAIN_LENGTH ) ]
        except:
            pass

        # Some timers.
        s.initialize_timers( )
        s.iteration_time_elapsed = time( )

        # Tensor indices sorting.
        s.number_legs_contracted_tensor = { 'matvec': [ 4, 5, 5, 5 ]  ,
                                            'ltm_mpo_update': [ 3, 4, 4 ] ,
                                            'rtm_mpo_update': [ 3, 4, 4 ] ,
                                            'ltm_opt_update': [ 2, 3, 3 ] ,
                                            'rtm_opt_update': [ 2, 3, 3 ] }

        s.contracted_tensor_transposition_order = { 'matvec':     [ (1,2,3,0)   ,
                                                                    (3,1,2,4,0) ,
                                                                    (0,2,4,3,1) ,
                                                                    (0,2,4,1,3) ]     ,

                                                    'ltm_mpo_update': [ (1,2,0)     ,
                                                                        (2,0,1,3)   ,
                                                                        (0,2,1,3) ] ,

                                                    'rtm_mpo_update': [ (1,2,0)     ,
                                                                        (2,0,1,3)   ,
                                                                        (0,2,1,3) ] ,

                                                    'ltm_opt_update': [ (1,0)     ,
                                                                        (0,1,2)   ,
                                                                        (1,0,2) ] ,

                                                    'rtm_opt_update': [ (1,0)     ,
                                                                        (0,1,2)   ,
                                                                        (1,0,2) ]  }

        # Some other stuff
        s.stored_two_sites_indices = { }
        s.stored_dense_indices = { }

        s.how_many_times_each_charges_configuration_was_mapped = { }

        s.mapped_charges_configurations = { 'matvec_hamiltonian': [ ],
                                            'two_sites_svd_hamiltonian': [ ] }

        s.stored_charges_configurations = { 'matvec_hamiltonian': [ ],
                                            'two_sites_svd_hamiltonian': [ ] }

        for key in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:
            s.mapped_charges_configurations['ltm_mpo_update' + '_' + key ] = [ ]
            s.mapped_charges_configurations['rtm_mpo_update' + '_' + key ] = [ ]
            s.stored_charges_configurations['ltm_mpo_update' + '_' + key ] = [ ]
            s.stored_charges_configurations['rtm_mpo_update' + '_' + key ] = [ ]

        for key in s.NAMES_NORMAL_MATRIX_OPERATORS_FOR_CORRELATIONS_AND_LOCAL_EXPECTATION_VALUES:
            s.mapped_charges_configurations['ltm_opt_update' + '_' + key ] = [ ]
            s.mapped_charges_configurations['rtm_opt_update' + '_' + key ] = [ ]
            s.stored_charges_configurations['ltm_opt_update' + '_' + key ] = [ ]
            s.stored_charges_configurations['rtm_opt_update' + '_' + key ] = [ ]


    def initialize_timers(s):
        s.PKL_STORE_TIMER = time( )
        s.STDOUT_FLUSH_TIMER = time( )
        s.TIMERS_TAGS =[
                        'ITERATION_TIMER',
                        'FIRST_PART_TIMER',
                        'GRND_STA_TIMER',
                        'LAST_PART_TIMER',
                        #'SVD_TIMER',
                        #'TMS_UPDATE_TIMER',
                        #'INTERROG_TIMER',
                        ]

        s.TIMERS_TAGS_SHORT = [ ]
        if s.DISPLAY_TIMERS:
            s.TIMERS_TAGS_SHORT = [x.replace('_TIMER','') for x in s.TIMERS_TAGS ]
        s.reset_timers( )


    def reset_timers(s):
        """
        List of timers scattered in the code, whether or not they are displayed.
        """
        s.ITERATION_TIMER =  1
        s.FIRST_PART_TIMER = 0
        s.GRND_STA_TIMER =   0
        s.LAST_PART_TIMER =  0
        s.SVD_TIMER =        0
        s.TMS_UPDATE_TIMER = 0
        s.INTERROG_TIMER =   0


    def prepare_transfer_matrices_warmup_sweep_or_inspection(s):
        """
        # 1234567890123456789012345678901234567890123456789012345678901234567890
        If s.REQUIRED_CHAIN_LENGTH > length of the chain, a chain-growing
        algorithm (warmup) is to use. If the state is loaded, two cases are
        possible: the stored state is a centre-normalized (arising from an
        unfinished warmup) or not (arising from a unterminated sweep).
        This function is mostly to reduce the transfer matrix list (plus minor
        resets) to the form adopted in the warmup. Hence the need for
        s.sweep_direction to not be 'C', which only occurs when the sweep was
        being performed.

        """
        s.current_chain_centre_index = s.get_current_chain_centre_index( )

        """
        # 1234567890123456789012345678901234567890123456789012345678901234567890
        Once the initial state has been created/loaded, we can create the transfer
        matrices: since a warmup is required, tms are created that span up to the
        central site.
        """
        # Initialize the left/right transfer matrix dictionaries.
        s.left_mpo_transfer_matrix_list =  { }
        s.right_mpo_transfer_matrix_list = { }
        s.left_opt_transfer_matrix_list =  { }
        s.right_opt_transfer_matrix_list = { }

        s.get_current_chain_length( )
        s.get_current_chain_centre_index( )

        if s.POST_RUN_INSPECTION:

            s.REQUIRED_CHAIN_LENGTH = s.get_current_chain_length( )
            s.HALF_REQUIRED_CHAIN_LENGTH = int( s.REQUIRED_CHAIN_LENGTH / 2)
            s.BOND_DIMENSION = max( [ len(_) for _ in s.lambda_list ] )
            if not s.centered_mixed_canonical_form:
                s.print_only_post_run('The state is not in a centre-canonical form. Please, re-run.' )
            s.main_streamline_current_step_tag = 'post_run_analysis' # useless?

        else:

            if s.REQUIRED_CHAIN_LENGTH > s.get_current_chain_length( ):
                if not s.centered_mixed_canonical_form:
                    """
                    In this case, a sweep was being performed and did not stop in the centre:
                    let's retrieve a mixed-canonical form centered in the centre of the chain.
                    """
                    s.give_chain_centered_mixed_canonical_form( )

                for s.mpo_operator_tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:
                    s.printb('Getting trasfer matrix list for operator ', s.mpo_operator_tag )
                    # All transfer matrices are csr_matrix except for the inner most, which is a sparse ndarray.
                    s.generating_transfer_matrices_list = True
                    s.get_transfer_matrices_list( transfer_matrix_list_purpose='iDMRG_step' )
                    del s.generating_transfer_matrices_list

                # Reset the sweep-related indices.
                s.sweeps_counter = 0
                s.sweep_index = s.HALF_REQUIRED_CHAIN_LENGTH

                # At this stage all gammas are stored as dense csr_matrices.
                # Make the two central gammas sparse.
                if s.warmup_step > 0:
                    s.open_one_gamma( s.gamma_index_left )
                    s.open_one_gamma( s.gamma_index_right )
                    s.open_one_ltm( s.get_current_ltm_length( ) - 1 )
                    s.open_one_rtm( - s.get_current_rtm_length( ) )

                # Update some warmup-related indices.
                s.sweep_direction = 'C'
                s.main_streamline_current_step_tag = 'iDMRG_step'

            else:
                if s.NUMBER_SWEEPS > 0 or not s.centered_mixed_canonical_form:
                    for s.mpo_operator_tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:
                        s.printb('Getting trasfer matrix list for operator ', s.mpo_operator_tag )
                        # All transfer matrices are csr_matrix except for the inner most, which is a sparse ndarray.
                        s.get_transfer_matrices_list( transfer_matrix_list_purpose='DMRG_sweep' )
                    if s.sweep_direction == 'C':
                        # If the state comes from a warmup, the sweep_direction is to be set.
                        s.sweep_direction = 'R'
                        s.sweep_index = s.get_current_chain_centre_index( )
                s.main_streamline_current_step_tag = 'DMRG_sweep'
            # All transfer matrices on the left of the centre-left site are csr_matrix.
            # All gammas are csr_matrix.


    def get_current_chain_centre_index(s):
        s.current_chain_centre_index = int( s.get_current_chain_length( ) / 2)
        return s.current_chain_centre_index


    def get_current_chain_length(s):
        s.current_chain_length = len( s.gamma_list )
        return s.current_chain_length


    def check_canonical_form_at_chain_centre(s):
        s.get_current_chain_centre_index( )
        s.get_current_mixed_canonical_form_centre_index( )


    def get_current_mixed_canonical_form_centre_index(s):
        """
        This could be done by checking whether a identity ltm
        is or not an isometry: is it == lambda^2?
        """
        if s.internal_streamline_current_step_tag == 'iDMRG':
            s.current_mixed_canonical_form_centre_index = s.current_chain_centre_index

        elif s.internal_streamline_current_step_tag == 'DMRG':
            s.current_mixed_canonical_form_centre_index = s.sweep_index

        return s.current_mixed_canonical_form_centre_index


# =============================================================================
# (i)DMRG run.
# =============================================================================
    def run(s):
        """

        """
        time_beginning_run = time( )

        # The warmup (or infinite-DMRG) part grows a finite-size chain.
        if s.main_streamline_current_step_tag == 'iDMRG_step':
            s.warmup_starting_routine( ) # print some initial information
            s.internal_streamline_current_step_tag = 'iDMRG'

            # A while loop extends the chain until the sought length is reached.
            while len( s.gamma_list ) < s.REQUIRED_CHAIN_LENGTH:
                s.iDMRG_warmup_step( )

            s.conclude_warmup( )
            s.get_current_chain_length( )
            s.get_current_chain_centre_index( )

            # To perform sweeps, bridge some elements onto the finite-DMRG algorithm.
            if s.NUMBER_SWEEPS > 0:
                # Transfer matrices and gamma tensors are csr_matrix except for the innermost.
                s.bridge_infinite_DMRG_to_finite_DMRG( )
                # Transfer matrices and gammas are all csr_matrix.

        # If the warmup has been already performed - or it is not required - do the sweeps.
        if s.main_streamline_current_step_tag == 'DMRG_sweep':
            s.internal_streamline_current_step_tag = 'DMRG'

            if s.NUMBER_SWEEPS > 0:
                s.sweep_starting_routine( )
                while s.sweeps_counter < s.NUMBER_SWEEPS:
                    s.DMRG_sweep( )

            if not s.centered_mixed_canonical_form:
                """
                In this case, a sweep was being performed and did not stop in the centre:
                let's retrieve a mixed-canonical form centered in the centre of the chain.
                """
                s.sweep_starting_routine( )
                s.NUMBER_SWEEPS = 1
                s.give_chain_centered_mixed_canonical_form( )

        s.conclude_run( time_beginning_run ) # print some conclusive information
        s.measure_and_print_gamma_list_weight( )


    #==========================================================================
    # iDMRG, warmup.
    #==========================================================================
    def warmup_starting_routine(s):

        s.printb('\n********************************** Start warmup **********************************')
        s.first_interrogation( )

    def iDMRG_warmup_step(s):
        """
        Description here ...
        """
        s.iteration_time_elapsed = time( )
        first_part_timer = time( )
        # Add two sites in the center of the chain.
        s.extend_chain( )
        # Update basic run indices.
        s.update_indices( )
        # Close and open some elements according to the current pivot position.
        s.open_elements_in_use_for_DMRG( )
        s.close_elements_not_in_use( )

        if s.ABELIAN_SYMMETRIES:
            # Define the elements needed to build the dense two-sites tensor with Abelian symmetries.
            s.get_two_sites_indices_per_charges_set( )
            # Define the elements needed for the block-diagonal (Abelian) matvec.
            s.tensor_contractions_tag = 'matvec'
            s.mpo_operator_tag = 'hamiltonian'
            s.get_objects_for_abelian_tensordot( )
            # Get the mapped indices, from sorted vector to matrix sectors.
            s.load_or_map_indices_for_dense_vector_representation( )
            # Define the sectors of the two-sites matrix, shaped for the svd.
            s.get_two_sites_svd_sectors( )
            # Get the sorted two-sites vector, shaped for the Krylov space.
            s.map_two_sites_svd_sectors_to_krylov_vector( )
        else:
            vec = tdot( s.gamma_list[ s.gamma_index_left ], diag( s.lambda_list[ s.lambda_index_centre ] ), axes=(2,0) )
            vec = tdot( vec, s.gamma_list[ s.gamma_index_right ], axes=(2,1) )
            s.dense_flattened_contracted_tensor = transpose( vec, (1,0,2,3) ).flatten( )

        s.FIRST_PART_TIMER += time() - first_part_timer
        # Diagonalize the effective Hamiltonian.
        s.ground_state_Heff( ) # 0123
        last_part_timer = time()
        # Update the elements of a two-sites cell.
        s.update_two_sites_and_transfer_matrices( )
        s.LAST_PART_TIMER += time() - last_part_timer
        # Some last conclusive operations: if required, store the datafile at each step, display the timers, display the RAM.
        s.warmup_step += 1
        s.conclude_iteration( )


    def extend_chain(s):
        """
        Update the self objects:
          - s.gamma_list
          - s.lambda_list
          - s.virtual_charges_normalized[ symmetry ]
        """
        if not s.warmup_step == 0:

            lambda_external_index = s.lambda_index_left # lambda_index_right == lambda_index_left in iDMRG
            lambda_internal_index = s.lambda_index_centre
            lambda_internal = s.lambda_list[ lambda_internal_index ]
            lambda_external = s.lambda_list[ lambda_external_index ]

            # The gamma tensor to be inserted on the left-hand side of the new pair of central sites.
            gamma_left_lg = s.gamma_list[ s.gamma_index_right ]
            gamma_left_lg = tdot( gamma_left_lg, diag( s.lambda_list[ lambda_internal_index ]), axes=(1,1) )
            gamma_left_lg = tdot( gamma_left_lg, diag( s.lambda_list[ lambda_external_index ]**(-1) ), axes=(1,0) )

            # The gamma tensor to be inserted on the right-hand side of the new pair of central sites.
            gamma_right_gl = s.gamma_list[ s.gamma_index_left ]
            gamma_right_gl = tdot( gamma_right_gl, diag( s.lambda_list[ lambda_external_index ]**(-1) ), axes=(1,1) )
            gamma_right_gl = tdot( gamma_right_gl, diag( s.lambda_list[ lambda_internal_index ]), axes=(1,0) )

            # Insert the items just defined.
            index = int( ( len( s.gamma_list ) + 1 ) / 2 )
            s.gamma_list.insert( index, gamma_left_lg )  # Insert gamma_left_lg in the center ..
            index = int( ( len( s.gamma_list ) + 1 ) / 2 )
            s.gamma_list.insert( index, gamma_right_gl ) # .. and gamma_right_gl to its right.

            # Place a copy of the internal Lambda (the latest having been updated) to the left of the current internal (central) Lambda.
            index = int( ( len( s.lambda_list) )/2)
            s.lambda_list.insert(index, lambda_internal)
            index = int( ( len( s.lambda_list) )/2)
            s.lambda_list.insert(index, lambda_external)

            if s.ABELIAN_SYMMETRIES:
                # Place a copy of the external Lambda (updated during the second-to-last iteration) to the center, between the internal Lambdas.
                new_virtual_charges_centre = { }
                for symmetry in s.LIST_SYMMETRIES_NAMES:
                    new_virtual_charges_centre[ symmetry ] = empty(0, dtype=npint64)
                    virtual_charges_internal = s.virtual_charges_normalized[ symmetry ][ lambda_internal_index ]
                    virtual_charges_external = s.virtual_charges_normalized[ symmetry ][ lambda_external_index ]
                    index = int( ( len( s.virtual_charges_normalized[ symmetry ]) )/2)
                    s.virtual_charges_normalized[ symmetry ].insert(index, virtual_charges_internal)
                    index = int( ( len( s.virtual_charges_normalized[ symmetry ]) )/2)
                    s.virtual_charges_normalized[ symmetry ].insert(index, virtual_charges_external)

        # Extend the left transfer matrices.
        s.add_element_to_left_transfer_matrices( )
        s.add_element_to_right_transfer_matrices( )


    def conclude_warmup(s):
        """
        During a run in the cluster, when the warmup finished, most likely
        there are stdout strings to flush and the latest state is not stored.
        """
        # Print the accumulated list of stdout information strings, if running on cluster.
        s.print_multiple_lines( s.list_stdout_strings )
        s.list_stdout_strings = [ ]
        s.STDOUT_FLUSH_TIMER = time( )

        if s.STORE_STATE:
            s.store_state_as_pkl( )
            s.PKL_STORE_TIMER = time( )


    def bridge_infinite_DMRG_to_finite_DMRG(s):
        # Up to now, the list of left transfer matrices include the sites from
        # the leftmost to the centre-left one (analogously for the right side ).
        # For the finite-DMRG sweep we need the transfer matrices to include
        # the whole chain. In the end, the transfer matrices will have length
        # L-1 and won't include the 2 right(left)-most sites.
        s.printb('\n********************** Bridging warmup and sweep algorithms **********************')
        s.get_current_chain_length( )
        s.get_current_chain_centre_index( )

        for s.mpo_operator_tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:
            s.get_transfer_matrices_list( transfer_matrix_list_purpose='DMRG_sweep' )

        # Let the finite DMRG method start.
        s.main_streamline_current_step_tag = 'DMRG_sweep'


    #==========================================================================
    # DMRG, sweep.
    #==========================================================================
    def sweep_starting_routine(s):
        s.printb('\n\t\t\tThe chain extends to the desired length.')
        s.printb('\n*********************************** Start sweep ***********************************')

        s.update_indices( )
        # Close and open some elements according to the current pivot position.
        s.open_elements_in_use_for_DMRG( )
        s.close_elements_not_in_use( )
        s.first_interrogation( )
        if s.sweep_direction == 'L':
            s.sweep_unit_step = -1


    def DMRG_sweep(s): # x
        """
        Description here ...
        """
        s.iteration_time_elapsed = time( )
        # Update the sweep direction and the step index.
        s.update_sweep_direction_and_sweep_index( )
        # Update basic run indices.
        s.update_indices( pivot_index=s.sweep_index )
        # Close and open some elements according to the current pivot position.
        s.open_elements_in_use_for_DMRG( )
        s.close_elements_not_in_use( )
        # Prepare the site tensors to have lambda on the external side.
        s.give_addressed_sites_form_lg_gl( )

        if s.ABELIAN_SYMMETRIES:
            # Define the elements needed to build the dense two-sites tensor.
            s.get_two_sites_indices_per_charges_set( )
            # Map the theta indices per charges sets so that they draw from the dense array theta.
            s.tensor_contractions_tag = 'matvec'
            s.mpo_operator_tag = 'hamiltonian'
            s.get_objects_for_abelian_tensordot( )
            # Get the mapped indices, from sorted vector to matrix sectors.
            s.load_or_map_indices_for_dense_vector_representation( )
            # Define the sectors of the two-sites matrix, shaped for the svd.
            s.get_two_sites_svd_sectors( )
            # Get the sorted two-sites vector, shaped for the Krylov space.
            s.map_two_sites_svd_sectors_to_krylov_vector( )
        else:
            vec = tdot( s.gamma_list[ s.gamma_index_left ], diag( s.lambda_list[ s.lambda_index_centre ] ), axes=(2,0) )
            vec = tdot( vec, s.gamma_list[ s.gamma_index_right ], axes=(2,1) )
            s.dense_flattened_contracted_tensor = transpose( vec, (1,0,2,3) ).flatten( )

        # Diagonalize the effective Hamiltonian.
        s.ground_state_Heff( ) # 0123
        # Update the elements of a two-sites cell.
        s.update_two_sites_and_transfer_matrices( )
        # Some last conclusive operations: if required, store the datafile at each step, display the timers, display the RAM.
        s.update_counters_DMRG( )
        s.conclude_iteration( )
        s.close_one_ltm( s.ltm_index_external )
        s.close_one_rtm( s.rtm_index_external )


    def update_counters_DMRG(s): # x
        """
        Update the number of sweeps performed and set the variable centre_canonical.
        """
        if s.sweep_index == s.current_chain_centre_index:
            s.centered_mixed_canonical_form = True

            if s.sweep_direction == 'R':
                s.sweeps_counter += 1

        else:
            s.centered_mixed_canonical_form = False


    def update_sweep_direction_and_sweep_index(s): # x
        if s.sweep_direction == 'C':
            # If the state comes from a warmup, the sweep_direction is to be set.
            s.sweep_direction = 'R'
            s.sweep_index = s.get_current_chain_centre_index( )
            s.sweep_unit_step = 1

        # If the sweep direction, set in the previous step, is towards right,
        # then increase the sweep index by one unit (viceversa for the left).
        if s.sweep_index == s.get_current_chain_length( ) - 1:
            s.sweep_unit_step = -1
            s.sweep_direction = 'L'

        elif s.sweep_index == 1:
            s.sweep_unit_step = 1
            s.sweep_direction = 'R'

        # If the sweep direction, set in the previous step, is towards right,
        # then increase the sweep index by one unit (viceversa for the left).
        s.sweep_index += s.sweep_unit_step



    #==========================================================================
    # Shared functions within an (i)DMRG step.
    #==========================================================================
    def print_all_indices(s):

        lista = ['ltm_index_external', \
                 'ltm_index_internal', \

                 'rtm_index_internal', \
                 'rtm_index_external', \

                 'gamma_index_left', \
                 'gamma_index_right', \

                 'lambda_index_left', \
                 'lambda_index_centre', \
                 'lambda_index_right', \

                 'chi_left', \
                 'chi_centre', \
                 'chi_right', \

                 'mpo_index_left', \
                 'mpo_index_right', \

                 'physical_charges_index_left', \
                 'physical_charges_index_right']

        s.printb( )
        for elem in lista:
            s.printb( elem, getattr(s, elem) )


    def open_elements_in_use_for_DMRG(s):
        """

        """
        s.open_one_gamma( s.gamma_index_left  )
        s.open_one_gamma( s.gamma_index_right )
        for s.mpo_operator_tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:
            s.open_one_ltm( s.ltm_index_external )
            s.open_one_rtm( s.rtm_index_external )
            s.open_one_ltm( s.ltm_index_internal )
            s.open_one_rtm( s.rtm_index_internal )


    def close_elements_not_in_use(s):
        """

        """
        s.close_one_gamma( s.gamma_index_left - 1 )
        s.close_one_gamma( s.gamma_index_right + 1 )
        for s.mpo_operator_tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:
            s.close_one_ltm( s.ltm_index_external - 1 )
            s.close_one_ltm( s.ltm_index_internal + 1 )
            s.close_one_rtm( s.rtm_index_external + 1 )
            s.close_one_rtm( s.rtm_index_internal - 1 )


    def update_indices(s, pivot_index=None, gamma_list_span=0):
        """Update the self objects:
           - s.ltm_index_internal / rtm
           - s.ltm_index_external / rtm
           - s.gamma_index_left / right
           - s.lambda_index_left / centre / right
           - s.chi_left / centre / right
           - s.mpo_index_left / right
           - s.physical_charges_index_left / right

        """
        if pivot_index == None:
            pivot_index = s.get_current_chain_centre_index( )

        # Depend on pivot
        s.get_transfer_matrix_index( pivot_index, gamma_list_span )

        # Depend on pivot
        s.get_gamma_index( pivot_index )

        # Depend on gamma_index_right
        s.get_lambda_index( pivot_index )

        # Depend on lambda_index
        s.get_chi_for_two_sites_methods( )

        # Depend on gamma_index, maps onto LATTICE_UNIT_CELL_SIZE
        s.get_mpo_index( )

        if s.ABELIAN_SYMMETRIES:
            s.physical_charges_index_left =  mod( s.gamma_index_left,  s.PHYSICAL_CHARGES_UNIT_CELL_SIZE )
            s.physical_charges_index_right = mod( s.gamma_index_right, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE )


    def get_transfer_matrix_index(s, pivot_index, gamma_list_span):

        if gamma_list_span == 0:
            gamma_list_span = s.get_current_chain_length( )

        delta =  int( ( s.current_chain_length - gamma_list_span ) / 2 )

        s.ltm_index_internal = pivot_index - delta
        s.ltm_index_external = s.ltm_index_internal - 1
        s.rtm_index_internal = pivot_index - s.current_chain_length - 1 + delta
        s.rtm_index_external = s.rtm_index_internal + 1

        if s.internal_streamline_current_step_tag == 'product_onsite_opt_transfer_matrix_buildup':

            s.ltm_index_external = 0
            s.ltm_index_internal = 1
            s.rtm_index_internal = -2
            s.rtm_index_external = -1


    def get_gamma_index(s, pivot_index):

        s.gamma_index_left =  pivot_index - 1
        s.gamma_index_right = pivot_index


    def get_lambda_index(s, pivot_index):

        s.lambda_index_left =   pivot_index - 1
        s.lambda_index_centre = pivot_index
        s.lambda_index_right =  pivot_index + 1


    def get_chi_for_two_sites_methods(s):

        s.chi_left =   len( s.lambda_list[ s.lambda_index_left ]   )
        s.chi_centre = len( s.lambda_list[ s.lambda_index_centre ] )
        s.chi_right =  len( s.lambda_list[ s.lambda_index_right ]  )


    def get_mpo_index(s):

        s.mpo_index_left =  mod( s.gamma_index_left,  s.LATTICE_UNIT_CELL_SIZE)
        s.mpo_index_right = mod( s.gamma_index_right, s.LATTICE_UNIT_CELL_SIZE)


    def conclude_iteration(s):

        if s.INFO_EVERY_SWEEP_STEP or s.main_streamline_current_step_tag == 'iDMRG_step' or s.centered_mixed_canonical_form:
            # Display the information.
            s.interrogate( )
            # Store the state.
            if s.STORE_STATE:
                s.store_state( )
            # If a panorama of the RAM consumed, per element that exceeds a certain threshold.
            if s.DISPLAY_RAM:
                s.display_memory_size_main_elements( )
            # If time benchmarking is required, print this info.
            s.ITERATION_TIMER = time( ) - s.iteration_time_elapsed
            if s.DISPLAY_TIMERS:
                s.time_benchmarking( )


    def conclude_run(s, time_beginning_run):

        s.printb('\nRun concluded.')

        # Display the information.
        s.printb( s.latest_stdout_info_string )

        # Scan the chain and comrpess the gammas still in ndarray form.
        for index in range( len( s.gamma_list ) ):
            s.close_one_gamma( index )

        # Store the state.
        if s.STORE_STATE:
            s.store_state_as_pkl( )

        # If a panorama of the RAM consumed, per element that exceeds a certain threshold.
        if s.DISPLAY_RAM:
            s.display_memory_size_main_elements( )

        # If time benchmarking is required, print this info.
        s.ITERATION_TIMER = time( ) - s.iteration_time_elapsed
        if s.DISPLAY_TIMERS:
            s.time_benchmarking( )

        s.printb('\nFull-run time elapsed: ', s.time_conversion(time( ) - time_beginning_run) )

        # This must be the last thing, otherwise the line just above won't be printed on the stdout .txt file.
        if not s.LOCAL_RUN:
            s.print_multiple_lines( s.list_stdout_strings)


    def store_state(s):

        if s.LOCAL_RUN:
            s.store_state_as_pkl( )

        if not s.LOCAL_RUN and (time( ) - s.PKL_STORE_TIMER)/60 > s.PKL_STORE_TIME_INTERVAL:
            s.store_state_as_pkl( )
            s.PKL_STORE_TIMER = time( )


# =============================================================================
# Two-sites ground state search.
# =============================================================================
    def ground_state_Heff(s): # ?
        """
        Diagonalize the effective hamiltonian. The two gamma matrices are not
        supposed to be left-/right-canonical but they are expected to be in
        the form lg/gl.
        """
        gs_heff_timer = time( )

        if s.ABELIAN_SYMMETRIES:
            if s.LANCZOS_ALGORITHM == 'SCIPY':
                dim = len( s.dense_flattened_contracted_tensor )

                H = H_effective_abelian( s.central_charges_sets, \
                                         s.row_col_charges_sets, \
                                         s.contracted_tensor_blocks_shape, \
                                         s.row_col_tensor_blocks_shape, \
                                         s.indices_from_dense_flattened_tensor_to_blocks_dict, \
                                         s.hltm_blocks, \
                                         s.hmpo_blocks_for_matvec[ s.mpo_index_left ], \
                                         s.hmpo_blocks_for_matvec[ s.mpo_index_right ], \
                                         s.hrtm_blocks, \
                                         nparray( [ dim, dim ] ) )
                               
                s.lowest_eigenvalue_lanczos, s.dense_flattened_contracted_tensor = \
                    arp.eigsh( H, \
                               k=1, \
                               tol=s.SCIPY_EIGSH_TOLERANCE, \
                               which='SA', \
                               return_eigenvectors=True, \
                               v0=s.dense_flattened_contracted_tensor )

                s.dense_flattened_contracted_tensor = s.dense_flattened_contracted_tensor.flatten( )
                s.lowest_eigenvalue_lanczos = s.lowest_eigenvalue_lanczos[ 0 ]

            elif s.LANCZOS_ALGORITHM == 'HOMEMADE':
                # Find an approximation of the lowest state with the Lanczos algorithm.
                s.lanczos_algorithm( full_orthonormalization = False ) # flat, ordered 0123

        else:
            H = H_effective( s.left_mpo_transfer_matrix_list['hamiltonian'][ s.ltm_index_external ], \
                             s.right_mpo_transfer_matrix_list['hamiltonian'][ s.rtm_index_external ], \
                             s.matrix_product_operator['hamiltonian'][ s.mpo_index_left ], \
                             s.matrix_product_operator['hamiltonian'][ s.mpo_index_right ] )

            s.lowest_eigenvalue_lanczos, s.dense_flattened_contracted_tensor = \
                arp.eigsh( H, \
                           k=1, \
                           tol=s.SCIPY_EIGSH_TOLERANCE, \
                           which='SA', \
                           return_eigenvectors=True, \
                           v0=s.dense_flattened_contracted_tensor )

        s.lowest_eigenvalue_lanczos /= len( s.gamma_list )
        s.check_goodness_ground_state_vector( )

        s.GRND_STA_TIMER += time( ) - gs_heff_timer


    def lanczos_algorithm(s, full_orthonormalization = False ): # ?
        """
        Homemade implementation of the Lanczos algorithm.
        krylov_vector_v0 = initial guess
        a_i = Hamiltonian expectation value < v_i | H | v_i >
        b_i = off diagonal term < v_(i-1) | H | v_i >
        n_i = vector norm < v_i | v_i >
        """

        s.lanczos_step = 0
        s.index_wip_krylov_vectors_0 = 0
        s.index_wip_krylov_vectors_1 = 1
        s.index_wip_krylov_vectors_2 = 2

        krylov_space_dimension =     s.KRYLOV_SPACE_DIMENSION
        s.list_diagonal_entries =    [ ]
        s.list_offdiagonal_entries = [ ]
        s.latest_diagonal_entry =    0
        s.latest_offdiagonal_entry = 0

        # Restrict the Krylov set to be in the allowed charges set subspace.
        krylov_vector_size = len( s.dense_flattened_contracted_tensor )
        s.wip_krylov_vectors = zeros([3, krylov_vector_size ])
        s.wip_krylov_vectors[ s.index_wip_krylov_vectors_1 ] = s.dense_flattened_contracted_tensor

        # Caution on krylov_space_dimension at the beginning: in case the size
        # of the krylov vector is smaller than Krylov_space_dimension, we need
        # to truncate the size of the Krylov space to krylov_vector_size.
        krylov_space_dimension = min( krylov_space_dimension, krylov_vector_size )

        s.krylov_basis_set_row_vectors = zeros([ krylov_space_dimension, krylov_vector_size ])

        # Iteratively obtain the remaining Krylov vectors.
        for s.krylov_iterator in range( krylov_space_dimension ):
            # Do the matrix vector multiplication and obtain the new Krylov-basis vector.
            too_small = s.krylov_step( )
            if too_small:
                break

        del s.wip_krylov_vectors
        s.list_offdiagonal_entries.remove(0)

        # ============================== LEGEND ===============================
        # T = tridiagonal matrix.
        # S = transformation to make T diagonal.
        # s.krylov_basis_set_row_vectors = Krylov orthonormal basis.
        s.list_diagonal_entries = nparray( s.list_diagonal_entries )
        krylov_space_dimension = len( s.list_diagonal_entries )

        # The Krylov (almost)-orthonormal-columns basis is transposed.
        s.krylov_basis_set_row_vectors = s.krylov_basis_set_row_vectors[:krylov_space_dimension, : ].T # 0123
        s.tridiag_eigenvalues, s.tridiag_eigenvectors = eigtdm( nparray( s.list_diagonal_entries ), nparray( s.list_offdiagonal_entries ) )

        # We use the same variable 'krylov_basis_set_row_vectors' for the H_eff
        # matrix eigenvectors (obtained multiplying by S) for overhead reasons.
        s.krylov_basis_set_row_vectors = dot( s.krylov_basis_set_row_vectors, s.tridiag_eigenvectors ) # 0123
        s.krylov_vector_v0 = s.krylov_basis_set_row_vectors[ :, 0 ] # 0123
        s.lowest_eigenvalue_lanczos = s.tridiag_eigenvalues[ 0 ]

        # Return theta as a sparse vector, with legs sorted as 0123
        # Embed the abelian sector into two_sites_sparse.
        s.dense_flattened_contracted_tensor = s.krylov_vector_v0 # 0123


    def check_goodness_ground_state_vector(s):
        # Is the ground state a 'good' eigenstate of the effective Hamiltonian?
        # If v was an exact eigenvector, the overlap between v and H*v/norm(H*v)
        # would be equal to 1.

        # The ground-state vector in dense representation is krylov_vector_v0.
        # The function abelian tensor contraction takes dense_flattened_contracted_tensor as
        # print parameter and returns the same variable.
        if s.ABELIAN_SYMMETRIES:
            lowest_eigenvector_cache = s.dense_flattened_contracted_tensor.copy( )

            s.abelian_tensor_contraction( )
            s.dense_flattened_contracted_tensor /= norm( s.dense_flattened_contracted_tensor )

            overlap_lowest_eigenstate_vector = abs( asscalar( dot( lowest_eigenvector_cache, s.dense_flattened_contracted_tensor ) ) )
            relative_error_krylov_eigenvector = abs( 1 - overlap_lowest_eigenstate_vector) / min(1, overlap_lowest_eigenstate_vector )
            s.rek_vector = relative_error_krylov_eigenvector

            s.dense_flattened_contracted_tensor = lowest_eigenvector_cache

        else:
            x = s.two_sites_mps_state_matvec( s.dense_flattened_contracted_tensor )
            x /= norm( x )
            overlap_lowest_eigenstate_vector = abs( asscalar( dot( x, s.dense_flattened_contracted_tensor ) ) )
            relative_error_krylov_eigenvector = abs( 1 - overlap_lowest_eigenstate_vector) / min(1, overlap_lowest_eigenstate_vector )
            s.rek_vector = relative_error_krylov_eigenvector


    def krylov_step(s):
        """
        Calculate the new orthonormal vector.
        """
        if False: # Make it normal with respect to all the previous vectors.

            x = s.wip_krylov_vectors[ s.index_wip_krylov_vectors_1 ] - s.latest_diagonal_entry*s.wip_krylov_vectors[ s.index_wip_krylov_vectors_0 ] - s.latest_offdiagonal_entry*s.wip_krylov_vectors[ s.index_wip_krylov_vectors_2 ]

            for i in range( s.krylov_iterator - 2):

                x -= dot(conj( s.krylov_basis_set_row_vectors[ i ]), s.wip_krylov_vectors[ s.index_wip_krylov_vectors_1 ])*s.krylov_basis_set_row_vectors[ i ]
                # ACHTUNG! We should have stored s.wip_krylov_vectors[ s.index_wip_krylov_vectors_2 ] before the redefinition above: that's the one to be called below..
                # s.wip_krylov_vectors[ s.index_wip_krylov_vectors_2 ] -= dot(conj( s.krylov_basis_set_row_vectors[ i ]), s.wip_krylov_vectors[ s.index_wip_krylov_vectors_2 ])*s.krylov_basis_set_row_vectors[ i ]

            s.wip_krylov_vectors[ s.index_wip_krylov_vectors_1 ] = x

        else: # Make it normal with respect to the previous two vectors.
            s.wip_krylov_vectors[ s.index_wip_krylov_vectors_1 ] = s.wip_krylov_vectors[ s.index_wip_krylov_vectors_1 ] - s.latest_diagonal_entry*s.wip_krylov_vectors[ s.index_wip_krylov_vectors_0 ] - s.latest_offdiagonal_entry*s.wip_krylov_vectors[ s.index_wip_krylov_vectors_2]

        # Normalize the new basis vector.
        norm_latest_vector = norm( s.wip_krylov_vectors[ s.index_wip_krylov_vectors_1] )

        # If the norm falls below some threshold, the krylov step is interrupted
        # and the krylov matrix is returned as it is, without adding the last
        # smalle vector.
        if norm_latest_vector <= 1.E-9:
            s.printb('The norm of the latest Krylov vector is very small (%.2E),' % norm_latest_vector, 'the krylov iteration is interruped at step', s.krylov_iterator)
            return True

        s.wip_krylov_vectors[ s.index_wip_krylov_vectors_1 ] /= norm_latest_vector
        s.krylov_basis_set_row_vectors[ s.krylov_iterator ] = s.wip_krylov_vectors[ s.index_wip_krylov_vectors_1 ] # 0123, from s.krylov_vector_v2, from two_sites_sparse of matvec

        # Update the vectors.
        s.index_wip_krylov_vectors_0 = mod( s.index_wip_krylov_vectors_0 + 1, 3 )
        s.index_wip_krylov_vectors_1 = mod( s.index_wip_krylov_vectors_1 + 1, 3 )
        s.index_wip_krylov_vectors_2 = mod( s.index_wip_krylov_vectors_2 + 1, 3 )

        # Do the matrix vector multiplication.
        s.dense_flattened_contracted_tensor = s.wip_krylov_vectors[ s.index_wip_krylov_vectors_0 ].copy( )
        s.wip_krylov_vectors[ s.index_wip_krylov_vectors_1 ] = s.abelian_tensor_contraction( )

        s.lanczos_step += 1

        del s.dense_flattened_contracted_tensor

        # Obtain the diagonal and offdiagonal entries of the tridiagonal Krylov matrix.
        s.latest_diagonal_entry =    dot( conj( s.wip_krylov_vectors[ s.index_wip_krylov_vectors_0 ].T), s.wip_krylov_vectors[ s.index_wip_krylov_vectors_1 ] )
        s.latest_offdiagonal_entry = dot( conj( s.wip_krylov_vectors[ s.index_wip_krylov_vectors_2 ].T), s.wip_krylov_vectors[ s.index_wip_krylov_vectors_1 ] )

        # Store the diagonal and offdiagonal entries of the tridiagonal Krylov matrix.
        s.list_diagonal_entries.append( s.latest_diagonal_entry )
        s.list_offdiagonal_entries.append( s.latest_offdiagonal_entry )


    def abelian_tensor_contraction(s): # ?
        """
        Contract two tensors exploiting the Abelian symmetries.
        """

        if s.tensor_contractions_tag == 'matvec':

            s.abelian_tensordot( tensordot_step = 0 ) # 1) Contraction theta-ltm.
            s.abelian_tensordot( tensordot_step = 1 ) # 2) Contraction theta-hmpo1.
            s.abelian_tensordot( tensordot_step = 2 ) # 3) Contraction theta-hmpo2.
            s.abelian_tensordot( tensordot_step = 3 ) # 4) Contraction theta-rtm.

        elif s.tensor_contractions_tag in ['ltm_mpo_update', 'ltm_opt_update']:

            s.abelian_tensordot( tensordot_step = 0 ) # 1) Contraction ltm-gamma_ket.
            s.abelian_tensordot( tensordot_step = 1 ) # 2) Contraction ltm-mpo/opt.
            s.abelian_tensordot( tensordot_step = 2 ) # 3) Contraction ltm-gamma_bra.

        elif s.tensor_contractions_tag in ['rtm_mpo_update', 'rtm_opt_update']:

            s.abelian_tensordot( tensordot_step = 0 ) # 1) Contraction rtm-gamma_ket.
            s.abelian_tensordot( tensordot_step = 1 ) # 2) Contraction rtm-mpo/opt.
            s.abelian_tensordot( tensordot_step = 2 ) # 3) Contraction rtm-gamma_bra.

        return s.dense_flattened_contracted_tensor # 0123


    def abelian_tensordot(s, tensordot_step = 0):
        """
        The goal is to perform a block-wise tensor cotnraction.
        """

        matrix_diagonal_block = { }
        new_dense_flattened_contracted_tensor = empty(0)

        for charges_set in s.row_col_charges_sets[ tensordot_step ]:

            # If the set is not in central_charges_set, create an empty block.
            if charges_set not in s.central_charges_sets[ tensordot_step ]:

                matrix_diagonal_block = zeros( prod( s.row_col_tensor_blocks_shape[ tensordot_step ][ charges_set ] ) )

            else:
                # First cython part. Extract the elements from the vector and populate the blocks.
                matrix_diagonal_block = zeros( s.contracted_tensor_blocks_shape[ tensordot_step ][ charges_set ] )

                cython_extract_from_vector_to_matrix( s.dense_flattened_contracted_tensor, \
                                                      matrix_diagonal_block, \
                                                      s.indices_from_dense_flattened_tensor_to_blocks_dict[ tensordot_step ][ charges_set ] )

                if s.tensor_contractions_tag == 'matvec':

                    if tensordot_step == 0:
                        matrix_diagonal_block = dot( matrix_diagonal_block, \
                                                     s.hltm_blocks[ charges_set ] )

                    elif tensordot_step == 1:
                        matrix_diagonal_block = dot( matrix_diagonal_block, \
                                                     s.hmpo_blocks_for_matvec[ s.mpo_index_left ][ charges_set ] )

                    elif tensordot_step == 2:
                        matrix_diagonal_block = dot( matrix_diagonal_block, \
                                                     s.hmpo_blocks_for_matvec[ s.mpo_index_right ][ charges_set ] )

                    elif tensordot_step == 3:
                        matrix_diagonal_block = dot( matrix_diagonal_block, \
                                                     s.hrtm_blocks[ charges_set ] )

                elif s.tensor_contractions_tag in ['ltm_mpo_update', 'ltm_opt_update']:

                    if tensordot_step == 0:
                        matrix_diagonal_block = dot( matrix_diagonal_block, \
                                                     s.gamma_blocks_ket[ charges_set ] )

                    elif tensordot_step == 1:
                        if s.tensor_contractions_tag == 'ltm_mpo_update':
                            matrix_diagonal_block = dot( matrix_diagonal_block, \
                                                         s.mpo_blocks_for_ltm_update[ s.mpo_operator_tag ][ s.mpo_index_left ][ charges_set ] )

                        elif s.tensor_contractions_tag == 'ltm_opt_update':
                            matrix_diagonal_block = dot( matrix_diagonal_block, \
                                                         s.opt_blocks_for_ltm_update[ s.mpo_operator_tag ][ s.mpo_index_left ][ charges_set ] )

                    elif tensordot_step == 2:
                        matrix_diagonal_block = dot( matrix_diagonal_block, \
                                                     s.gamma_blocks_bra[ charges_set ] )

                elif s.tensor_contractions_tag in ['rtm_mpo_update', 'rtm_opt_update']:

                    if tensordot_step == 0:
                        matrix_diagonal_block = dot( matrix_diagonal_block, \
                                                     s.gamma_blocks_ket[ charges_set ] )

                    elif tensordot_step == 1:
                        if s.tensor_contractions_tag == 'rtm_mpo_update':
                            matrix_diagonal_block = dot( matrix_diagonal_block, \
                                                         s.mpo_blocks_for_rtm_update[ s.mpo_operator_tag ][ s.mpo_index_right ][ charges_set ] )

                        elif s.tensor_contractions_tag == 'rtm_opt_update':
                            matrix_diagonal_block = dot( matrix_diagonal_block, \
                                                         s.opt_blocks_for_rtm_update[ s.mpo_operator_tag ][ s.mpo_index_right ][ charges_set ] )

                    elif tensordot_step == 2:
                        matrix_diagonal_block = dot( matrix_diagonal_block, \
                                                     s.gamma_blocks_bra[ charges_set ] )

            new_dense_flattened_contracted_tensor = npappend( new_dense_flattened_contracted_tensor, matrix_diagonal_block.flatten( ) )

        s.dense_flattened_contracted_tensor = new_dense_flattened_contracted_tensor


    def two_sites_mps_state_matvec(s, x):
        ltm = s.left_mpo_transfer_matrix_list['hamiltonian'][ s.ltm_index_external ]
        rtm = s.right_mpo_transfer_matrix_list['hamiltonian'][ s.rtm_index_external ]
        lmpo = s.matrix_product_operator['hamiltonian'][ s.mpo_index_left ]
        rmpo = s.matrix_product_operator['hamiltonian'][ s.mpo_index_right ]
        #
        x = reshape( x, (s.chi_left, s.d, s.d, s.chi_right) )
        x = tdot( ltm, x,  axes=(0,0) )
        x = tdot( x, lmpo, axes=([1,2], [0,3]) )
        x = tdot( x, rmpo, axes=([3,1], [0,3]) )
        x = tdot( x, rtm,  axes=([1,3], [0,2]) )
        x = x.flatten( )

        return x



# =============================================================================
# SVD and state update.
# =============================================================================
    def svd_and_update_gamma_lambda_virtual_abelian(s): # 01.23
        """
        After we found the lowest-eigenvalue eigenvector theta, be it through
        zero_sites_DMRG or two_sites_DMRG, we can perform a block-wise SVD
        on the two-sites central block.
        """
        # s.two_sites_gamma_sparse_vector enters as 01x23. Do a Abelian-wise SVD on a two-sites matrix s.two_sites_gamma_sparse_vector.
        X_dict, Y_dict, Z_dict, number_Schmidt_eigenvalues_kept, smallest_Schmidt_eigenvalue_kept = s.svd_two_sites_two_sites_abelian( )

        # Define dense Gamma matrices, initialized to zeros.
        s.gamma_list[ s.gamma_index_left ]  = zeros( [ s.chi_left*s.d, number_Schmidt_eigenvalues_kept  ] )
        s.gamma_list[ s.gamma_index_right ] = zeros( [ number_Schmidt_eigenvalues_kept, s.chi_right*s.d ] )
        s.lambda_list[ s.lambda_index_centre ] = empty(0)
        new_virtual_charges_centre = { }

        for symmetry in s.LIST_SYMMETRIES_NAMES:
            new_virtual_charges_centre[ symmetry ] = empty(0, dtype=npint64)

        dict_svd_cen_charges = { }
        chi_block = { }
        init = 0

        for charges_set in s.svd_charges_sets:

            # Find the central indices to extract because of the Schmidt threshold requirement.
            # The number of S.e. (sorted in decreasing order by default) picked from this block.
            chi_block[ charges_set ] = len(Y_dict[ charges_set ][ Y_dict[ charges_set ] >= smallest_Schmidt_eigenvalue_kept ])

            # Slice out the cols of X[ charges_set ] and place them by charge in consecutive
            # blocks of Gamma_Left, defined by the following indices.
            dict_svd_cen_charges[ charges_set ] = arange(init, init+chi_block[ charges_set ]).astype( npuint64 )
            init += chi_block[ charges_set ]

        for charges_set in s.svd_charges_sets:

            # Assign the row/column indices with quantum number == charge.
            row_ind_charge = s.two_sites_fused_row_legs_indices_per_charges_set[ charges_set ]
            cen_ind_charge = dict_svd_cen_charges[ charges_set ]
            col_ind_charge = s.two_sites_fused_col_legs_indices_per_charges_set[ charges_set ]

            # If the SVDed block is not empty, we move on with the Gamma reconstruction.
            chi_block_charge = chi_block[ charges_set ]

            if chi_block_charge != 0:

                range_accepted_se_block = arange(chi_block_charge )

                # Place the entries of the U (V) matrix into Gamma left (right).
                cython_place_back( s.gamma_list[ s.gamma_index_left ],  X_dict[ charges_set ][:, range_accepted_se_block ], row_ind_charge, cen_ind_charge )
                cython_place_back( s.gamma_list[ s.gamma_index_right ], Z_dict[ charges_set ][range_accepted_se_block, : ], cen_ind_charge, col_ind_charge )
                s.lambda_list[ s.lambda_index_centre ] = npappend( s.lambda_list[ s.lambda_index_centre ], Y_dict[ charges_set ][range_accepted_se_block ] )

                for i in range( len(charges_set) ):

                    symmetry = s.LIST_SYMMETRIES_NAMES[ i ]
                    charge = charges_set[ i ]
                    new_virtual_charges_centre[ symmetry ] = npappend( new_virtual_charges_centre[ symmetry ], ones(chi_block[ charges_set ])*charge ).astype( npint64 )

        # Update the Gamma tensors, the Lambda vectors and the virtual_charges vectors.
        # Update virtual_charges_normalized[ symmetry ][internal ]
        for symmetry in s.LIST_SYMMETRIES_NAMES:
            s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ] = new_virtual_charges_centre[ symmetry ]

        # Sort the eigenvalues in decreasing order. Necessary to be consistent with the full-theta SVD?
        # arg_sorted_desc = flip(argsort( s.lambda_list[ s.lambda_index_centre ]), 0)
        # s.virtual_charges_normalized[ symmetry ][internal ] = s.virtual_charges_normalized[ symmetry ][internal ][arg_sorted_desc ]
        # s.gamma_list[ left ] = s.gamma_list[ left ][:, arg_sorted_desc ]
        # s.gamma_list[right ] = s.gamma_list[right ][arg_sorted_desc, : ]

        # Truncate and get the new MPS and other objects.
        s.lambda_list[ s.lambda_index_centre ] /= norm( s.lambda_list[s.lambda_index_centre])
        s.chi_centre = len(s.lambda_list[s.lambda_index_centre])

        # Reshape Gamma.
        s.gamma_list[s.gamma_index_left] =  transpose( reshape( s.gamma_list[ s.gamma_index_left ],  ( s.chi_left, s.d, number_Schmidt_eigenvalues_kept) ),  (1,0,2) ) # 102
        s.gamma_list[s.gamma_index_right] = transpose( reshape( s.gamma_list[ s.gamma_index_right ], (number_Schmidt_eigenvalues_kept, s.d, s.chi_right) ), (1,0,2) ) # 102

        # Update the index relative to the centre of the canonical form.
        s.get_current_mixed_canonical_form_centre_index( )

    def svd_and_update_gamma_lambda(s): # 01.23
        # Schmidt deomposition #
        two_stes_mps = reshape( s.dense_flattened_contracted_tensor, (s.chi_left*s.d, s.d*s.chi_right) )
        X, Y, Z = svd( two_stes_mps )

        Y = Y[ Y > s.SCHMIDT_TOLERANCE ]
        chi = min( s.BOND_DIMENSION, len(Y) )
        Y = Y[:chi]
        X = X[:, :chi]
        Z = Z[:chi, :]
        X = reshape( X, (s.chi_left, s.d, chi) )
        Z = reshape( Z, (chi, s.d, s.chi_right) )

        # Obtain the new values for B and s #
        s.lambda_list[ s.lambda_index_centre ] = Y
        s.gamma_list[ s.gamma_index_left ] = transpose( X, (1,0,2) )
        s.gamma_list[ s.gamma_index_right ] = transpose( Z, (1,0,2) )


    #==========================================================================
    # SVD.
    #==========================================================================
    def svd_two_sites_two_sites_abelian(s):
        """
        Do a Abelian-wise SVD on the two-sites tensor in matrix form.
        Define the dictionaries where to collect the svd outcome.
        """
        X_dict = { }; Y_dict = { }; Z_dict = { }; not_sorted_full_Y_array = empty(0);
        null_blocks_svd_charges_sets = [ ]

        for charges_set in s.svd_charges_sets:
            # Get the row/column indices with quantum number == charge.
            row_ind_charge = s.two_sites_fused_row_legs_indices_per_charges_set[ charges_set ]
            col_ind_charge = s.two_sites_fused_col_legs_indices_per_charges_set[ charges_set ]

            # Populate the dense block with the corresponding entries of the two sites tensor.
            rows = len( row_ind_charge )
            cols = len( col_ind_charge )
            matrix_diagonal_block = zeros( [ rows, cols ] )

            # matrix_diagonal_block[ charges_set ] = zeros( s.contracted_tensor_blocks_shape[ tensordot_step ][ charges_set ] )
            cython_extract_from_vector_to_matrix( s.dense_flattened_contracted_tensor, matrix_diagonal_block, s.indices_from_Krylov_vector_to_svd_sectors[ charges_set ] )
            if array_equal( matrix_diagonal_block, zeros( matrix_diagonal_block.shape ) ):
                null_blocks_svd_charges_sets.append( charges_set )
                #showwarning('SVD, block null for charges set ' + str(charges_set) + ' block eliminated. Full list of charges for SVD.', category = UserWarning, filename = '', lineno = 0)
            else:
                x, y, z = svd( matrix_diagonal_block )
                X_dict[ charges_set ] = x
                Y_dict[ charges_set ] = y
                Z_dict[ charges_set ] = z
                not_sorted_full_Y_array = npappend( not_sorted_full_Y_array, y )

        # Check that all the Schmidt eigenvalues are positive.
        if not (not_sorted_full_Y_array > 0).all( ):
            not_sorted_full_Y_array[ not_sorted_full_Y_array < 0 ] = 0
            s.printb('At least one Schmidt eigenvalue is negative.' )

        # Discard the smallest Schmidt eigenvalues.
        incresing_order_sorted_full_Y_array = sort( not_sorted_full_Y_array )
        number_relevant_Schmidt_states = npsum( not_sorted_full_Y_array > s.SCHMIDT_TOLERANCE )

        number_entries_to_scan = 5

        if mod(number_relevant_Schmidt_states, 2) == 1 and number_relevant_Schmidt_states < s.BOND_DIMENSION:

            # If the number of relevant Schmidt eigenvalues is odd and less than
            # CHI, we try to remove states that might break the symmetry.
            number_Schmidt_eigenvalues_kept = number_relevant_Schmidt_states
            sorted_Schmidt_eigenvalues_kept = incresing_order_sorted_full_Y_array[ - number_Schmidt_eigenvalues_kept : ]

            # First we find the relative delta between those smallest Schmidt eigenvalues.
            interval = min( number_entries_to_scan, number_Schmidt_eigenvalues_kept )
            x = sorted_Schmidt_eigenvalues_kept[ : interval ]
            delta = x[ 1: ] - x[:-1 ]
            relative_delta = divide( delta, x[:-1 ] )

            # Cut at the biggest delta, just to make sure.
            max_rel = max(relative_delta)

            # So we keep all the states but that ones that appears to belong to a
            # a different degenerate set.
            for i in range(len(relative_delta) ):
                if relative_delta[ i ] == max_rel:
                    number_elems_to_remove = i + 1

            # If the spectrum is degenerate, this largest relative delta should
            # be an even number. If it is not, there is no signature of a double
            # (or multiple of two) degeneracy and we leave the Schimdt spectrum
            # untouched.
            if mod(number_elems_to_remove, 2) == 0:
                number_elems_to_remove = 0

            sorted_Schmidt_eigenvalues_kept = sorted_Schmidt_eigenvalues_kept[ number_elems_to_remove: ]

            # s.printb('%d Schmidt eigenvalues discarded: ' %number_elems_to_remove, ' '.join('%.5E' % _ for _ in x ) )

        elif mod(number_relevant_Schmidt_states, 2) == 0 and number_relevant_Schmidt_states < s.BOND_DIMENSION:

            # Well in this case we just keep them all.
            number_Schmidt_eigenvalues_kept = number_relevant_Schmidt_states
            sorted_Schmidt_eigenvalues_kept = incresing_order_sorted_full_Y_array[ - number_Schmidt_eigenvalues_kept : ]

        elif number_relevant_Schmidt_states > s.BOND_DIMENSION:

            # Let us just cut it to CHI.
            number_Schmidt_eigenvalues_kept = s.BOND_DIMENSION
            sorted_Schmidt_eigenvalues_kept = incresing_order_sorted_full_Y_array[ - s.BOND_DIMENSION : ]

            """
            # In this case we cut again at the biggest relative delta, if it
            # leaves an even number of state in the Schmidt spectrum.
            number_Schmidt_eigenvalues_kept = s.BOND_DIMENSION + 1
            sorted_Schmidt_eigenvalues_kept = incresing_order_sorted_full_Y_array[ - number_Schmidt_eigenvalues_kept : ]

            # First we find the relative delta between f those smallest Schmidt eigenvalues.
            interval = min( number_entries_to_scan, number_Schmidt_eigenvalues_kept )
            sorted_Schmidt_eigenvalues_interval = sorted_Schmidt_eigenvalues_kept[ : interval ]
            delta = sorted_Schmidt_eigenvalues_interval[ 1: ] - sorted_Schmidt_eigenvalues_interval[:-1 ]
            relative_delta = divide( delta, sorted_Schmidt_eigenvalues_interval[:-1 ] )

            # To establish where to cut the spectrum, we use the following criteria.
            #
            # 1) To be degenerate, the Schmidt eigenvalues are supposed to be
            #   equal down to the 3-rd digit at least. If the relative delta
            #   between last kept and first discarded eigenvalues is above 1.E-3
            #   consider them as non-degenerate.
            if relative_delta[ 0 ] > 1.E-3:
                number_elems_to_remove = 1

            # 2) If there is a degeneracy, the separation between two degenerate
            #   sets is larger than the one within the degeneracy. So, cut at the
            #   largest delta within the first 5 lowest eigenvalues.
            else:
                max_rel = max(relative_delta)
                # So we keep all the states but that one that appears to belong to a
                # a different degenerate set.
                for i in range(len(relative_delta) ):
                    if relative_delta[ i ] == max_rel:
                        number_elems_to_remove = i + 1

            # If the spectrum is degenerate, this largest relative delta should
            # be an even number. If it is not, there is no signature of a double
            # (or multiple of two) degeneracy and we leave the Schimdt spectrum
            # untouched.
            if mod(number_elems_to_remove, 2) == 0:
                number_elems_to_remove = 1

            sorted_Schmidt_eigenvalues_kept = sorted_Schmidt_eigenvalues_kept[ number_elems_to_remove: ]
            """

        elif number_relevant_Schmidt_states == s.BOND_DIMENSION:
            # Well in this case we just keep them all.
            number_Schmidt_eigenvalues_kept = s.BOND_DIMENSION
            sorted_Schmidt_eigenvalues_kept = incresing_order_sorted_full_Y_array[ - s.BOND_DIMENSION : ]
#            s.printb('Weird enough, we just have %d relevant Schmidt eigenvalues. How uncanny!')

        smallest_Schmidt_eigenvalue_kept = sorted_Schmidt_eigenvalues_kept[ 0 ]
        number_Schmidt_eigenvalues_kept = len(sorted_Schmidt_eigenvalues_kept)

        # Remove the charges sets associated to discarded blocks from svd_charges_sets.
        for charges_set in null_blocks_svd_charges_sets:
            s.svd_charges_sets.remove( charges_set )

        return X_dict, Y_dict, Z_dict, number_Schmidt_eigenvalues_kept, smallest_Schmidt_eigenvalue_kept


    #==========================================================================
    # State update.
    #==========================================================================
    def update_two_sites_and_transfer_matrices(s):
        """
        The two-sites tensor enters as 01x23.
        """
        if s.ABELIAN_SYMMETRIES:
            # Update the Gamma tensors, the Lambda vectors and the virtual_charges vectors.
            s.svd_and_update_gamma_lambda_virtual_abelian( )
        else:
            # Update the Gamma tensors, the Lambda vectors.
            s.svd_and_update_gamma_lambda( )
        # Update the left/right transfer matrices.
        s.update_transfer_matrices( )


    #==========================================================================
    # Update transfer matrices.
    #==========================================================================
    def update_transfer_matrices(s):
        for s.mpo_operator_tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:
            s.update_left_mpo_transfer_matrix( )
            s.update_right_mpo_transfer_matrix( )


    #==========================================================================
    # Update left transfer matrices.
    #==========================================================================
    def add_element_to_left_transfer_matrices(s):
        for s.mpo_operator_tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:
            s.add_element_to_left_mpo_transfer_matrix( )


    def add_element_to_left_mpo_transfer_matrix(s):
        s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ].append( zeros( [ s.BOND_DIMENSION, s.BOND_DIMENSION, s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ] ] ) )


    def update_left_mpo_transfer_matrix(s):
        if s.ABELIAN_SYMMETRIES:
            # Define the elements needed for the block-diagonal (Abelian) matvec.
            s.tensor_contractions_tag = 'ltm_mpo_update'
            s.get_objects_for_abelian_tensordot( )
            # Get the mapped indices, from sorted vector to matrix sectors.
            s.load_or_map_indices_for_dense_vector_representation( )
            # Get the indices for the last mapping, from dense vector onto sparse tensor.
            s.get_contracted_tensor_sparse_dense_map( )
            # Gamma is expected to have lg form.
            s.get_dense_flattened_contracted_tensor( )  # --> s.dense_flattened_contracted_tensor, (0,1,2) bottom-up
            s.abelian_tensor_contraction( )             # --> s.dense_flattened_contracted_tensor, (2,1,0) bottom-up
            s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag  ][ s.ltm_index_internal ] = s.map_dense_to_sparse_contracted_tensor( ) # (1,2,0) bottom-up
        else:
            lgamma = s.gamma_list[ s.gamma_index_left ]
            ltm_ext = s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ s.ltm_index_external ]
            lmpo = s.matrix_product_operator[ s.mpo_operator_tag ][ s.mpo_index_left ]
            #
            ltm_int = tdot( ltm_ext, lgamma, axes=(0,1) )
            ltm_int = tdot( ltm_int, lmpo, axes=([1,2],[0,3]) )
            ltm_int = tdot( ltm_int, conj(lgamma), axes=([0,3],[1,0]) )
            s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ s.ltm_index_internal ] = transpose( ltm_int, (0,2,1) )


    def update_left_opt_transfer_matrix(s):
        # Define the elements needed for the block-diagonal (Abelian) matvec.
        s.tensor_contractions_tag = 'ltm_opt_update'
        s.get_objects_for_abelian_tensordot( )
        # Get the mapped indices, from sorted vector to matrix sectors.
        s.load_or_map_indices_for_dense_vector_representation( )
        # Get the indices for the last mapping, from dense vector onto sparse tensor.
        s.get_contracted_tensor_sparse_dense_map( )
        s.update_left_opt_transfer_matrix_abelian( )


    def update_left_opt_transfer_matrix_abelian(s):
        """
        Gamma is expected to have lg form.
        """
        s.get_dense_flattened_contracted_tensor( )
        s.abelian_tensor_contraction( )
        s.left_opt_transfer_matrix_list[ s.mpo_operator_tag  ][ s.ltm_index_internal ] = s.map_dense_to_sparse_contracted_tensor( )



    #==========================================================================
    # Update right transfer matrix.
    #==========================================================================
    def add_element_to_right_transfer_matrices(s):
        for s.mpo_operator_tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:
            s.add_element_to_right_mpo_transfer_matrix( )


    def add_element_to_right_mpo_transfer_matrix(s):
        s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ].insert(0, zeros([ s.BOND_DIMENSION, s.BOND_DIMENSION, s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ] ]) )


    def update_right_mpo_transfer_matrix(s):
        if s.ABELIAN_SYMMETRIES:
            s.tensor_contractions_tag = 'rtm_mpo_update'
            s.get_objects_for_abelian_tensordot( )
            # Appendix
            s.load_or_map_indices_for_dense_vector_representation( )
            s.get_contracted_tensor_sparse_dense_map( )
            # Gamma is expected to have gl form.
            s.get_dense_flattened_contracted_tensor( )
            s.abelian_tensor_contraction( )
            s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ s.rtm_index_internal ] = s.map_dense_to_sparse_contracted_tensor( )
        else:
            rgamma = s.gamma_list[ s.gamma_index_right ]
            rtm_ext = s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ s.rtm_index_external ]
            rmpo = s.matrix_product_operator[ s.mpo_operator_tag ][ s.mpo_index_right ]
            #
            rtm_int = tdot( rtm_ext, rgamma, axes=(0,2) )
            rtm_int = tdot( rtm_int, rmpo, axes=([1,2],[1,3]) )
            rtm_int = tdot( rtm_int, conj(rgamma), axes=([0,3],[2,0]) )
            s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ s.rtm_index_internal ] = transpose( rtm_int, (0,2,1) )


    def update_right_opt_transfer_matrix(s):
        s.tensor_contractions_tag = 'rtm_opt_update'
        s.get_objects_for_abelian_tensordot( )
        # Appendix
        s.load_or_map_indices_for_dense_vector_representation( )
        s.get_contracted_tensor_sparse_dense_map( )
        s.update_right_opt_transfer_matrix_abelian( )


    def update_right_opt_transfer_matrix_abelian(s):
        """
        Gamma is expected to have lg form.
        """
        s.get_dense_flattened_contracted_tensor( )
        s.abelian_tensor_contraction( )
        s.right_opt_transfer_matrix_list[ s.mpo_operator_tag  ][ s.rtm_index_internal ] = s.map_dense_to_sparse_contracted_tensor( )


    #==========================================================================
    # Open and close tensors.
    #==========================================================================
    def open_this_gamma(s, index, gamma):
        """

        """
        if not 0 <= index < s.current_chain_length:
            return
        label = 'gamma'
        if type( gamma ) == csr_matrix:
            sparse_matrix = gamma.toarray( )
            shape = s.get_tensor_shape( label, sparse_matrix )
            gamma = reshape( sparse_matrix, shape )
        else:
            s.open_warning( label )

        return gamma


    def open_one_gamma(s, index):
        """

        """
        if not 0 <= index < s.current_chain_length:
            return
        label = 'gamma'
        if type( s.gamma_list[ index ] ) == csr_matrix:
            sparse_matrix = s.gamma_list[ index ].toarray( )
            shape =         s.get_tensor_shape( label, sparse_matrix )
            s.gamma_list[ index ] = reshape( sparse_matrix, shape )
        else:
            s.open_warning( label )


    def close_one_gamma(s, index):
        """

        """
        if not 0 <= index < s.current_chain_length:
            return
        label = 'gamma'
        if type( s.gamma_list[ index ] ) == ndarray:
            shape =         s.get_csr_shape( s.gamma_list[ index ] )
            sparse_matrix = reshape( s.gamma_list[ index ], shape )
            s.gamma_list[ index ] = csr_matrix( sparse_matrix )
        else:
            s.close_warning( label )


    def open_one_ltm(s, index):
        """

        """
        if not 0 <= index < s.get_current_ltm_length( ):
            return
        label = 'ltm'
        if type( s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ] ) == csr_matrix:
            sparse_matrix = s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ].toarray( )
            shape =         s.get_tensor_shape( label, sparse_matrix )
            s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ] = reshape( sparse_matrix, shape )
        else:
            s.open_warning( label )


    def close_one_ltm(s, index):
        """

        """
        if not 0 <= index < s.get_current_ltm_length( ):
            return
        label = 'ltm'
        if type( s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ] ) == ndarray:
            shape =         s.get_csr_shape( s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ] )
            sparse_matrix = reshape( s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ], shape )
            s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ] = csr_matrix( sparse_matrix )
        else:
            s.close_warning( label )


    def open_one_rtm(s, index):
        """

        """
        if not - s.get_current_rtm_length( ) <= index < 0:
            return
        label = 'rtm'
        if type( s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ] ) == csr_matrix:
            sparse_matrix = s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ].toarray( )
            shape =         s.get_tensor_shape( label, sparse_matrix )
            s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ] = reshape( sparse_matrix, shape )
        else:
            s.open_warning( label )


    def close_one_rtm(s, index):
        """

        """
        if not - s.get_current_rtm_length( ) <= index < 0:
            return
        label = 'rtm'
        if type( s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ] ) == ndarray:
            shape =         s.get_csr_shape( s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ] )
            sparse_matrix = reshape( s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ], shape )
            s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ index ] = csr_matrix( sparse_matrix )
        else:
            s.close_warning( label )


    def get_current_ltm_length(s):
        """

        """
        length = len( s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ] )
        return length


    def get_current_rtm_length(s):
        """

        """
        length = len( s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ] )
        return length


    def close_all_ltms(s):
        """

        """
        for index in range( len( s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ] ) ):
            s.close_one_ltm( index )


    def close_all_rtms(s):
        """

        """
        for index in range( len( s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ] ) ):
            s.close_one_rtm( - index - 1 )


    def open_warning(s, label):
        """

        """
        return
        s.printb('\nTrying to open a tensor not in dense csr_matrix form.' )
        s.printb( s.main_streamline_current_step_tag )
        s.printb( s.internal_streamline_current_step_tag )
        s.printb( s.tensor_contractions_tag )
        s.printb( label )
        s.printb( )


    def close_warning(s, label):
        """

        """
        return
        s.printb('\nTrying to close a tensor not in ndarray form.' )
        s.printb( s.main_streamline_current_step_tag )
        s.printb( s.internal_streamline_current_step_tag )
        s.printb( s.tensor_contractions_tag )
        s.printb( label )
        s.printb( )


# =============================================================================
# Abelian hmpo. Define the elements related to the block diagonal hamiltonian mpo.
# =============================================================================
    def get_matrix_product_operator_blocks_and_indices_per_charges_set(s):
        """
        Define the following self objects:

          - s.hmpo_fused_row_legs_indices_per_charges_set[ charges_set ]
          - s.hmpo_fused_col_legs_indices_per_charges_set[ charges_set ]

          - s.hmpo_blocks_for_matvec[ charges_set ]
        """
        s.update_indices( pivot_index=1 )

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        s.sought_tensor = 'mpo'
        s.tensor_contractions_tag = 'matvec'

        if not s.POST_RUN_INSPECTION:
            s.mpo_operator_tag = 'hamiltonian'

            s.hmpo_blocks_for_matvec = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]
            s.hmpo_fused_row_legs_indices_per_charges_set = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]
            s.hmpo_fused_col_legs_indices_per_charges_set = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]

            for s.lattice_unit_cell_index in range( s.LATTICE_UNIT_CELL_SIZE ):

                t = time()
                s.hmpo_blocks_for_matvec[ s.lattice_unit_cell_index ],                      \
                s.hmpo_fused_row_legs_indices_per_charges_set[ s.lattice_unit_cell_index ], \
                s.hmpo_fused_col_legs_indices_per_charges_set[ s.lattice_unit_cell_index ]  \
                    = s.get_tensor_blocks_and_indices_per_charges_set( )
                dt = time() - t
                print('s.lattice_unit_cell_index, s.tensor_contractions_tag, s.sought_tensor, s.mpo_operator_tag, dt', s.lattice_unit_cell_index, s.tensor_contractions_tag, s.sought_tensor, s.mpo_operator_tag, dt )

        s.tensor_contractions_tag = 'ltm_mpo_update'

        s.mpo_blocks_for_ltm_update = { }
        s.mpo_fused_row_legs_indices_per_charges_set_for_ltm_update = { }
        s.mpo_fused_col_legs_indices_per_charges_set_for_ltm_update = { }

        for s.mpo_operator_tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:

            s.mpo_blocks_for_ltm_update[ s.mpo_operator_tag ] = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]
            s.mpo_fused_row_legs_indices_per_charges_set_for_ltm_update[ s.mpo_operator_tag ] = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]
            s.mpo_fused_col_legs_indices_per_charges_set_for_ltm_update[ s.mpo_operator_tag ] = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]

            for s.lattice_unit_cell_index in range( s.LATTICE_UNIT_CELL_SIZE ):

                t = time()
                s.mpo_blocks_for_ltm_update[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ],                                 \
                s.mpo_fused_row_legs_indices_per_charges_set_for_ltm_update[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ], \
                s.mpo_fused_col_legs_indices_per_charges_set_for_ltm_update[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ]  \
                    = s.get_tensor_blocks_and_indices_per_charges_set( )
                dt = time() - t
                print('s.lattice_unit_cell_index, s.tensor_contractions_tag, s.sought_tensor, s.mpo_operator_tag, dt', s.lattice_unit_cell_index, s.tensor_contractions_tag, s.sought_tensor, s.mpo_operator_tag, dt )

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        s.tensor_contractions_tag = 'rtm_mpo_update'

        s.mpo_blocks_for_rtm_update = { }
        s.mpo_fused_row_legs_indices_per_charges_set_for_rtm_update = { }
        s.mpo_fused_col_legs_indices_per_charges_set_for_rtm_update = { }

        for s.mpo_operator_tag in s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS:

            s.mpo_blocks_for_rtm_update[ s.mpo_operator_tag ] = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]
            s.mpo_fused_row_legs_indices_per_charges_set_for_rtm_update[ s.mpo_operator_tag ] = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]
            s.mpo_fused_col_legs_indices_per_charges_set_for_rtm_update[ s.mpo_operator_tag ] = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]

            for s.lattice_unit_cell_index in range( s.LATTICE_UNIT_CELL_SIZE ):

                t = time()
                s.mpo_blocks_for_rtm_update[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ],                                 \
                s.mpo_fused_row_legs_indices_per_charges_set_for_rtm_update[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ], \
                s.mpo_fused_col_legs_indices_per_charges_set_for_rtm_update[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ]  \
                    = s.get_tensor_blocks_and_indices_per_charges_set( )
                dt = time() - t
                print('s.lattice_unit_cell_index, s.tensor_contractions_tag, s.sought_tensor, s.mpo_operator_tag, dt', s.lattice_unit_cell_index, s.tensor_contractions_tag, s.sought_tensor, s.mpo_operator_tag, dt )

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        s.sought_tensor = 'opt'

        s.tensor_contractions_tag = 'ltm_opt_update'

        s.opt_blocks_for_ltm_update = { }
        s.opt_fused_row_legs_indices_per_charges_set_for_ltm_update = { }
        s.opt_fused_col_legs_indices_per_charges_set_for_ltm_update = { }

        for s.mpo_operator_tag in s.NAMES_NORMAL_MATRIX_OPERATORS_FOR_CORRELATIONS_AND_LOCAL_EXPECTATION_VALUES:

            s.opt_blocks_for_ltm_update[ s.mpo_operator_tag ] = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]
            s.opt_fused_row_legs_indices_per_charges_set_for_ltm_update[ s.mpo_operator_tag ] = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]
            s.opt_fused_col_legs_indices_per_charges_set_for_ltm_update[ s.mpo_operator_tag ] = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]

            for s.lattice_unit_cell_index in range( s.LATTICE_UNIT_CELL_SIZE ):

                t = time()
                s.opt_blocks_for_ltm_update[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ],                                 \
                s.opt_fused_row_legs_indices_per_charges_set_for_ltm_update[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ], \
                s.opt_fused_col_legs_indices_per_charges_set_for_ltm_update[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ]  \
                    = s.get_tensor_blocks_and_indices_per_charges_set( )
                dt = time() - t
                print('s.lattice_unit_cell_index, s.tensor_contractions_tag, s.sought_tensor, s.mpo_operator_tag, dt', s.lattice_unit_cell_index, s.tensor_contractions_tag, s.sought_tensor, s.mpo_operator_tag, dt )

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        s.tensor_contractions_tag = 'rtm_opt_update'

        s.opt_blocks_for_rtm_update = { }
        s.opt_fused_row_legs_indices_per_charges_set_for_rtm_update = { }
        s.opt_fused_col_legs_indices_per_charges_set_for_rtm_update = { }

        for s.mpo_operator_tag in s.NAMES_NORMAL_MATRIX_OPERATORS_FOR_CORRELATIONS_AND_LOCAL_EXPECTATION_VALUES:

            s.opt_blocks_for_rtm_update[ s.mpo_operator_tag ] = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]
            s.opt_fused_row_legs_indices_per_charges_set_for_rtm_update[ s.mpo_operator_tag ] = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]
            s.opt_fused_col_legs_indices_per_charges_set_for_rtm_update[ s.mpo_operator_tag ] = [ 0 for lattice_cell_index in range( s.LATTICE_UNIT_CELL_SIZE) ]

            for s.lattice_unit_cell_index in range( s.LATTICE_UNIT_CELL_SIZE ):

                t = time()
                s.opt_blocks_for_rtm_update[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ],                                 \
                s.opt_fused_row_legs_indices_per_charges_set_for_rtm_update[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ], \
                s.opt_fused_col_legs_indices_per_charges_set_for_rtm_update[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ]  \
                    = s.get_tensor_blocks_and_indices_per_charges_set( )
                dt = time() - t
                print('s.lattice_unit_cell_index, s.tensor_contractions_tag, s.sought_tensor, s.mpo_operator_tag, dt', s.lattice_unit_cell_index, s.tensor_contractions_tag, s.sought_tensor, s.mpo_operator_tag, dt )

        for self_object in ['tensor_contractions_tag', 'sought_tensor', 'mpo_operator_tag']:
            if hasattr(s, self_object):
                delattr(s, self_object)


    def get_tensor_fused_legs_charges(s, charges_lists_dict):
        """
        Return the following local objects:
          - tensor_fused_row_legs_charges
          - tensor_fused_row_legs_charges
        """

        if s.tensor_contractions_tag == 'matvec':

            if s.sought_tensor == 'mpo':
                # =================== two-sites matmul steps 1, 2 =================== #
                #                                                                     #
                #                                  1                                  #
                #                                  |                                  #
                #                                  ^                                  #
                #                                  ⟂                                  #
                #                          0 -->--| |-->-- 2   MPO                    #
                #                                  T                                  #
                #                                  ^                                  #
                #                                  |                                  #
                #                                  3                                  #
                #                                                                     #
                # =================== two-sites matmul steps 1, 2 =================== #
                physical_charges =  charges_lists_dict['physical_charges']
                mpo_charges_left =  charges_lists_dict['mpo_charges_left']
                mpo_charges_right = charges_lists_dict['mpo_charges_right']

                # hmpo row legs
                tensor_fused_row_legs_charges = s.generate_list_charges([ - mpo_charges_left ,
                                                                          + physical_charges ], dtype=npint64 )
                # hmpo col legs
                tensor_fused_col_legs_charges = s.generate_list_charges([ + mpo_charges_right ,
                                                                          - physical_charges  ], dtype=npint64 )

            if s.sought_tensor == 'transfer_matrix':

                if s.transfer_matrix_side == 'left':
                    # ===================== two-sites matmul step 0 ===================== #
                    #    _                   _____________________                        #
                    #   | |-->-- 0   3 -->--|_____________________|-->-- 2                #
                    #   | |                       ^         ^                             #
                    #   | |                       |  theta  |                             #
                    #   | |                       0         1                             #
                    #   | |  ltm                                                          #
                    #   | |                                                               #
                    #   | |-->-- 2                                                        #
                    #   | |                                                               #
                    #   | |                                                               #
                    #   | |                                                               #
                    #   | |                                                               #
                    #   | |                                                               #
                    #   |_|--<-- 1                                                        #
                    #                                                                     #
                    # ===================== two-sites matmul step 0 ===================== #
                    virtual_charges_left = charges_lists_dict['virtual_charges_left']
                    mpo_charges_left =     charges_lists_dict['mpo_charges']

                    # htm row legs
                    tensor_fused_row_legs_charges = s.generate_list_charges([ + virtual_charges_left ], dtype=npint64 )

                    # htm col legs
                    tensor_fused_col_legs_charges = s.generate_list_charges([ - virtual_charges_left ,
                                                                              + mpo_charges_left     ], dtype=npint64 )

                elif s.transfer_matrix_side == 'right':
                    # ===================== two-sites matmul step 3 ===================== #
                    #    _________________________________________                   _    #
                    #   |                                         |-->-- 3   0 -->--| |   #
                    #   |                                         |                 | |   #
                    #   |                                         |                 | |   #
                    #   |                                         |                 | |   #
                    #   |                                         |                 | |   #
                    #   |                                         |                 | |   #
                    #   |                                         |           hrtm  | |   #
                    #   |                  theta                  |                 | |   #
                    #   |                                         |                 | |   #
                    #   |                                         |                 | |   #
                    #   |                                         |                 | |   #
                    #   |                                         |-->-- 4   1 -->--| |   #
                    #   |                                         |                 | |   #
                    #   |                                         |                 | |   #
                    #   |                                         |                 | |   #
                    #   |  _______________________________________|                 | |   #
                    #   | |           |               |                             | |   #
                    #   | |           ^               ^                             | |   #
                    #   | |           |               |                             | |   #
                    #   | |           1               2                             | |   #
                    #   | |                                                         | |   #
                    #   | |                                                         | |   #
                    #   |_|--<-- 0                                           2 --<--|_|   #
                    #                                                                     #
                    # ===================== two-sites matmul step 3 ===================== #
                    virtual_charges_right = charges_lists_dict['virtual_charges_right']
                    mpo_charges_right =     charges_lists_dict['mpo_charges']

                    # hrtm row legs
                    tensor_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_right ,
                                                                              - mpo_charges_right    ], dtype = npint64 )
                    # hrtm col legs
                    tensor_fused_col_legs_charges = s.generate_list_charges([ + virtual_charges_right ], dtype = npint64 )

        elif s.tensor_contractions_tag in ['ltm_mpo_update', 'ltm_opt_update']:

            if s.sought_tensor == 'mpo':
                # ======================== ltm matmul step 1 ======================== #
                #                                                                     #
                #                                  1                                  #
                #                                  |                                  #
                #                                  ^                                  #
                #                                  ⟂                                  #
                #                          0 -->--| |-->-- 2   MPO                    #
                #                                  T                                  #
                #                                  ^                                  #
                #                                  |                                  #
                #                                  3                                  #
                #                                                                     #
                # ======================== ltm matmul step 1 ======================== #
                physical_charges =  charges_lists_dict['physical_charges']
                mpo_charges_left =  charges_lists_dict['mpo_charges_left']
                mpo_charges_right = charges_lists_dict['mpo_charges_right']

                # hmpo row legs
                tensor_fused_row_legs_charges = s.generate_list_charges([ - mpo_charges_left ,
                                                                          + physical_charges ], dtype=npint64 )
                # hmpo col legs
                tensor_fused_col_legs_charges = s.generate_list_charges([ + mpo_charges_right ,
                                                                          - physical_charges  ], dtype=npint64 )

            elif s.sought_tensor == 'opt':
                # ======================== ltm matmul step 1 ======================== #
                #                                                                     #
                #                                  0                                  #
                #                                  |                                  #
                #                                  ^                                  #
                #                                  ⟂                                  #
                #                                 | |   opt                           #
                #                                  T                                  #
                #                                  ^                                  #
                #                                  |                                  #
                #                                  1                                  #
                #                                                                     #
                # ======================== ltm matmul step 1 ======================== #
                physical_charges =  charges_lists_dict['physical_charges']

                tensor_fused_row_legs_charges = s.generate_list_charges([ + physical_charges ], dtype=npint64 )
                # opt col legs
                tensor_fused_col_legs_charges = s.generate_list_charges([ - physical_charges ], dtype=npint64 )

            elif s.sought_tensor == 'gamma':

                if s.gamma_bra_or_ket == 'ket':
                    # ==================== ltm matmul step 0 =================== #
                    #                        _____________                       #
                    #                0 -->--|_____________|-->-- 1               #
                    #                              ^                             #
                    #                Gamma_left    |                             #
                    #                              2                             #
                    #                                                            #
                    # ==================== ltm matmul step 0 =================== #
                    virtual_charges_left =  charges_lists_dict['virtual_charges_left']
                    virtual_charges_right = charges_lists_dict['virtual_charges_right']
                    physical_charges =      charges_lists_dict['physical_charges']

                    # gamma row legs
                    tensor_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_left ], dtype=npint64 )

                    # gamma col legs
                    tensor_fused_col_legs_charges = s.generate_list_charges([ + virtual_charges_right ,
                                                                              - physical_charges      ], dtype=npint64 )

                elif s.gamma_bra_or_ket == 'bra':
                    # ==================== ltm matmul step 1 =================== #
                    #                                                            #
                    #                              ^                             #
                    #                Gamma_left    1                             #
                    #                        ______|______                       #
                    #                0 --<--|_____________|--<-- 2               #
                    #                                                            #
                    # ==================== ltm matmul step 1 =================== #
                    virtual_charges_left =  charges_lists_dict['virtual_charges_left']
                    virtual_charges_right = charges_lists_dict['virtual_charges_right']
                    physical_charges =      charges_lists_dict['physical_charges']

                    # gamma row legs
                    tensor_fused_row_legs_charges = s.generate_list_charges([ + virtual_charges_left ,
                                                                              + physical_charges     ], dtype=npint64 )

                    # gamma col legs
                    tensor_fused_col_legs_charges = s.generate_list_charges([ - virtual_charges_right ], dtype=npint64 )

        elif s.tensor_contractions_tag in ['rtm_mpo_update', 'rtm_opt_update']:

            if s.sought_tensor == 'mpo':
                # ======================== rtm matmul step 2 ======================== #
                #                                                                     #
                #                                  1                                  #
                #                                  |                                  #
                #                                  ^                                  #
                #                                  ⟂                                  #
                #                          2 -->--| |-->-- 0   MPO                    #
                #                                  T                                  #
                #                                  ^                                  #
                #                                  |                                  #
                #                                  3                                  #
                #                                                                     #
                # ======================== rtm matmul step 2 ======================== #
                physical_charges =  charges_lists_dict['physical_charges']
                mpo_charges_left =  charges_lists_dict['mpo_charges_left']
                mpo_charges_right = charges_lists_dict['mpo_charges_right']

                # hmpo row legs
                tensor_fused_row_legs_charges = s.generate_list_charges([ + mpo_charges_right ,
                                                                          + physical_charges ], dtype=npint64 )
                # hmpo col legs
                tensor_fused_col_legs_charges = s.generate_list_charges([ - mpo_charges_left ,
                                                                          - physical_charges  ], dtype=npint64 )

            elif s.sought_tensor == 'opt':
                # ======================== rtm matmul step 2 ======================== #
                #                                                                     #
                #                                  0                                  #
                #                                  |                                  #
                #                                  ^                                  #
                #                                  ⟂                                  #
                #                                 | |   opt                           #
                #                                  T                                  #
                #                                  ^                                  #
                #                                  |                                  #
                #                                  1                                  #
                #                                                                     #
                # ======================== rtm matmul step 2 ======================== #
                physical_charges = charges_lists_dict['physical_charges']

                # opt row legs
                tensor_fused_row_legs_charges = s.generate_list_charges([ + physical_charges ], dtype=npint64 )
                # opt col legs
                tensor_fused_col_legs_charges = s.generate_list_charges([ - physical_charges ], dtype=npint64 )

            elif s.sought_tensor == 'gamma':

                if s.gamma_bra_or_ket == 'ket':
                    # ==================== rtm matmul step 0 =================== #
                    #                        _____________                       #
                    #                1 -->--|_____________|-->-- 0               #
                    #                              ^                             #
                    #                              |   Gamma_right               #
                    #                              2                             #
                    #                                                            #
                    # ==================== rtm matmul step 0 =================== #
                    virtual_charges_left =  charges_lists_dict['virtual_charges_left']
                    virtual_charges_right = charges_lists_dict['virtual_charges_right']
                    physical_charges =      charges_lists_dict['physical_charges']

                    # hrtm row legs
                    tensor_fused_row_legs_charges = s.generate_list_charges([ + virtual_charges_right ], dtype = npint64 )

                    # hrtm col legs
                    tensor_fused_col_legs_charges = s.generate_list_charges([ - virtual_charges_left ,
                                                                              - physical_charges     ], dtype = npint64 )

                if s.gamma_bra_or_ket == 'bra':
                    # ==================== rtm matmul step 1 =================== #
                    #                                                            #
                    #                              ^                             #
                    #                              1   Gamma_right               #
                    #                        ______|______                       #
                    #                2 --<--|_____________|--<-- 0               #
                    #                                                            #
                    # ==================== rtm matmul step 1 =================== #
                    virtual_charges_left =  charges_lists_dict['virtual_charges_left']
                    virtual_charges_right = charges_lists_dict['virtual_charges_right']
                    physical_charges =      charges_lists_dict['physical_charges']

                    # hrtm row legs
                    tensor_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_right ,
                                                                              + physical_charges      ], dtype = npint64 )

                    # hrtm col legs
                    tensor_fused_col_legs_charges = s.generate_list_charges([ + virtual_charges_left ], dtype = npint64 )

        tensor_fused_row_legs_charges *= -1 # see WIKI CHARGE CONVENTION

        return tensor_fused_row_legs_charges, tensor_fused_col_legs_charges


    # =====================================================================
    # Some functions used for both the htm and the hmpo objects.

    def combine_charges_sets_with_unique_charges(s, charges_sets, unique_charges):
        dummie = [ ]

        for charges_set in charges_sets:
            for charge in unique_charges:
                x = copy( charges_set ) # wwwww is it necessary to copy?
                x = x + (charge,) # is the uint preserved?
                dummie.append(x)

        return dummie

    def get_blocks_per_charges_set(s, matrix, tensor_charges_sets, tensor_fused_row_legs_indices_per_charges_set, tensor_fused_col_legs_indices_per_charges_set):

        blocks = { }

        for charges_set in tensor_charges_sets:
            # Extract the elements from the matrices and populate the blocks.
            row_ind_charges_set = tensor_fused_row_legs_indices_per_charges_set[ charges_set ]
            col_ind_charges_set = tensor_fused_col_legs_indices_per_charges_set[ charges_set ]
            row_size = len( row_ind_charges_set )
            col_size = len( col_ind_charges_set )

            blocks[ charges_set ] = zeros( [ row_size, col_size ] )
            cython_extract( matrix, blocks[ charges_set ], row_ind_charges_set, col_ind_charges_set )

        return blocks



# =============================================================================
# Abelian matvec. Define the elements for the block-diagonal matvec.
# =============================================================================
    def get_objects_for_abelian_tensordot(s):
        """
        This function returns blocks and indices for the three sets of tensor
        contractions needed in DMRG:
            - the matrix-vector multiplication within the Lanczos method
            - the update of the left transfer matrix
            - the update of the right transfer matrix

        We define contracted_tensor the one which is basically updated through
        the tensor contraction, namely:
            - theta
            - ltm
            - rtm

        We define absorbed_tensors those that participate in the tensor contraction,
        namely:
            - ltm, hmpo_left, hmpo_right, rtm
            - gamma_left, mpo_left
            - gamma_right, mpo_right

        Among those, the ones related to matrix product operators are obtained
        once and for all at the beginning of the algorithm since they do not
        change afterwards: they are the static_absorbed_tensors.

        We define absorbed_dynamic_tensors the remaining ones, namely:
            - ltm, rtm
            - gamma_left
            - gamma_right
        """

        # First obtain what we call the absorbed_dynamic_tensors.
        #
        # Define the following self objects, at each iteration:
        #   - s.hltm_blocks[ charges_set ] / s.hrtm_blocks[ charges_set ]
        #   - s.gamma_blocks_bra[ charges_set ] / s.gamma_blocks_ket[ charges_set ]
        #   - s.contracting_tensors_fused_cen_legs_indices_per_charges_set[ tensordot_step ][ charges_set ]
        #   - s.contracting_tensors_fused_col_legs_indices_per_charges_set[ tensordot_step ][ charges_set ]
        s.get_absorbed_dynamic_tensors( )


        # Then obtain the elements related to the contracted tensor.
        # Get the indices to load the theta matrix blocks from the matrix theta.
        #
        # Define the following self objects, at each iteration:
        #   - s.contracting_tensors_fused_row_legs_indices_per_charges_set[ tensordot_step ][ charges_set ]
        s.get_contracted_tensor_indices_per_charges_set( )


        # Define the following self objects:
        #   - central_charges_sets[ matmul_step ] = charges_sets
        #   - contracted_tensor_as_matrix_shape_post_contraction
        #   - contracted_tensor_as_matrix_shape_pre_contraction
        #   - contracted_tensor_shape_after_transpose
        s.get_contracted_tensor_shape( )
        s.get_shared_charges_sets_as_self_objects( )
        s.get_tensor_blocks_shape( )

        """
        # Get the indices to map the contracted vector between sparse and dense form.
        #   - s.indices_from_dense_flattened_tensor_to_blocks_dict = [ s.tensordot_step ][ charges_set ]
        s.load_or_map_indices_for_dense_vector_representation( )

        # Get the indices to map the contracted vector between sparse and dense form.
        #
        #   - s.indices_map_sparse_and_dense_contracted_tensor = nparray
        #
        # Arguments:
        #
        #   - s.contracting_tensors_fused_row_legs_indices_per_charges_set
        #   - s.contracting_tensors_fused_col_legs_indices_per_charges_set
        #   - s.central_charges_sets
        #   - s.contracted_tensor_as_matrix_shape_post_contraction
        if s.tensor_contractions_tag != 'matvec':
            s.get_contracted_tensor_sparse_dense_map( )

        del s.contracted_tensor_shape_after_transpose
        del s.contracted_tensor_as_matrix_shape_pre_contraction
        del s.contracted_tensor_as_matrix_shape_post_contraction

        del s.contracting_tensors_fused_row_legs_indices_per_charges_set
        del s.contracting_tensors_fused_cen_legs_indices_per_charges_set
        del s.contracting_tensors_fused_col_legs_indices_per_charges_set
        """


    def load_or_map_indices_for_dense_vector_representation(s):

        def compare(a, b):
            same = array_equal(a, b)
            return same

        current_charges_configuration = { }

        if s.tensor_contractions_tag in ['matvec', 'two_sites_svd']:

            for key in s.physical_charges_normalized.keys( ):
                current_charges_configuration['physical_charges_left_' + key ] =  s.physical_charges_normalized[ key ][ s.physical_charges_index_left ]
                current_charges_configuration['physical_charges_right_' + key ] = s.physical_charges_normalized[ key ][ s.physical_charges_index_right ]

            for key in s.virtual_charges_normalized.keys( ):
                current_charges_configuration['virtual_charges_left_' + key ] =  s.virtual_charges_normalized[ key ][ s.lambda_index_left ]
                current_charges_configuration['virtual_charges_right_' + key ] = s.virtual_charges_normalized[ key ][ s.lambda_index_right ]

        elif s.tensor_contractions_tag in ['ltm_mpo_update', 'ltm_opt_update']:

            for key in s.physical_charges_normalized.keys( ):
                current_charges_configuration['physical_charges_' + key ] = s.physical_charges_normalized[ key ][ s.physical_charges_index_left ]

            for key in s.virtual_charges_normalized.keys( ):
                current_charges_configuration['virtual_charges_left_' + key ] =  s.virtual_charges_normalized[ key ][ s.lambda_index_left ]
                current_charges_configuration['virtual_charges_right_' + key ] = s.virtual_charges_normalized[ key ][ s.lambda_index_centre ]

        elif s.tensor_contractions_tag in ['rtm_mpo_update', 'rtm_opt_update']:

            for key in s.physical_charges_normalized.keys( ):
                current_charges_configuration['physical_charges_' + key ] = s.physical_charges_normalized[ key ][ s.physical_charges_index_right ]

            for key in s.virtual_charges_normalized.keys( ):
                current_charges_configuration['virtual_charges_left_' + key ] =  s.virtual_charges_normalized[ key ][ s.lambda_index_centre ]
                current_charges_configuration['virtual_charges_right_' + key ] = s.virtual_charges_normalized[ key ][ s.lambda_index_right ]

        configuration_already_mapped = False
        configuration_already_stored = False

        charges_configurations_tag = s.tensor_contractions_tag + '_' + s.mpo_operator_tag

        for mapped_charges_configuration in s.mapped_charges_configurations[ charges_configurations_tag ]:

            configuration_already_mapped = all(_ is True for _ in map(compare, mapped_charges_configuration.values( ), current_charges_configuration.values( ) ) )

            if configuration_already_mapped:

                mapped_unique_random_key = mapped_charges_configuration['unique_random_key']
                s.how_many_times_each_charges_configuration_was_mapped[ mapped_unique_random_key ] += 1

                for stored_charges_configuration in s.stored_charges_configurations[ charges_configurations_tag ]:

                    configuration_already_stored = all(_ is True for _ in map(compare, stored_charges_configuration.values( ), current_charges_configuration.values( ) ) )
                    if configuration_already_stored:
                        break

                break

        if configuration_already_stored:

            s.indices_from_dense_flattened_tensor_to_blocks_dict = s.stored_dense_indices[ mapped_unique_random_key ]

            if s.tensor_contractions_tag == 'matvec':

                indices_list = s.stored_two_sites_indices[ mapped_unique_random_key ]
                s.indices_from_svd_vector_to_Krylov_sectors = indices_list[ 0 ]
                s.indices_from_Krylov_vector_to_svd_sectors = indices_list[ 1 ]

        else:

            s.indices_from_dense_flattened_tensor_to_blocks_dict = [ { } for _ in range( s.number_tensor_contractions[ s.tensor_contractions_tag ] ) ]

            for tensordot_step in range( s.number_tensor_contractions[ s.tensor_contractions_tag ] ):

                s.map_indices_for_dense_vector_representation( tensordot_step )

            if s.tensor_contractions_tag == 'matvec':

                #############################################################################

                # Obtain the indices to load the sectors of the two-sites Krylov vector,
                # with shape (012,3), from the sorted two-sites vector, with shape (01,23).

                ########### Part 1 - the source

                # Set the source indices.
                row_sectors_indices = s.two_sites_fused_row_legs_indices_per_charges_set
                col_sectors_indices = s.two_sites_fused_col_legs_indices_per_charges_set

                # Set the source matrix shape.
                source_matrix_shape = ( s.chi_left*s.d, s.d*s.chi_right )

                # Set the source charges sets: the order of the charges determines the sectors sorting.
                source_charges_sets = s.svd_charges_sets

                # Call the function that returns the maps unsorted-matrix onto sorted-vector.
                map_AB_col, map_C_row, map_D_col = \
                    s.get_maps_from_unsorted_matrix_to_sorted_vector( source_matrix_shape,
                                                                      source_charges_sets,
                                                                      row_sectors_indices,
                                                                      col_sectors_indices )

                ########### Part 2 - the querying sectors

                s.indices_from_svd_vector_to_Krylov_sectors = { }

                number_tensor_legs = 4
                tensor_legs_transposition_order = (0,1,2,3)

                # Set the querying matrix shape and other related variables.
                querying_matrix_shape = s.contracted_tensor_as_matrix_shape_post_contraction[ - 1 ]
                querying_tensor_shape = s.contracted_tensor_shape_after_matmul[ - 1 ]

                # Set the querying charges sets: in this case, they correspond to the source charges sets. The order does not matter.
                querying_charges_sets = s.row_col_charges_sets[ - 1 ]

                # Map the querying matrix sector indices.
                for charges_set in querying_charges_sets:

                    # Call the row and colum querying sector indices.
                    querying_sector_indices_row = s.contracting_tensors_fused_row_legs_indices_per_charges_set[ - 1 ][ charges_set ]
                    querying_sector_indices_col = s.contracting_tensors_fused_col_legs_indices_per_charges_set[ - 1 ][ charges_set ]

                    s.indices_from_svd_vector_to_Krylov_sectors[ charges_set ] = \
                        s.get_qsi_vectorial_form( querying_sector_indices_row,
                                                  querying_sector_indices_col,
                                                  querying_matrix_shape,
                                                  querying_tensor_shape,
                                                  number_tensor_legs,
                                                  tensor_legs_transposition_order,
                                                  source_matrix_shape,
                                                  charges_set,
                                                  map_AB_col,
                                                  map_C_row,
                                                  map_D_col )

                #############################################################################

                # Obtain the indices to load the sectors of the two-sites vector, with shape (01,23),
                # from the sorted two-sites Krylov vector, with shape (012,3).

                ########### Part 1 - the source

                # Set the source indices.
                row_sectors_indices = s.contracting_tensors_fused_row_legs_indices_per_charges_set[ - 1 ]
                col_sectors_indices = s.contracting_tensors_fused_col_legs_indices_per_charges_set[ - 1 ]

                # Set the source matrix shape and charges_sets.
                source_matrix_shape = s.contracted_tensor_as_matrix_shape_post_contraction[ - 1 ]

                # Set the source charges sets: the order of the charges determines the sectors sorting.
                source_charges_sets = s.row_col_charges_sets[ - 1 ]

                # Call the function that returns the maps unsorted-source-matrix onto sorted-source-vector.
                map_AB_col, map_C_row, map_D_col = \
                    s.get_maps_from_unsorted_matrix_to_sorted_vector( source_matrix_shape,
                                                                      source_charges_sets,
                                                                      row_sectors_indices,
                                                                      col_sectors_indices )

                ########### Part 2 - the querying sectors

                s.indices_from_Krylov_vector_to_svd_sectors = { }

                number_tensor_legs =    4
                tensor_legs_transposition_order = (0,1,2,3)

                # Set the querying matrix shape and other related variables.
                querying_matrix_shape = ( s.chi_left*s.d, s.d*s.chi_right  )
                querying_tensor_shape = ( s.chi_left, s.d, s.d, s.chi_right  )

                # Set the querying charges sets: in this case, they correspond to the source charges sets. The order does not matter.
                querying_charges_sets = s.svd_charges_sets

                # Map the querying matrix sector indices.
                for charges_set in querying_charges_sets:

                    # Call the row and colum querying sector indices.
                    querying_sector_indices_row = s.two_sites_fused_row_legs_indices_per_charges_set[ charges_set ]
                    querying_sector_indices_col = s.two_sites_fused_col_legs_indices_per_charges_set[ charges_set ]

                    s.indices_from_Krylov_vector_to_svd_sectors[ charges_set ] = \
                        s.get_qsi_vectorial_form( querying_sector_indices_row,
                                                  querying_sector_indices_col,
                                                  querying_matrix_shape,
                                                  querying_tensor_shape,
                                                  number_tensor_legs,
                                                  tensor_legs_transposition_order,
                                                  source_matrix_shape,
                                                  charges_set,
                                                  map_AB_col,
                                                  map_C_row,
                                                  map_D_col )

                #############################################################################

            if not configuration_already_mapped:

                new_unique_random_key = ''.join( choices( ascii_uppercase + digits, k=10 ) )
                s.how_many_times_each_charges_configuration_was_mapped[ new_unique_random_key ] = 1
                current_charges_configuration['unique_random_key'] = new_unique_random_key
                s.mapped_charges_configurations[ charges_configurations_tag ].append( current_charges_configuration )

            else:

                n = 10
                if s.how_many_times_each_charges_configuration_was_mapped[ mapped_unique_random_key ] >= n:
                    print('A charge configuration for %s has been mapped more than %d times, hence it is stored.' %(s.tensor_contractions_tag, n) )

                    s.stored_dense_indices[ mapped_unique_random_key ] = s.indices_from_dense_flattened_tensor_to_blocks_dict

                    if s.tensor_contractions_tag == 'matvec':

                        s.stored_two_sites_indices[ mapped_unique_random_key ] = \
                            [ s.indices_from_svd_vector_to_Krylov_sectors, \
                              s.indices_from_Krylov_vector_to_svd_sectors ]

                    current_charges_configuration['unique_random_key'] = mapped_unique_random_key
                    s.stored_charges_configurations[ charges_configurations_tag ].append( current_charges_configuration )


    def map_indices_for_dense_vector_representation(s, tensordot_step):
        # Map the theta indices per charges sets so that they draw from the dense array theta.
        # Get the indices to map the contracted vector between sparse and dense form.
        #
        #   - s.indices_from_dense_flattened_tensor_to_blocks_dict = [ tensordot_step ][ charges_set ]
        #
        # Arguments:
        #   - s.contracting_tensors_fused_row_legs_indices_per_charges_set
        #   - s.contracting_tensors_fused_col_legs_indices_per_charges_set
        #   - s.central_charges_sets
        #   - s.contracted_tensor_as_matrix_shape_post_contraction

        if s.tensor_contractions_tag in ['ltm_mpo_update', 'rtm_mpo_update', 'ltm_opt_update', 'rtm_opt_update'] and tensordot_step == 0:

            # Get the indices related to theta after/before a matmul.
            s.get_indices_tms_contraction_first_step( )         # s.indices_from_dense_flattened_tensor_to_blocks_dict

        else:

            #############################################################################

            # Obtain the indices to load the sectors of the . . . vector,
            # from the sorted . . . vector.

            ########### Part 1 - the source

            # Set the source indices.
            row_sectors_indices = s.contracting_tensors_fused_row_legs_indices_per_charges_set[ - 1 + tensordot_step ]
            col_sectors_indices = s.contracting_tensors_fused_col_legs_indices_per_charges_set[ - 1 + tensordot_step ]

            # Set the source matrix shape and charges_sets.
            source_matrix_shape = s.contracted_tensor_as_matrix_shape_post_contraction[ - 1 + tensordot_step ]

            # Set the source charges_set: the order of the charges determines the sectors sorting.
            source_charges_sets = s.row_col_charges_sets[ - 1 + tensordot_step ]

            # Call the function that returns the maps unsorted-source-matrix onto sorted-source-vector.
            map_AB_col, map_C_row, map_D_col = \
                s.get_maps_from_unsorted_matrix_to_sorted_vector( source_matrix_shape,
                                                                  source_charges_sets,
                                                                  row_sectors_indices,
                                                                  col_sectors_indices )

            ########### Part 2 - the querying sectors

            number_tensor_legs =    s.number_legs_contracted_tensor[ s.tensor_contractions_tag ][ tensordot_step ]
            tensor_legs_transposition_order = s.contracted_tensor_transposition_order[ s.tensor_contractions_tag ][ tensordot_step ]

            # Set the querying matrix shape and other related variables.
            querying_matrix_shape = s.contracted_tensor_as_matrix_shape_pre_contraction[ tensordot_step ]
            querying_tensor_shape = s.contracted_tensor_shape_after_transpose[ tensordot_step ]

            # Set the querying charges sets: they might be a subset of the source charges sets. The order does not matter.
            querying_charges_sets = s.central_charges_sets[ tensordot_step ]

            # Map the querying matrix sector indices.
            for charges_set in querying_charges_sets:

                # Call the row and colum querying sector indices.
                querying_sector_indices_row = s.contracting_tensors_fused_row_legs_indices_per_charges_set[ tensordot_step ][ charges_set ]
                querying_sector_indices_col = s.contracting_tensors_fused_cen_legs_indices_per_charges_set[ tensordot_step ][ charges_set ]

                s.indices_from_dense_flattened_tensor_to_blocks_dict[ tensordot_step ][ charges_set ] = \
                    s.get_qsi_vectorial_form( querying_sector_indices_row,
                                              querying_sector_indices_col,
                                              querying_matrix_shape,
                                              querying_tensor_shape,
                                              number_tensor_legs,
                                              tensor_legs_transposition_order,
                                              source_matrix_shape,
                                              charges_set,
                                              map_AB_col,
                                              map_C_row,
                                              map_D_col )


    # =========================================================================
    # The left and right transfer matrices.
    def get_absorbed_dynamic_tensors(s):
        """
        Define the following self objects:

          - s.xxx_blocks[ charges_set ]
          - s.contracting_tensors_fused_cen_legs_indices_per_charges_set
          - s.contracting_tensors_fused_col_legs_indices_per_charges_set

        The Lanczos matvec is performed by means of 4 matmuls, through
        the reshaping of tensors into block-diagonal matrices.

        The quantum numbers associated to rows and columns of both matrices
        are stored in the following lists:
            - s.matvec_fused_row_legs_charges[ matmul_step ][ symmetry ]
            - s.matvec_fused_cen_legs_charges[ matmul_step ][ symmetry ]
            - s.matvec_fused_col_legs_charges[ matmul_step ][ symmetry ]
        'row' accounts for the rows of theta, the 'left' matrix, 'col' accounts
        for the columns of the 'right' matrix, 'cen' accounts for the remaining
        legs, necessarily corresponding in both matrices.

        The dictionaries defined in this function collect the leg indices per
        charge for each matmul step.

        signed_sum: the sum of tensor legs with the vector direction sign preprended.

        General rule: signed_sum(left legs) = -signed_sum(right legs)
        By convention: central_charge = signed_sum(right legs of left tensor)

        L tensor rule:  signed_sum(left legs) = -central_charge
        R tensor rule:      (-central_charge ) = -signed_sum(right legs)
                     : signed_sum(right legs) = central_charge


        Define the following self objects:
          - contracted_tensor_fused_row_legs_indices_per_charges_set[ matmul_step ][ charges_set ] = nparray
          - contracted_tensor_fused_cen_legs_indices_per_charges_set[ matmul_step ][ charges_set ] = nparray
          - contracted_tensor_fused_col_legs_indices_per_charges_set[ matmul_step ][ charges_set ] = nparray
        """

        if s.tensor_contractions_tag == 'matvec':

            s.sought_tensor = 'transfer_matrix'

            s.transfer_matrix_side = 'left'

            s.hltm_blocks,                               \
            hltm_fused_row_legs_indices_per_charges_set, \
            hltm_fused_col_legs_indices_per_charges_set  \
                = s.get_tensor_blocks_and_indices_per_charges_set( )

            s.transfer_matrix_side = 'right'

            s.hrtm_blocks,                               \
            hrtm_fused_row_legs_indices_per_charges_set, \
            hrtm_fused_col_legs_indices_per_charges_set  \
                = s.get_tensor_blocks_and_indices_per_charges_set( )

            # list of the (fused) central legs
            s.contracting_tensors_fused_cen_legs_indices_per_charges_set = [ hltm_fused_row_legs_indices_per_charges_set,
                                                                             s.hmpo_fused_row_legs_indices_per_charges_set[ s.mpo_index_left ],
                                                                             s.hmpo_fused_row_legs_indices_per_charges_set[ s.mpo_index_right ],
                                                                             hrtm_fused_row_legs_indices_per_charges_set ]

            # list of the (fused) column legs
            s.contracting_tensors_fused_col_legs_indices_per_charges_set = [ hltm_fused_col_legs_indices_per_charges_set,
                                                                             s.hmpo_fused_col_legs_indices_per_charges_set[ s.mpo_index_left ],
                                                                             s.hmpo_fused_col_legs_indices_per_charges_set[ s.mpo_index_right ],
                                                                             hrtm_fused_col_legs_indices_per_charges_set ]

            del s.transfer_matrix_side

        elif s.tensor_contractions_tag in ['ltm_mpo_update', 'ltm_opt_update']:

            s.sought_tensor = 'gamma'

            s.gamma_bra_or_ket = 'ket'

            s.gamma_blocks_ket,                               \
            gamma_ket_fused_row_legs_indices_per_charges_set, \
            gamma_ket_fused_col_legs_indices_per_charges_set  \
                = s.get_tensor_blocks_and_indices_per_charges_set( )

            s.gamma_bra_or_ket = 'bra'

            s.gamma_blocks_bra,                               \
            gamma_bra_fused_row_legs_indices_per_charges_set, \
            gamma_bra_fused_col_legs_indices_per_charges_set  \
                = s.get_tensor_blocks_and_indices_per_charges_set( )

            if s.tensor_contractions_tag == 'ltm_mpo_update':

                # list of the (fused) central legs
                s.contracting_tensors_fused_cen_legs_indices_per_charges_set = [ gamma_ket_fused_row_legs_indices_per_charges_set,
                                                                                 s.mpo_fused_row_legs_indices_per_charges_set_for_ltm_update[ s.mpo_operator_tag ][ s.mpo_index_left ],
                                                                                 gamma_bra_fused_row_legs_indices_per_charges_set ]

                # list of the (fused) column legs
                s.contracting_tensors_fused_col_legs_indices_per_charges_set = [ gamma_ket_fused_col_legs_indices_per_charges_set,
                                                                                 s.mpo_fused_col_legs_indices_per_charges_set_for_ltm_update[ s.mpo_operator_tag ][ s.mpo_index_left ],
                                                                                 gamma_bra_fused_col_legs_indices_per_charges_set ]

            elif s.tensor_contractions_tag == 'ltm_opt_update':

                # list of the (fused) central legs
                s.contracting_tensors_fused_cen_legs_indices_per_charges_set = [ gamma_ket_fused_row_legs_indices_per_charges_set,
                                                                                 s.opt_fused_row_legs_indices_per_charges_set_for_ltm_update[ s.mpo_operator_tag ][ s.mpo_index_left ],
                                                                                 gamma_bra_fused_row_legs_indices_per_charges_set ]

                # list of the (fused) column legs
                s.contracting_tensors_fused_col_legs_indices_per_charges_set = [ gamma_ket_fused_col_legs_indices_per_charges_set,
                                                                                 s.opt_fused_col_legs_indices_per_charges_set_for_ltm_update[ s.mpo_operator_tag ][ s.mpo_index_left ],
                                                                                 gamma_bra_fused_col_legs_indices_per_charges_set ]

        elif s.tensor_contractions_tag in ['rtm_mpo_update', 'rtm_opt_update']:

            s.sought_tensor = 'gamma'

            s.gamma_bra_or_ket = 'ket'

            s.gamma_blocks_ket,                               \
            gamma_ket_fused_row_legs_indices_per_charges_set, \
            gamma_ket_fused_col_legs_indices_per_charges_set  \
                = s.get_tensor_blocks_and_indices_per_charges_set( )

            s.gamma_bra_or_ket = 'bra'

            s.gamma_blocks_bra,                               \
            gamma_bra_fused_row_legs_indices_per_charges_set, \
            gamma_bra_fused_col_legs_indices_per_charges_set  \
                = s.get_tensor_blocks_and_indices_per_charges_set( )

            if s.tensor_contractions_tag == 'rtm_mpo_update':

                # list of the (fused) central legs
                s.contracting_tensors_fused_cen_legs_indices_per_charges_set = [ gamma_ket_fused_row_legs_indices_per_charges_set,
                                                                                 s.mpo_fused_row_legs_indices_per_charges_set_for_rtm_update[ s.mpo_operator_tag ][ s.mpo_index_right ],
                                                                                 gamma_bra_fused_row_legs_indices_per_charges_set ]

                # list of the (fused) column legs
                s.contracting_tensors_fused_col_legs_indices_per_charges_set = [ gamma_ket_fused_col_legs_indices_per_charges_set,
                                                                                 s.mpo_fused_col_legs_indices_per_charges_set_for_rtm_update[ s.mpo_operator_tag ][ s.mpo_index_right ],
                                                                                 gamma_bra_fused_col_legs_indices_per_charges_set ]

            elif s.tensor_contractions_tag == 'rtm_opt_update':

                # list of the (fused) central legs
                s.contracting_tensors_fused_cen_legs_indices_per_charges_set = [ gamma_ket_fused_row_legs_indices_per_charges_set,
                                                                                 s.opt_fused_row_legs_indices_per_charges_set_for_rtm_update[ s.mpo_operator_tag ][ s.mpo_index_right ],
                                                                                 gamma_bra_fused_row_legs_indices_per_charges_set ]

                # list of the (fused) column legs
                s.contracting_tensors_fused_col_legs_indices_per_charges_set = [ gamma_ket_fused_col_legs_indices_per_charges_set,
                                                                                 s.opt_fused_col_legs_indices_per_charges_set_for_rtm_update[ s.mpo_operator_tag ][ s.mpo_index_right ],
                                                                                 gamma_bra_fused_col_legs_indices_per_charges_set ]

        del s.sought_tensor

    def get_tensor_blocks_and_indices_per_charges_set(s):
        """
        Return the following local objects:
          - tensor_blocks[ charges_set ]
          - tensor_fused_row_legs_indices_per_charges_set[ charges_set ]
          - tensor_fused_col_legs_indices_per_charges_set[ charges_set ]
        """

        # The charges of the row and col legs are fused and their charges per Abelian
        # symmeyty are summed, the sign being set by the charge vector orientation.
        # These charges arrays are returned as dictionaries in which the keys are,
        # in turn, the name of the Abelian symmetry and the charge numeral.
        # Also, the charges set shared by both legs are obtained and returned.

        # Get the following local objects:
        #   - tensor_fused_row_legs_indices_per_charges_set[ charges_set ]
        #   - tensor_fused_col_legs_indices_per_charges_set[ charges_set ]
        #   - tensor_charges_sets[ ( , , ), ( , , ), ( , , ), ( , , ) ]
        tensor_fused_row_legs_indices_per_charges_set, \
        tensor_fused_col_legs_indices_per_charges_set, \
        row_tensor_charges_sets,                       \
        col_tensor_charges_sets                        \
            = s.get_tensor_indices_per_symmetry_per_charges_set( )

        # The indices are utilized to extract the dense blocks of the block diagonal
        # transfer matrix. Some charges set might have no correspondance to a non-empty
        # or not-null block, in which case the block is discarded and so are the
        # corresponding entries of tensor_fused_row_legs_indices_per_charges_set, which is returned updated.
        #
        # Fix me... check the type etc...
        tensor_charges_sets = s.intersect_charges_sets( row_tensor_charges_sets, col_tensor_charges_sets )

        # Get the following local objects:
        #   - tensor_blocks[ charges_set ]
        #   - tensor_charges_sets
        tensor_blocks = s.get_tensor_not_null_blocks_and_indices_per_charges_set( tensor_charges_sets,
                                                                                  tensor_fused_row_legs_indices_per_charges_set,
                                                                                  tensor_fused_col_legs_indices_per_charges_set )

        return tensor_blocks, tensor_fused_row_legs_indices_per_charges_set, tensor_fused_col_legs_indices_per_charges_set

    def get_tensor_indices_per_symmetry_per_charges_set(s):
        """
        Return the following local objects:
          - tensor_row_indices_per_symmetry_per_charges_set[ charges_set ]
          - tensor_col_indices_per_symmetry_per_charges_set[ charges_set ]
          - row_tensor_charges_sets = [ ( , , ),  ( , , ),  ( , , ),  ( , , ) ]
          - col_tensor_charges_sets = [ ( , , ),  ( , , ),  ( , , ),  ( , , ) ]
        """

        def local_routine( charges_lists_dict, tensor_fused_row_legs_charges_mat, tensor_fused_col_legs_charges_mat ):

            tensor_fused_row_legs_charges, \
            tensor_fused_col_legs_charges, \
                = s.get_tensor_fused_legs_charges( charges_lists_dict )

            tensor_fused_row_legs_charges = atleast_2d( tensor_fused_row_legs_charges )
            tensor_fused_col_legs_charges = atleast_2d( tensor_fused_col_legs_charges )

            if tensor_fused_row_legs_charges_mat.size == 0:
                tensor_fused_row_legs_charges_mat = tensor_fused_row_legs_charges
            else:
                tensor_fused_row_legs_charges_mat = concatenate( ( tensor_fused_row_legs_charges_mat, tensor_fused_row_legs_charges ) )

            if tensor_fused_col_legs_charges_mat.size == 0:
                tensor_fused_col_legs_charges_mat = tensor_fused_col_legs_charges
            else:
                tensor_fused_col_legs_charges_mat = concatenate( ( tensor_fused_col_legs_charges_mat, tensor_fused_col_legs_charges ) )

            return tensor_fused_row_legs_charges_mat, tensor_fused_col_legs_charges_mat

        physical_charges_index = mod( s.lattice_unit_cell_index, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE )

        row_tensor_charges_sets = [ ( ) ]
        col_tensor_charges_sets = [ ( ) ]

        tensor_fused_row_legs_indices_per_charges_set = { }
        tensor_fused_col_legs_indices_per_charges_set = { }

        tensor_fused_row_legs_charges_mat = empty( 0, dtype=npint64 )
        tensor_fused_col_legs_charges_mat = empty( 0, dtype=npint64 )

        for symmetry in s.LIST_SYMMETRIES_NAMES:

            charges_lists_dict = { }

            if s.sought_tensor == 'mpo':

                physical_charges_index = mod( s.lattice_unit_cell_index, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE )
                V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ]

                if symmetry == 'links_alignment':

                    # Left-hand side links.
                    charges_lists_dict['physical_charges'] =  s.physical_charges_normalized['links_set_left'][ physical_charges_index ]
                    charges_lists_dict['mpo_charges_left'] =  s.mpo_charges[ s.mpo_operator_tag ][ symmetry ]
                    charges_lists_dict['mpo_charges_right'] = zeros(V_mpo, dtype = npint64 )

                    tensor_fused_row_legs_charges_mat, \
                    tensor_fused_col_legs_charges_mat, \
                        = local_routine( charges_lists_dict, tensor_fused_row_legs_charges_mat, tensor_fused_col_legs_charges_mat )
                    # ------------------------------------------------------------------------------------------------------------------------------------------
                    # Right-hand side links.
                    charges_lists_dict['physical_charges'] =  s.physical_charges_normalized['links_set_right'][ physical_charges_index ]
                    charges_lists_dict['mpo_charges_left'] =  zeros(V_mpo, dtype = npint64 )
                    charges_lists_dict['mpo_charges_right'] = s.mpo_charges[ s.mpo_operator_tag ][ symmetry ]

                else:

                    charges_lists_dict['physical_charges'] =  s.physical_charges_normalized[ symmetry ][ physical_charges_index ]
                    charges_lists_dict['mpo_charges_left'] =  s.mpo_charges[ s.mpo_operator_tag ][ symmetry ]
                    charges_lists_dict['mpo_charges_right'] = s.mpo_charges[ s.mpo_operator_tag ][ symmetry ]

            elif s.sought_tensor == 'opt':

                physical_charges_index = mod( s.lattice_unit_cell_index, s.PHYSICAL_CHARGES_UNIT_CELL_SIZE )

                if symmetry == 'links_alignment':

                    # Left-hand side links.
                    charges_lists_dict['physical_charges'] =  s.physical_charges_normalized['links_set_left'][ physical_charges_index ]

                    tensor_fused_row_legs_charges_mat, \
                    tensor_fused_col_legs_charges_mat, \
                        = local_routine( charges_lists_dict, tensor_fused_row_legs_charges_mat, tensor_fused_col_legs_charges_mat )

                    # Right-hand side links.
                    charges_lists_dict['physical_charges'] =  s.physical_charges_normalized['links_set_right'][ physical_charges_index ]

                else:

                    charges_lists_dict['physical_charges'] =  s.physical_charges_normalized[ symmetry ][ physical_charges_index ]

            elif s.sought_tensor == 'transfer_matrix':

                if s.transfer_matrix_side == 'left':

                    charges_lists_dict['virtual_charges_left'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                    charges_lists_dict['mpo_charges'] =          s.mpo_charges['hamiltonian'][ symmetry ]

                elif s.transfer_matrix_side == 'right':

                    charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]
                    charges_lists_dict['mpo_charges'] =           s.mpo_charges['hamiltonian'][ symmetry ]

            elif s.sought_tensor == 'gamma':

                if s.tensor_contractions_tag in ['ltm_mpo_update', 'ltm_opt_update']:

                    if symmetry == 'links_alignment':

                        # The ket only contracts the left virtual leg, hence only the left virtual charges are shared.
                        if s.gamma_bra_or_ket == 'ket':

                            # Left-hand side links.
                            charges_lists_dict['physical_charges'] =      s.physical_charges_normalized['links_set_left'][ s.physical_charges_index_left ]
                            charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                            charges_lists_dict['virtual_charges_right'] = zeros( s.chi_centre, dtype=npint64 )

                        # The bra contracts both virtual legs, hence both virtual charges are shared.
                        elif s.gamma_bra_or_ket == 'bra':

                            # Left-hand side links.
                            charges_lists_dict['physical_charges'] =      s.physical_charges_normalized['links_set_left'][ s.physical_charges_index_left ]
                            charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                            charges_lists_dict['virtual_charges_right'] = zeros( s.chi_centre, dtype=npint64 )
                            tensor_fused_row_legs_charges_mat, \
                            tensor_fused_col_legs_charges_mat, \
                                = local_routine( charges_lists_dict, tensor_fused_row_legs_charges_mat, tensor_fused_col_legs_charges_mat )
                            # ------------------------------------------------------------------------------------------------------------------------------------------
                            # Right-hand side links.
                            charges_lists_dict['physical_charges'] =      s.physical_charges_normalized['links_set_right'][ s.physical_charges_index_left ]
                            charges_lists_dict['virtual_charges_left'] =  zeros( s.chi_left, dtype=npint64 )
                            charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]

                    else:

                        charges_lists_dict['physical_charges'] =      s.physical_charges_normalized[ symmetry ][ s.physical_charges_index_left ]
                        charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]

                elif s.tensor_contractions_tag in ['rtm_mpo_update', 'rtm_opt_update']:

                    if symmetry == 'links_alignment':

                        # The ket only contracts the right virtual leg, hence only the left virtual charges are shared.
                        if s.gamma_bra_or_ket == 'ket':

                            # Right-hand side links.
                            charges_lists_dict['physical_charges'] =      s.physical_charges_normalized['links_set_right'][ s.physical_charges_index_right ]
                            charges_lists_dict['virtual_charges_left'] =  zeros( s.chi_centre, dtype=npint64 )
                            charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]

                        # The bra contracts both virtual legs, hence both virtual charges are shared.
                        elif s.gamma_bra_or_ket == 'bra':

                            # Left-hand side links.
                            charges_lists_dict['physical_charges'] =      s.physical_charges_normalized['links_set_left'][ s.physical_charges_index_right ]
                            charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]
                            charges_lists_dict['virtual_charges_right'] = zeros( s.chi_right, dtype=npint64 )
                            tensor_fused_row_legs_charges_mat, \
                            tensor_fused_col_legs_charges_mat, \
                                = local_routine( charges_lists_dict, tensor_fused_row_legs_charges_mat, tensor_fused_col_legs_charges_mat )
                            # ------------------------------------------------------------------------------------------------------------------------------------------
                            # Right-hand side links.
                            charges_lists_dict['physical_charges'] =      s.physical_charges_normalized['links_set_right'][ s.physical_charges_index_right ]
                            charges_lists_dict['virtual_charges_left'] =  zeros( s.chi_centre, dtype=npint64 )
                            charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]

                    else:

                        charges_lists_dict['physical_charges'] =      s.physical_charges_normalized[ symmetry ][ s.physical_charges_index_right ]
                        charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]

            tensor_fused_row_legs_charges_mat, \
            tensor_fused_col_legs_charges_mat, \
                = local_routine( charges_lists_dict, tensor_fused_row_legs_charges_mat, tensor_fused_col_legs_charges_mat )

        # Once the matrix of charges is ready, extract the dictionaries per charges_set.
        tensor_fused_row_legs_charges_mat = atleast_2d( tensor_fused_row_legs_charges_mat )
        tensor_fused_col_legs_charges_mat = atleast_2d( tensor_fused_col_legs_charges_mat )

        fused_rows_len = tensor_fused_row_legs_charges_mat.shape[1]
        fused_cols_len = tensor_fused_col_legs_charges_mat.shape[1]

        map_fused_row_legs_charges = rand( tensor_fused_row_legs_charges_mat.shape[0] )
        map_fused_col_legs_charges = rand( tensor_fused_col_legs_charges_mat.shape[0] )

        fused_rows_mat = concatenate( ( atleast_2d( arange( fused_rows_len ) ), atleast_2d( dot(map_fused_row_legs_charges, tensor_fused_row_legs_charges_mat) ) ) )
        fused_cols_mat = concatenate( ( atleast_2d( arange( fused_cols_len ) ), atleast_2d( dot(map_fused_col_legs_charges, tensor_fused_col_legs_charges_mat) ) ) )

        while fused_rows_mat.shape[1] > 0:

            indices = where( fused_rows_mat[1] == fused_rows_mat[1,0] )[0]
            first_index = int( fused_rows_mat[0,0] )

            charges_set = tuple( tensor_fused_row_legs_charges_mat.T[ first_index ] )

            tensor_fused_row_legs_indices_per_charges_set[ charges_set ] = fused_rows_mat[0][ indices ].astype( npuint64 )
            fused_rows_mat = delete( fused_rows_mat, indices, axis=1 )

        while fused_cols_mat.shape[1] > 0:

            indices = where( fused_cols_mat[1] == fused_cols_mat[1,0] )[0]
            first_index = int( fused_cols_mat[0,0] )

            charges_set = tuple( tensor_fused_col_legs_charges_mat.T[ first_index ] )

            tensor_fused_col_legs_indices_per_charges_set[ charges_set ] = fused_cols_mat[0][ indices ].astype( npuint64 )
            fused_cols_mat = delete( fused_cols_mat, indices, axis=1 )

        row_tensor_charges_sets = list( tensor_fused_row_legs_indices_per_charges_set.keys( ) )
        col_tensor_charges_sets = list( tensor_fused_col_legs_indices_per_charges_set.keys( ) )

        return tensor_fused_row_legs_indices_per_charges_set, tensor_fused_col_legs_indices_per_charges_set, row_tensor_charges_sets, col_tensor_charges_sets

    def get_tensor_not_null_blocks_and_indices_per_charges_set(s, tensor_charges_sets, tensor_fused_row_legs_indices_per_charges_set, tensor_fused_col_legs_indices_per_charges_set):
        """
        Return the following local objects:
          - tensor_blocks[ charges_set ]
          - tensor_fused_row_legs_indices_per_charges_set
        """

        if s.sought_tensor == 'transfer_matrix':

            if s.transfer_matrix_side == 'left':
                tensor = reshape( s.left_mpo_transfer_matrix_list['hamiltonian'][ s.ltm_index_external ], ( s.chi_left, s.chi_left*s.V) )

            elif s.transfer_matrix_side == 'right':
                tensor = reshape( transpose( s.right_mpo_transfer_matrix_list['hamiltonian'][ s.rtm_index_external ], (0,2,1) ), ( s.chi_right*s.V, s.chi_right) )

        elif s.sought_tensor == 'mpo':

            V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ]

            if s.tensor_contractions_tag == 'matvec':
                tensor = reshape( transpose( s.matrix_product_operator['hamiltonian'][ s.lattice_unit_cell_index ], (0,3,1,2) ), (V_mpo*s.d, V_mpo*s.d) )

            elif s.tensor_contractions_tag == 'ltm_mpo_update':
                tensor = reshape( transpose( s.matrix_product_operator[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ], (0,3,1,2) ), (V_mpo*s.d, V_mpo*s.d) )

            elif s.tensor_contractions_tag == 'rtm_mpo_update':
                tensor = reshape( transpose( s.matrix_product_operator[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ], (1,3,0,2) ), (V_mpo*s.d, V_mpo*s.d) )

        elif s.sought_tensor == 'opt':

            tensor = reshape( s.nparray_operator[ s.mpo_operator_tag ][ s.lattice_unit_cell_index ], (s.d, s.d) )

        elif s.sought_tensor == 'gamma':

            if s.tensor_contractions_tag in ['ltm_mpo_update', 'ltm_opt_update']:

                gamma_left = s.gamma_list[ s.gamma_index_left ]

                if s.internal_streamline_current_step_tag == 'mpo_transfer_matrix_list_buildup' and not hasattr(s, 'generating_transfer_matrices_list') and s.lambda_index_centre == s.current_chain_centre_index + 1:
                    gamma_left = tdot( gamma_left, diag( s.lambda_list[ s.lambda_index_left ] ), axes=(1,1) )
                    gamma_left = transpose( gamma_left, (0,2,1) )

                if s.gamma_bra_or_ket == 'ket':
                    tensor = reshape( transpose( gamma_left, (1,2,0) ), ( s.chi_left, s.chi_centre*s.d ) )

                elif s.gamma_bra_or_ket == 'bra':
                    tensor = conj( reshape( transpose( gamma_left, (1,0,2) ), ( s.chi_left*s.d, s.chi_centre ) ) )

            if s.tensor_contractions_tag in ['rtm_mpo_update', 'rtm_opt_update']:

                gamma_right = s.gamma_list[ s.gamma_index_right ]

                if s.internal_streamline_current_step_tag == 'mpo_transfer_matrix_list_buildup' and not hasattr(s, 'generating_transfer_matrices_list') and s.lambda_index_centre == s.current_chain_centre_index:
                    gamma_right = tdot( gamma_right, diag( s.lambda_list[ s.lambda_index_centre ] ), axes=(1,1) )
                    gamma_right = transpose( gamma_right, (0,2,1) )

                if s.gamma_bra_or_ket == 'ket':
                    tensor = reshape( transpose( gamma_right, (2,1,0) ), ( s.chi_right, s.chi_centre*s.d ) )

                elif s.gamma_bra_or_ket == 'bra':
                    tensor = conj( reshape( transpose( gamma_right, (2,0,1) ), ( s.chi_right*s.d, s.chi_centre ) ) )

        tensor_blocks = s.get_blocks_per_charges_set( tensor,
                                                      tensor_charges_sets,
                                                      tensor_fused_row_legs_indices_per_charges_set,
                                                      tensor_fused_col_legs_indices_per_charges_set )

        return tensor_blocks

    # =========================================================================
    # The tensor theta.
    def get_contracted_tensor_indices_per_charges_set(s):
        """
        Define the following self objects:
          - s.contracting_tensors_fused_row_legs_indices_per_charges_set[ tensordot_step ][ charges_set ]
          - s.indices_from_dense_flattened_tensor_to_blocks_dict[ tensordot_step ][ charges_set ]
        """
        s.contracting_tensors_fused_row_legs_indices_per_charges_set = [ ]

        for s.tensordot_step in range( s.number_tensor_contractions[ s.tensor_contractions_tag ] ):
            s.get_contracted_tensor_indices_per_charges_set_per_tensordot_step( )

        del s.tensordot_step

    def get_contracted_tensor_indices_per_charges_set_per_tensordot_step(s):
        """
r        Define the following self objects:
          - contracting_tensors_fused_row_legs_indices_per_charges_set
        """

        # We start by identifying those rows of theta that are to be excluded
        # due to the presence of local symmetries.
        # The next function returns the indices of such rows.

        # Get the following local object:
        #   - invalid_two_sites_fused_row_legs_indices_matvec
        #
        # For practical purposes, we make it a self object and delete it before
        # the end of the function.
        if 'links_alignment' in s.LIST_SYMMETRIES_NAMES and s.tensor_contractions_tag == 'matvec':
            s.invalid_two_sites_fused_row_legs_indices_matvec = s.add_local_symmetries_two_sites_matvec( )

        # We continue by defining the indices per matmul step and, in turn, per
        # charges set, that serve to load the blocks of the block diagonal theta
        # matrix from the vector of not null entries derived from the previous
        # matmul step.

        # Get the following local objects:
        #
        #   - contracted_tensor_row_indices_per_symmetry_per_charge[ symmetry ][ charge ]
        #   - contracted_tensor_charges_sets[ ( , , ), ( , , ), ( , , ), ( , , ) ]
        contracted_tensor_row_indices_per_symmetry_per_charge, \
        contracted_tensor_charges_sets \
            = s.get_contracted_tensor_indices_per_symmetry_per_charge( )

        # Theta has always the same or a higher number of legs than it has the
        # tensor it multiplies to.
        # Therefore, for the purpose of finding the charges sets of theta after
        # contraction, we intersect the row and column sets.
        col_charges_sets = list( s.contracting_tensors_fused_col_legs_indices_per_charges_set[ s.tensordot_step ].keys( ) )
        row_col_tensor_charges_sets = s.intersect_charges_sets( contracted_tensor_charges_sets, col_charges_sets )

        # For the purpose of the Abelian tensordot, the sets are derived from
        # the central charges (see s.get_central_charges_sets).
        #
        # The theta row indices refer to the charges sets shared with the
        # column index, despite some charges sets might not be present in the
        # central leg: those simply won't be considered for the Abelian tdot.
        contracted_tensor_fused_row_legs_indices_per_charges_set \
            = s.build_contracted_tensor_indices_per_charges_set( row_col_tensor_charges_sets, \
                                                                 contracted_tensor_row_indices_per_symmetry_per_charge )

        # list of the (fused) row legs
        s.contracting_tensors_fused_row_legs_indices_per_charges_set.append( contracted_tensor_fused_row_legs_indices_per_charges_set )

        if 'links_alignment' in s.LIST_SYMMETRIES_NAMES and s.tensor_contractions_tag == 'matvec':
            del s.invalid_two_sites_fused_row_legs_indices_matvec


    def add_local_symmetries_two_sites_matvec(s):
        """
        The tensor might have some local constraints that apply to the row legs but
        do not extend to the other legs. Hence, it is about reducing the size
        of the fused legs rather than adding block diagonal structures.

        Return the following local object:
          - invalid_two_sites_fused_row_legs_indices_matvec
        """
        def local_routine( charges_lists_dict, invalid_two_sites_fused_row_legs_indices_matvec ):
            """
            Return the following local objects:
              - invalid_two_sites_fused_row_legs_indices_matvec = nparray
            """
            # theta row legs
            two_sites_fused_row_legs_charges = s.get_contracted_tensor_fused_legs_charges( charges_lists_dict )

            # Extract the indices that preserve the right-most links alignment.
            valid_two_sites_fused_row_legs_indices = where(two_sites_fused_row_legs_charges == 0)[ 0 ].astype( npuint64 )
            new_invalid_two_sites_fused_row_legs_indices_matvec = setdiff1d( arange( len(two_sites_fused_row_legs_charges) ), valid_two_sites_fused_row_legs_indices )

            # Obtain its complementary set.
            invalid_two_sites_fused_row_legs_indices_matvec = union1d( invalid_two_sites_fused_row_legs_indices_matvec, new_invalid_two_sites_fused_row_legs_indices_matvec )

            return invalid_two_sites_fused_row_legs_indices_matvec

        charges_lists_dict = { }
        invalid_two_sites_fused_row_legs_indices_matvec = empty(0, dtype=npint64)
        # arange( s.get_size_two_sites_fused_row_legs_matvec( ) )

        if s.tensordot_step == 0:
            # alignment of the set of links in the centre
            charges_lists_dict['physical_charges_left'] =  s.physical_charges_normalized['links_set_right'][ s.physical_charges_index_left ]
            charges_lists_dict['physical_charges_right'] = s.physical_charges_normalized['links_set_left'][ s.physical_charges_index_right ]
            charges_lists_dict['virtual_charges_right'] =  zeros( s.chi_right, dtype=npint64)
            invalid_two_sites_fused_row_legs_indices_matvec = local_routine( charges_lists_dict, invalid_two_sites_fused_row_legs_indices_matvec )

            # alignment of the set of links on the right-hand side
            charges_lists_dict['physical_charges_left'] =  zeros( s.d, dtype=npint64)
            charges_lists_dict['physical_charges_right'] = s.physical_charges_normalized['links_set_right'][ s.physical_charges_index_right ]
            charges_lists_dict['virtual_charges_right'] =  s.virtual_charges_normalized['links_alignment'][ s.lambda_index_right ]
            invalid_two_sites_fused_row_legs_indices_matvec = local_routine( charges_lists_dict, invalid_two_sites_fused_row_legs_indices_matvec )

        elif s.tensordot_step == 1:
            # alignment of the set of links on the right-hand side
            charges_lists_dict['virtual_charges_left'] =   zeros( s.chi_left, dtype=npint64)
            charges_lists_dict['physical_charges_right'] = s.physical_charges_normalized['links_set_right'][ s.physical_charges_index_right ]
            charges_lists_dict['virtual_charges_right'] =  s.virtual_charges_normalized['links_alignment'][ s.lambda_index_right ]
            invalid_two_sites_fused_row_legs_indices_matvec = local_routine( charges_lists_dict, invalid_two_sites_fused_row_legs_indices_matvec )

        elif s.tensordot_step == 2:
            # alignment of the set of links on the left-hand side
            charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized['links_alignment'][ s.lambda_index_left ]
            charges_lists_dict['virtual_charges_right'] = zeros( s.chi_right, dtype=npint64)
            charges_lists_dict['physical_charges_left'] = s.physical_charges_normalized['links_set_left'][ s.physical_charges_index_left ]
            invalid_two_sites_fused_row_legs_indices_matvec = local_routine( charges_lists_dict, invalid_two_sites_fused_row_legs_indices_matvec )

        elif s.tensordot_step == 3:
            # alignment of the set of links on the left-hand side
            charges_lists_dict['virtual_charges_left'] =   s.virtual_charges_normalized['links_alignment'][ s.lambda_index_left ]
            charges_lists_dict['physical_charges_left'] =  s.physical_charges_normalized['links_set_left'][ s.physical_charges_index_left ]
            charges_lists_dict['physical_charges_right'] = zeros( s.d, dtype=npint64 )
            invalid_two_sites_fused_row_legs_indices_matvec = local_routine( charges_lists_dict, invalid_two_sites_fused_row_legs_indices_matvec )

            # alignment of the set of links in the centre
            charges_lists_dict['virtual_charges_left'] =   zeros( s.chi_left, dtype=npint64)
            charges_lists_dict['physical_charges_left'] =  s.physical_charges_normalized['links_set_right'][ s.physical_charges_index_left ]
            charges_lists_dict['physical_charges_right'] = s.physical_charges_normalized['links_set_left'][ s.physical_charges_index_right ]
            invalid_two_sites_fused_row_legs_indices_matvec = local_routine( charges_lists_dict, invalid_two_sites_fused_row_legs_indices_matvec )

        return invalid_two_sites_fused_row_legs_indices_matvec

    def get_contracted_tensor_indices_per_symmetry_per_charge(s):
        """
        Return the following local objects, for the first matmul step:
          - contracted_tensor_row_indices_per_symmetry_per_charge[ charge ]
          - contracted_tensor_charges_sets[ ( , , ),  ( , , ),  ( , , ),  ( , , ) ]
        """

        def local_routine( symmetry_tag, contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets, charges_lists_dict ):
            """
            Return the following local objects:
              - contracted_tensor_row_indices_per_symmetry_per_charge[ symmetry_tag ] =
              - contracted_tensor_charges_sets = [ ( , , ),  ( , , ),  ( , , ),  ... ]
            """
            contracted_tensor_row_indices_per_charge, \
            unique_two_sites_charges \
                = s.get_contracted_tensor_indices_per_charge( charges_lists_dict )

            contracted_tensor_row_indices_per_symmetry_per_charge[ symmetry_tag ] = contracted_tensor_row_indices_per_charge

            # After identifying the unique charges per symmetry, those can be combined to form
            # all the possible charges set.
            contracted_tensor_charges_sets = s.combine_charges_sets_with_unique_charges( contracted_tensor_charges_sets, unique_two_sites_charges )

            return contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets

        contracted_tensor_charges_sets = [( ) ]
        contracted_tensor_row_indices_per_symmetry_per_charge = { }

        for symmetry in s.LIST_SYMMETRIES_NAMES:

            charges_lists_dict = { }
            symmetry_tag = symmetry

            if s.tensor_contractions_tag == 'matvec':

                if s.tensordot_step == 0:

                    if symmetry == 'links_alignment':
                        charges_lists_dict['physical_charges_left'] =  s.physical_charges_normalized['links_set_left'][ s.physical_charges_index_left ]
                        charges_lists_dict['physical_charges_right'] = zeros( s.d, dtype=npint64 )
                        charges_lists_dict['virtual_charges_right'] =  zeros( s.chi_right, dtype=npint64 )
                        symmetry_tag = 'alignment_external_left_links'

                    else:
                        charges_lists_dict['physical_charges_left'] =  s.physical_charges_normalized[ symmetry ][ s.physical_charges_index_left ]
                        charges_lists_dict['physical_charges_right'] = s.physical_charges_normalized[ symmetry ][ s.physical_charges_index_right ]
                        charges_lists_dict['virtual_charges_right'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]

                elif s.tensordot_step == 1:

                    if symmetry == 'links_alignment':
                        charges_lists_dict['virtual_charges_left'] =   s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                        charges_lists_dict['physical_charges_right'] = zeros( s.d, dtype=npint64 )
                        charges_lists_dict['virtual_charges_right'] =  zeros( s.chi_right, dtype=npint64 )
                        symmetry_tag = 'alignment_external_left_links'
                        contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets = local_routine( symmetry_tag, contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets, charges_lists_dict )
                        #
                        charges_lists_dict['virtual_charges_left'] =   zeros( s.chi_left, dtype=npint64 )
                        charges_lists_dict['physical_charges_right'] = s.physical_charges_normalized['links_set_left'][ s.physical_charges_index_right ]
                        charges_lists_dict['virtual_charges_right'] =  zeros( s.chi_right, dtype=npint64 )
                        symmetry_tag = 'alignment_central_links'

                    else:
                        charges_lists_dict['virtual_charges_left'] =   s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                        charges_lists_dict['physical_charges_right'] = s.physical_charges_normalized[ symmetry ][ s.physical_charges_index_right ]
                        charges_lists_dict['virtual_charges_right'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]

                elif s.tensordot_step == 2:

                    if symmetry == 'links_alignment':
                        charges_lists_dict['virtual_charges_left'] =  zeros( s.chi_left, dtype=npint64 )
                        charges_lists_dict['virtual_charges_right'] = zeros( s.chi_right, dtype=npint64 )
                        charges_lists_dict['physical_charges_left'] = s.physical_charges_normalized['links_set_right'][ s.physical_charges_index_left ]
                        symmetry_tag = 'alignment_central_links'
                        contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets = local_routine( symmetry_tag, contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets, charges_lists_dict )
                        #
                        charges_lists_dict['virtual_charges_left'] =  zeros( s.chi_left, dtype=npint64 )
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]
                        charges_lists_dict['physical_charges_left'] = zeros( s.d, dtype=npint64 )
                        symmetry_tag = 'alignment_external_right_links'

                    else:
                        charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]
                        charges_lists_dict['physical_charges_left'] = s.physical_charges_normalized[ symmetry ][ s.physical_charges_index_left ]

                elif s.tensordot_step == 3:

                    if symmetry == 'links_alignment':
                        charges_lists_dict['virtual_charges_left'] =   zeros( s.chi_left, dtype=npint64 )
                        charges_lists_dict['physical_charges_left'] =  zeros( s.d, dtype=npint64 )
                        charges_lists_dict['physical_charges_right'] = s.physical_charges_normalized['links_set_right'][ s.physical_charges_index_right ]
                        symmetry_tag = 'alignment_external_right_links'

                    else:
                        charges_lists_dict['virtual_charges_left'] =   s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                        charges_lists_dict['physical_charges_left'] =  s.physical_charges_normalized[ symmetry ][ s.physical_charges_index_left ]
                        charges_lists_dict['physical_charges_right'] = s.physical_charges_normalized[ symmetry ][ s.physical_charges_index_right ]

            elif s.tensor_contractions_tag == 'ltm_mpo_update':

                V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ]

                if s.tensordot_step == 0:

                    charges_lists_dict['virtual_charges_left'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                    charges_lists_dict['mpo_charges'] =          s.mpo_charges[ s.mpo_operator_tag ][ symmetry ]

                    if symmetry == 'links_alignment':
                        symmetry_tag = 'alignment_left_links'

                elif s.tensordot_step == 1:

                    if symmetry == 'links_alignment':

                        # Left-hand side links.
                        charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                        charges_lists_dict['virtual_charges_right'] = zeros( s.chi_centre, dtype=npint64 )

                        symmetry_tag = 'alignment_left_links'
                        contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets = \
                            local_routine( symmetry_tag, contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets, charges_lists_dict )
                        # ------------------------------------------------------------------------------------------------------------------------------------------
                        # Right-hand side links.
                        charges_lists_dict['virtual_charges_left'] =  zeros( s.chi_left, dtype=npint64 )
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]

                        symmetry_tag = 'alignment_right_links'

                    else:

                        charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]

                elif s.tensordot_step == 2:

                    if symmetry == 'links_alignment':

                        # Left-hand side links.
                        charges_lists_dict['virtual_charges_right'] = zeros( s.chi_centre, dtype=npint64 )
                        charges_lists_dict['mpo_charges'] =           zeros( V_mpo, dtype=npint64 )

                        symmetry_tag = 'alignment_left_links'
                        contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets = \
                            local_routine( symmetry_tag, contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets, charges_lists_dict )
                        # ------------------------------------------------------------------------------------------------------------------------------------------
                        # Right-hand side links.
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]
                        charges_lists_dict['mpo_charges'] =           s.mpo_charges[ s.mpo_operator_tag ][ symmetry ]

                        symmetry_tag = 'alignment_right_links'

                    else:

                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]
                        charges_lists_dict['mpo_charges'] =           s.mpo_charges[ s.mpo_operator_tag ][ symmetry ]

            elif s.tensor_contractions_tag == 'ltm_opt_update':

                if s.tensordot_step == 0:

                    charges_lists_dict['virtual_charges_left'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]

                    if symmetry == 'links_alignment':
                        symmetry_tag = 'alignment_left_links'

                elif s.tensordot_step == 1:

                    if symmetry == 'links_alignment':

                        # Left-hand side links.
                        charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                        charges_lists_dict['virtual_charges_right'] = zeros( s.chi_centre, dtype=npint64 )

                        symmetry_tag = 'alignment_left_links'
                        contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets = \
                            local_routine( symmetry_tag, contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets, charges_lists_dict )
                        # ------------------------------------------------------------------------------------------------------------------------------------------
                        # Right-hand side links.
                        charges_lists_dict['virtual_charges_left'] =  zeros( s.chi_left, dtype=npint64 )
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]

                        symmetry_tag = 'alignment_right_links'

                    else:

                        charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]

                elif s.tensordot_step == 2:

                    if symmetry == 'links_alignment':

                        # Left-hand side links.
                        charges_lists_dict['virtual_charges_right'] = zeros( s.chi_centre, dtype=npint64 )

                        symmetry_tag = 'alignment_left_links'
                        contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets = \
                            local_routine( symmetry_tag, contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets, charges_lists_dict )
                        # ------------------------------------------------------------------------------------------------------------------------------------------
                        # Right-hand side links.
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]

                        symmetry_tag = 'alignment_right_links'

                    else:

                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]

            elif s.tensor_contractions_tag == 'rtm_mpo_update':

                V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ]

                if s.tensordot_step == 0:

                    charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]
                    charges_lists_dict['mpo_charges'] =           s.mpo_charges[ s.mpo_operator_tag ][ symmetry ]

                    if symmetry == 'links_alignment':
                        symmetry_tag = 'alignment_right_links'

                elif s.tensordot_step == 1:

                    if symmetry == 'links_alignment':

                        # Left-hand side links.
                        charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]
                        charges_lists_dict['virtual_charges_right'] = zeros( s.chi_right, dtype=npint64 )

                        symmetry_tag = 'alignment_left_links'
                        contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets = \
                            local_routine( symmetry_tag, contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets, charges_lists_dict )
                        # ------------------------------------------------------------------------------------------------------------------------------------------
                        # Right-hand side links.
                        charges_lists_dict['virtual_charges_left'] =  zeros( s.chi_centre, dtype=npint64 )
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]

                        symmetry_tag = 'alignment_right_links'

                    else:

                        charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]

                elif s.tensordot_step == 2:

                    if symmetry == 'links_alignment':

                        # Left-hand side links.
                        charges_lists_dict['virtual_charges_left'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]
                        charges_lists_dict['mpo_charges'] =          s.mpo_charges[ s.mpo_operator_tag ][ symmetry ]

                        symmetry_tag = 'alignment_left_links'
                        contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets = \
                            local_routine( symmetry_tag, contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets, charges_lists_dict )
                        # ------------------------------------------------------------------------------------------------------------------------------------------
                        # Right-hand side links.
                        charges_lists_dict['virtual_charges_left'] = zeros( s.chi_centre, dtype=npint64 )
                        charges_lists_dict['mpo_charges'] =          zeros( V_mpo, dtype=npint64 )

                        symmetry_tag = 'alignment_right_links'

                    else:

                        charges_lists_dict['virtual_charges_left'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]
                        charges_lists_dict['mpo_charges'] =          s.mpo_charges[ s.mpo_operator_tag ][ symmetry ]

            elif s.tensor_contractions_tag == 'rtm_opt_update':

                if s.tensordot_step == 0:

                    charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]

                    if symmetry == 'links_alignment':
                        symmetry_tag = 'alignment_right_links'

                elif s.tensordot_step == 1:

                    if symmetry == 'links_alignment':

                        # Left-hand side links.
                        charges_lists_dict['virtual_charges_right'] = zeros( s.chi_right, dtype=npint64 )
                        charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]

                        symmetry_tag = 'alignment_left_links'
                        contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets = \
                            local_routine( symmetry_tag, contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets, charges_lists_dict )
                        # ------------------------------------------------------------------------------------------------------------------------------------------
                        # Right-hand side links.
                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]
                        charges_lists_dict['virtual_charges_left'] =  zeros( s.chi_centre, dtype=npint64 )

                        symmetry_tag = 'alignment_right_links'

                    else:

                        charges_lists_dict['virtual_charges_right'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]
                        charges_lists_dict['virtual_charges_left'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]

                elif s.tensordot_step == 2:

                    if symmetry == 'links_alignment':

                        # Left-hand side links.
                        charges_lists_dict['virtual_charges_left'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]

                        symmetry_tag = 'alignment_left_links'
                        contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets = \
                            local_routine( symmetry_tag, contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets, charges_lists_dict )
                        # ------------------------------------------------------------------------------------------------------------------------------------------
                        # Right-hand side links.
                        charges_lists_dict['virtual_charges_left'] = zeros( s.chi_centre, dtype=npint64 )

                        symmetry_tag = 'alignment_right_links'

                    else:

                        charges_lists_dict['virtual_charges_left'] = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]

            contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets = \
                local_routine( symmetry_tag, contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets, charges_lists_dict )

        return contracted_tensor_row_indices_per_symmetry_per_charge, contracted_tensor_charges_sets


    def build_contracted_tensor_indices_per_charges_set(s, row_col_tensor_charges_sets, contracted_tensor_row_indices_per_symmetry_per_charge ):
        """
        Return the following local objects:
          - contracted_tensor_fused_row_legs_indices_per_charges_set[ charges_set ]
        """
        contracted_tensor_fused_row_legs_indices_per_charges_set = { }
        ordered_symmetries_names = list( contracted_tensor_row_indices_per_symmetry_per_charge.keys( ) )

        for charges_set in row_col_tensor_charges_sets:
            row_ind_charges_set = empty(0)

            for i in range( len( charges_set ) ):
                charge = charges_set[ i ]
                symmetry_tag = ordered_symmetries_names[ i ]
                row_ind_charge = contracted_tensor_row_indices_per_symmetry_per_charge[ symmetry_tag ][charge ]

                if i == 0:
                    row_ind_charges_set = row_ind_charge.copy( )
                else:
                    row_ind_charges_set = intersect1d( row_ind_charges_set, row_ind_charge )

            contracted_tensor_fused_row_legs_indices_per_charges_set[ charges_set ] = row_ind_charges_set

        return contracted_tensor_fused_row_legs_indices_per_charges_set

    def get_contracted_tensor_fused_legs_charges(s, charges_lists_dict):
        """
        Return the following local objects:
          - contracted_tensor_fused_row_legs_charges[ charge ]
        """
        if s.tensor_contractions_tag == 'matvec':

            if s.tensordot_step == 0:
                # ===================== two-sites matmul step 0 ===================== #
                #    _                   _____________________                        #
                #   | |-->-- 0   3 -->--|_____________________|-->-- 2                #
                #   | |                       ^         ^                             #
                #   | |                       |  theta  |                             #
                #   | |                       0         1                             #
                #   | |  ltm                                                          #
                #   | |                                                               #
                #   | |-->-- 2                                                        #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   |_|--<-- 1                                                        #
                #                                                                     #
                # ===================== two-sites matmul step 0 ===================== #
                physical_charges_left =  charges_lists_dict['physical_charges_left']
                physical_charges_right = charges_lists_dict['physical_charges_right']
                virtual_charges_right =  charges_lists_dict['virtual_charges_right']

                # theta row legs
                two_sites_fused_row_legs_charges = s.generate_list_charges([ - physical_charges_left ,
                                                                             - physical_charges_right,
                                                                             + virtual_charges_right ], dtype=npint64 )

            elif s.tensordot_step == 1:
                # ===================== two-sites matmul step 1 ===================== #
                #    _________________________________________                        #
                #   |  _______________________________________|-->-- 2                #
                #   | |                  |               |                            #
                #   | |                  ^               ^                            #
                #   | |                  |               |                            #
                #   | |                  4               1                            #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                  1                                            #
                #   | |                  |                                            #
                #   | | ltm              ^                                            #
                #   | |                  ⟂                                            #
                #   | |-->-- 3   0 -->--| |-->-- 2   MPO                              #
                #   | |                  T                                            #
                #   | |                  ^                                            #
                #   | |                  |                                            #
                #   | |                  3                                            #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   |_|--<-- 0                                                        #
                #                                                                     #
                # ===================== two-sites matmul step 1 ===================== #
                virtual_charges_left =   charges_lists_dict['virtual_charges_left']
                physical_charges_right = charges_lists_dict['physical_charges_right']
                virtual_charges_right =  charges_lists_dict['virtual_charges_right']

                # theta row legs
                two_sites_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_left  ,
                                                                         - physical_charges_right,
                                                                         + virtual_charges_right ], dtype=npint64 )

            elif s.tensordot_step == 2:
                # ===================== two-sites matmul step 2 ===================== #
                #    _________________________________________                        #
                #   |          _______________________________|-->-- 1                #
                #   |         |                  |                                    #
                #   |         |                  ^                                    #
                #   |         |                  |                                    #
                #   |         |                  4                                    #
                #   |         |                                                       #
                #   |         |                                                       #
                #   |         |                  1                                    #
                #   |         |                  |                                    #
                #   |         |                  ^                                    #
                #   |         |                  ⟂                                    #
                #   |  theta  |-->-- 3   0 -->--| |-->-- 2   MPO                      #
                #   |         |                  T                                    #
                #   |         |                  ^                                    #
                #   |         |                  |                                    #
                #   |  _______|                  3                                    #
                #   | |   |                                                           #
                #   | |   ^                                                           #
                #   | |   |                                                           #
                #   | |   2                                                           #
                #   | |                                                               #
                #   | |                                                               #
                #   |_|--<-- 0                                                        #
                #                                                                     #
                # ===================== two-sites matmul step 2 ===================== #
                virtual_charges_left =  charges_lists_dict['virtual_charges_left']
                virtual_charges_right = charges_lists_dict['virtual_charges_right']
                physical_charges_left = charges_lists_dict['physical_charges_left']

                # theta row legs
                two_sites_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_left ,
                                                                         + virtual_charges_right,
                                                                         - physical_charges_left ], dtype=npint64 )

            elif s.tensordot_step == 3:
                # ===================== two-sites matmul step 3 ===================== #
                #    _________________________________________                   _    #
                #   |                                         |-->-- 3   0 -->--| |   #
                #   |                                         |                 | |   #
                #   |                                         |                 | |   #
                #   |                                         |                 | |   #
                #   |                                         |                 | |   #
                #   |                                         |                 | |   #
                #   |                                         |           hrtm  | |   #
                #   |                  theta                  |                 | |   #
                #   |                                         |                 | |   #
                #   |                                         |                 | |   #
                #   |                                         |                 | |   #
                #   |                                         |-->-- 4   1 -->--| |   #
                #   |                                         |                 | |   #
                #   |                                         |                 | |   #
                #   |                                         |                 | |   #
                #   |  _______________________________________|                 | |   #
                #   | |           |               |                             | |   #
                #   | |           ^               ^                             | |   #
                #   | |           |               |                             | |   #
                #   | |           1               2                             | |   #
                #   | |                                                         | |   #
                #   | |                                                         | |   #
                #   |_|--<-- 0                                           2 --<--|_|   #
                #                                                                     #
                # ===================== two-sites matmul step 3 ===================== #
                virtual_charges_left =   charges_lists_dict['virtual_charges_left']
                physical_charges_left =  charges_lists_dict['physical_charges_left']
                physical_charges_right = charges_lists_dict['physical_charges_right']

                # theta row legs
                two_sites_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_left   ,
                                                                         - physical_charges_left  ,
                                                                         - physical_charges_right ], dtype=npint64 )

            contracted_tensor_fused_row_legs_charges = - two_sites_fused_row_legs_charges # see WIKI CHARGE CONVENTION

        elif s.tensor_contractions_tag == 'ltm_mpo_update':

            if s.tensordot_step == 0:
                # ====================== ltm mpo matmul_step 0 ====================== #
                #    _                   ___________                                  #
                #   | |-->-- 2   0 -->--|___________|-->-- 1                          #
                #   | |                       ^                                       #
                #   | |                       |  Gamma                                #
                #   | |                       2                                       #
                #   | |  ltm                                                          #
                #   | |                                                               #
                #   | |-->-- 1                                                        #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   |_|--<-- 0                                                        #
                #                                                                     #
                # ====================== ltm mpo matmul_step 0 ====================== #
                virtual_charges_left = charges_lists_dict['virtual_charges_left']
                mpo_charges =          charges_lists_dict['mpo_charges']

                # ltm row legs
                ltm_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_left ,
                                                                       + mpo_charges          ], dtype=npint64 )

            elif s.tensordot_step == 1:
                # ====================== ltm mpo matmul_step 1 ====================== #
                #    _______________________________                                  #
                #   |  _____________________________|-->-- 0                          #
                #   | |                       ^                                       #
                #   | |                       |                  1                    #
                #   | |                       3                  |                    #
                #   | |  ltm                                     ^                    #
                #   | |                                          ⟂                    #
                #   | |-->-- 2                           0 -->--| |-->-- 2   MPO      #
                #   | |                                          T                    #
                #   | |                                          ^                    #
                #   | |                                          |                    #
                #   | |                                          3                    #
                #   | |                                                               #
                #   |_|--<-- 1                                                        #
                #                                                                     #
                # ====================== ltm mpo matmul_step 1 ====================== #
                virtual_charges_left =  charges_lists_dict['virtual_charges_left']
                virtual_charges_right = charges_lists_dict['virtual_charges_right']

                # ltm row legs
                ltm_fused_row_legs_charges = s.generate_list_charges([ + virtual_charges_right ,
                                                                       - virtual_charges_left  ], dtype=npint64 )

            elif s.tensordot_step == 2:
                # ====================== ltm mpo matmul_step 2 ====================== #
                #    _______________________________                                  #
                #   |                               |-->-- 0                          #
                #   |                               |                                 #
                #   |                               |                                 #
                #   |                               |-->-- 1                          #
                #   |  _____________________________|                                 #
                #   | |                       ^                                       #
                #   | |                       |                                       #
                #   | |                       3                                       #
                #   | |                                                               #
                #   | |                       ^                                       #
                #   | |                       1   Gamma                               #
                #   | |                  _____|_____                                  #
                #   |_|--<-- 2   0 --<--|___________|--<-- 2                          #
                #                                                                     #
                # ====================== ltm mpo matmul_step 2 ====================== #
                virtual_charges_right = charges_lists_dict['virtual_charges_right']
                mpo_charges =           charges_lists_dict['mpo_charges']

                # ltm row legs
                ltm_fused_row_legs_charges = s.generate_list_charges([ + virtual_charges_right ,
                                                                       + mpo_charges           ], dtype=npint64 )

            contracted_tensor_fused_row_legs_charges = - ltm_fused_row_legs_charges # see WIKI CHARGE CONVENTION

        elif s.tensor_contractions_tag == 'ltm_opt_update':

            if s.tensordot_step == 0:#scheminibelli
                # ====================== ltm opt matmul_step 1 ====================== #
                #    _                   ___________                                  #
                #   | |-->-- 1   0 -->--|___________|-->-- 1                          #
                #   | |                       ^                                       #
                #   | |                       |  Gamma                                #
                #   | |                       2                                       #
                #   | |  ltm                                                          #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   | |                                                               #
                #   |_|--<-- 0                                                        #
                #                                                                     #
                # ====================== ltm opt matmul_step 1 ====================== #
                virtual_charges_left = charges_lists_dict['virtual_charges_left']

                # ltm row legs
                ltm_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_left ], dtype=npint64 )

            elif s.tensordot_step == 1:
                # ====================== ltm opt matmul_step 1 ====================== #
                #    _______________________________                                  #
                #   |  _____________________________|-->-- 1                          #
                #   | |                       ^                                       #
                #   | |                       |                  0                    #
                #   | |                       2                  |                    #
                #   | |  ltm                                     ^                    #
                #   | |                                          ⟂                    #
                #   | |                                         | |    opt            #
                #   | |                                          T                    #
                #   | |                                          ^                    #
                #   | |                                          |                    #
                #   | |                                          1                    #
                #   | |                                                               #
                #   |_|--<-- 0                                                        #
                #                                                                     #
                # ====================== ltm opt matmul_step 1 ====================== #
                virtual_charges_left =  charges_lists_dict['virtual_charges_left']
                virtual_charges_right = charges_lists_dict['virtual_charges_right']

                # ltm row legs
                ltm_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_left  ,
                                                                       + virtual_charges_right ], dtype=npint64 )

            elif s.tensordot_step == 2:
                # ====================== ltm opt matmul_step 2 ====================== #
                #    _______________________________                                  #
                #   |  _____________________________|-->-- 0                          #
                #   | |                       ^                                       #
                #   | |                       |                                       #
                #   | |                       |                                       #
                #   | |  ltm                  |                                       #
                #   | |                       |                                       #
                #   | |                       |                                       #
                #   | |                       2                                       #
                #   | |                                                               #
                #   | |                       ^                                       #
                #   | |                       1   Gamma                               #
                #   | |                  _____|_____                                  #
                #   |_|--<-- 1   0 --<--|___________|--<-- 2                          #
                #                                                                     #
                # ====================== ltm opt matmul_step 2 ====================== #
                virtual_charges_right = charges_lists_dict['virtual_charges_right']

                # ltm row legs
                ltm_fused_row_legs_charges = s.generate_list_charges([ + virtual_charges_right ], dtype=npint64 )

            contracted_tensor_fused_row_legs_charges = - ltm_fused_row_legs_charges # see WIKI CHARGE CONVENTION

        elif s.tensor_contractions_tag == 'rtm_mpo_update':

            if s.tensordot_step == 0:
                # ====================== rtm mpo matmul_step 0 ====================== #
                #                                  ___________                   _    #
                #                          1 -->--|___________|-->-- 0   2 -->--| |   #
                #                                       ^                       | |   #
                #                                       |                       | |   #
                #                                       2                       | |   #
                #                                                          rtm  | |   #
                #                                                               | |   #
                #                                                        1 -->--| |   #
                #                                                               | |   #
                #                                                               | |   #
                #                                                               | |   #
                #                                                               | |   #
                #                                                               | |   #
                #                                                        0 --<--|_|   #
                #                                                                     #
                # ====================== rtm mpo matmul_step 0 ====================== #
                virtual_charges_right = charges_lists_dict['virtual_charges_right']
                mpo_charges =           charges_lists_dict['mpo_charges']

                # ltm row legs
                rtm_fused_row_legs_charges = s.generate_list_charges([ + virtual_charges_right ,
                                                                       - mpo_charges           ], dtype=npint64 )

            elif s.tensordot_step == 1:
                # ====================== rtm mpo matmul_step 1 ====================== #
                #                                  _______________________________    #
                #                          0 -->--|_____________________________  |   #
                #                                       ^                       | |   #
                #              1                        |                       | |   #
                #              |                        3                       | |   #
                #              ^                                           rtm  | |   #
                #              ⟂                                                | |   #
                #      2 -->--| |-->-- 0   MPO                           2 -->--| |   #
                #              T                                                | |   #
                #              ^                                                | |   #
                #              |                                                | |   #
                #              3                                                | |   #
                #                                                               | |   #
                #                                                        1 --<--|_|   #
                #                                                                     #
                # ====================== rtm mpo matmul_step 1 ====================== #
                virtual_charges_left =  charges_lists_dict['virtual_charges_left']
                virtual_charges_right = charges_lists_dict['virtual_charges_right']

                # ltm row legs
                rtm_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_left        ,
                                                                       + virtual_charges_right      ], dtype=npint64 )

            elif s.tensordot_step == 2:
                # ====================== rtm mpo matmul_step 2 ====================== #
                #                                  _______________________________    #
                #                          0 -->--|                               |   #
                #                                 |                               |   #
                #                                 |                               |   #
                #                          1 -->--|                               |   #
                #                                 |_____________________________  |   #
                #                                       ^                       | |   #
                #                                       |                       | |   #
                #                                       3                       | |   #
                #                                                               | |   #
                #                                       ^                       | |   #
                #                                       1                       | |   #
                #                                  _____|_____                  | |   #
                #                          2 --<--|___________|--<-- 0   2 --<--|_|   #
                #                                                                     #
                # ====================== rtm mpo matmul_step 2 ====================== #
                virtual_charges_left = charges_lists_dict['virtual_charges_left']
                mpo_charges =          charges_lists_dict['mpo_charges']

                # ltm row legs
                rtm_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_left ,
                                                                       - mpo_charges ], dtype=npint64 )

            contracted_tensor_fused_row_legs_charges = - rtm_fused_row_legs_charges # see WIKI CHARGE CONVENTION

        elif s.tensor_contractions_tag == 'rtm_opt_update':

            if s.tensordot_step == 0:
                # ====================== rtm opt matmul_step 0 ====================== #
                #                                  ___________                   _    #
                #                          1 -->--|___________|-->-- 0   1 -->--| |   #
                #                                       ^                       | |   #
                #                                       |                       | |   #
                #                                       2                       | |   #
                #                                                          rtm  | |   #
                #                                                               | |   #
                #                                                               | |   #
                #                                                               | |   #
                #                                                               | |   #
                #                                                               | |   #
                #                                                               | |   #
                #                                                               | |   #
                #                                                        0 --<--|_|   #
                #                                                                     #
                # ====================== rtm opt matmul_step 0 ====================== #
                virtual_charges_right = charges_lists_dict['virtual_charges_right']

                # ltm row legs
                rtm_fused_row_legs_charges = s.generate_list_charges([ + virtual_charges_right ], dtype=npint64 )

            elif s.tensordot_step == 1:
                # ====================== rtm opt matmul_step 1 ====================== #
                #                                  _______________________________    #
                #                          1 -->--|_____________________________  |   #
                #                                       ^                       | |   #
                #              0                        |                       | |   #
                #              |                        2                       | |   #
                #              ^                                           rtm  | |   #
                #              ⟂                                                | |   #
                #             | |   opt                                         | |   #
                #              T                                                | |   #
                #              ^                                                | |   #
                #              |                                                | |   #
                #              1                                                | |   #
                #                                                               | |   #
                #                                                        0 --<--|_|   #
                #                                                                     #
                # ====================== rtm opt matmul_step 2 ====================== #
                virtual_charges_right = charges_lists_dict['virtual_charges_right']
                virtual_charges_left =  charges_lists_dict['virtual_charges_left'  ]

                # ltm row legs
                rtm_fused_row_legs_charges = s.generate_list_charges([ + virtual_charges_right  ,
                                                                       - virtual_charges_left ], dtype=npint64 )

            elif s.tensordot_step == 2:
                # ====================== rtm opt matmul_step 2 ====================== #
                #                                  _______________________________    #
                #                          0 -->--|_____________________________| |   #
                #                                       ^                       | |   #
                #                                       |                       | |   #
                #                                       |                       | |   #
                #                                       |                  rtm  | |   #
                #                                       |                       | |   #
                #                                       |                       | |   #
                #                                       2                       | |   #
                #                                                               | |   #
                #                                       ^                       | |   #
                #                                       1                       | |   #
                #                                  _____|_____                  | |   #
                #                          2 --<--|___________|--<-- 0   1 --<--|_|   #
                #                                                                     #
                # ====================== rtm opt matmul_step 1 ====================== #
                virtual_charges_left = charges_lists_dict['virtual_charges_left']

                # ltm row legs
                rtm_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_left ], dtype=npint64 )

            contracted_tensor_fused_row_legs_charges = - rtm_fused_row_legs_charges # see WIKI CHARGE CONVENTION

        return contracted_tensor_fused_row_legs_charges

    def get_size_two_sites_fused_row_legs_matvec(s):

        if s.tensordot_step == 0:
            size_two_sites_fused_row_legs = s.d**2*s.chi_right

        elif s.tensordot_step == 1:
            size_two_sites_fused_row_legs = s.d*s.chi_left*s.chi_right

        elif s.tensordot_step == 2:
            size_two_sites_fused_row_legs = s.d*s.chi_left*s.chi_right

        elif s.tensordot_step == 3:
            size_two_sites_fused_row_legs = s.d**2*s.chi_left

        return size_two_sites_fused_row_legs

    def get_contracted_tensor_indices_per_charge(s, charges_lists_dict):
        """
        Return the following local objects:
          - contracted_tensor_row_indices_per_charge[ charge ] = nparray
          - unique_two_sites_charges = nparray
        """
        contracted_tensor_fused_row_legs_charges = s.get_contracted_tensor_fused_legs_charges( charges_lists_dict )

        # The unique set of charges in the fused row legs.
        unique_two_sites_charges = unique( contracted_tensor_fused_row_legs_charges )

        # For QLM, the overlap of the links sets determines an additional local symmetry
        # that allows to further reduce the size of the fused row leg of theta.
        if 'links_alignment' in s.LIST_SYMMETRIES_NAMES and s.tensor_contractions_tag == 'matvec':
            invalid_charge = choice( setdiff1d( arange( min( unique_two_sites_charges )-1, max(unique_two_sites_charges) + 2), unique_two_sites_charges ) )

            # Set off the elements that do not belong in valid_two_sites_fused_row_legs_indices.
            contracted_tensor_fused_row_legs_charges[ s.invalid_two_sites_fused_row_legs_indices_matvec ] = invalid_charge
            # UNCOMMENT INVALID CHARGE

        # Extract the indices of the col leg with entries equal to charge.
        contracted_tensor_row_indices_per_charge = { }
        for charge in unique_two_sites_charges:
            contracted_tensor_row_indices_per_charge[charge ] = where( contracted_tensor_fused_row_legs_charges == charge )[ 0 ].astype( npuint64 )

        return contracted_tensor_row_indices_per_charge, unique_two_sites_charges


    def get_contracted_tensor_shape(s):
        """
        If s.chi_left gets updated, the shape of tensor is to be updated as well.

        Define the following local objects: for convenience, we define them as
        self objects and delete them at the end of the function:

          - contracted_tensor_shape_after_transpose[ matmul_step ] =  ( tuple )
          - contracted_tensor_as_matrix_shape_pre_contraction[ matmul_step ] =    ( tuple )
          - contracted_tensor_as_matrix_shape_post_contraction[ matmul_step ] =     ( tuple )
        """

        if s.tensor_contractions_tag == 'matvec':

            V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ]

            s.contracted_tensor_shape_after_transpose  = [ ( s.d, s.d, s.chi_right,         s.chi_left         ),
                                                           ( s.chi_left, s.d, s.chi_right,  V_mpo, s.d         ),
                                                           ( s.chi_left, s.chi_right, s.d,  V_mpo, s.d         ),
                                                           ( s.chi_left, s.d, s.d,          s.chi_right, V_mpo ), ]

            s.contracted_tensor_as_matrix_shape_pre_contraction = [ ( s.d*s.d*s.chi_right,         s.chi_left        ),
                                                                  ( s.chi_left*s.d*s.chi_right,  V_mpo*s.d         ),
                                                                  ( s.chi_left*s.chi_right*s.d,  V_mpo*s.d         ),
                                                                  ( s.chi_left*s.d*s.d,          s.chi_right*V_mpo ), ]

            s.contracted_tensor_shape_after_matmul = [ ( s.d, s.d, s.chi_right,         s.chi_left, V_mpo ),
                                                       ( s.chi_left, s.d, s.chi_right,  V_mpo, s.d        ),
                                                       ( s.chi_left, s.chi_right, s.d,  V_mpo, s.d        ),
                                                       ( s.chi_left, s.d, s.d,          s.chi_right       ), ]

            s.contracted_tensor_as_matrix_shape_post_contraction = [ ( s.d*s.d*s.chi_right,         s.chi_left*V_mpo ),
                                                                     ( s.chi_left*s.d*s.chi_right,  V_mpo*s.d        ),
                                                                     ( s.chi_left*s.chi_right*s.d,  V_mpo*s.d        ),
                                                                     ( s.chi_left*s.d*s.d,          s.chi_right      ), ]

        elif s.tensor_contractions_tag == 'ltm_mpo_update':

            V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ]

            s.contracted_tensor_shape_after_transpose  = [ ( s.chi_left, V_mpo,           s.chi_left      ),
                                                           ( s.chi_centre, s.chi_left,    V_mpo, s.d      ),
                                                           ( s.chi_centre, V_mpo,         s.chi_left, s.d ), ]

            s.contracted_tensor_as_matrix_shape_pre_contraction = [ ( s.chi_left*V_mpo,          s.chi_left       ),
                                                                    ( s.chi_centre*s.chi_left,   V_mpo*s.d        ),
                                                                    ( s.chi_centre*V_mpo,        s.chi_left*s.d  ), ]

            s.contracted_tensor_shape_after_matmul = [ ( s.chi_left, V_mpo,           s.chi_centre, s.d ),
                                                       ( s.chi_centre, s.chi_left,    V_mpo, s.d        ),
                                                       ( s.chi_centre, V_mpo,         s.chi_centre      ), ]

            s.contracted_tensor_as_matrix_shape_post_contraction = [ ( s.chi_left*V_mpo,           s.chi_centre*s.d ),
                                                                     ( s.chi_centre*s.chi_left,    V_mpo*s.d        ),
                                                                     ( s.chi_centre*V_mpo,         s.chi_centre     ), ]

        elif s.tensor_contractions_tag == 'ltm_opt_update':

            s.contracted_tensor_shape_after_transpose  = [ ( s.chi_left,                    s.chi_left ),
                                                           ( s.chi_left, s.chi_centre,      s.d ),
                                                           ( s.chi_centre,                  s.chi_left, s.d ), ]

            s.contracted_tensor_as_matrix_shape_pre_contraction = [ ( s.chi_left,                    s.chi_left ),
                                                                    ( s.chi_left*s.chi_centre,       s.d ),
                                                                    ( s.chi_centre,                  s.chi_left*s.d ), ]

            s.contracted_tensor_shape_after_matmul = [ ( s.chi_left,                    s.chi_centre, s.d ),
                                                       ( s.chi_left, s.chi_centre,      s.d ),
                                                       ( s.chi_centre,                  s.chi_centre ), ]

            s.contracted_tensor_as_matrix_shape_post_contraction = [ ( s.chi_left,                    s.chi_centre*s.d ),
                                                                     ( s.chi_left*s.chi_centre,       s.d ),
                                                                     ( s.chi_centre,                  s.chi_centre ), ]

        elif s.tensor_contractions_tag == 'rtm_mpo_update':

            V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ]

            s.contracted_tensor_shape_after_transpose  = [ ( s.chi_right, V_mpo,          s.chi_right      ),
                                                           ( s.chi_centre, s.chi_right,   V_mpo, s.d       ),
                                                           ( s.chi_centre, V_mpo,         s.chi_right, s.d ), ]

            s.contracted_tensor_as_matrix_shape_pre_contraction = [ ( s.chi_right*V_mpo,         s.chi_right     ),
                                                                    ( s.chi_centre*s.chi_right,  V_mpo*s.d       ),
                                                                    ( s.chi_centre*V_mpo,        s.chi_right*s.d ), ]

            s.contracted_tensor_shape_after_matmul = [ ( s.chi_right, V_mpo,          s.chi_centre, s.d ),
                                                       ( s.chi_centre, s.chi_right,   V_mpo, s.d        ),
                                                       ( s.chi_centre, V_mpo,         s.chi_centre      ), ]

            s.contracted_tensor_as_matrix_shape_post_contraction = [ ( s.chi_right*V_mpo,         s.chi_centre*s.d ),
                                                                     ( s.chi_centre*s.chi_right,  V_mpo*s.d        ),
                                                                     ( s.chi_centre*V_mpo,        s.chi_centre     ), ]

        elif s.tensor_contractions_tag == 'rtm_opt_update':

            s.contracted_tensor_shape_after_transpose  = [ ( s.chi_right,                    s.chi_right ),
                                                           ( s.chi_right, s.chi_centre,      s.d ),
                                                           ( s.chi_centre,                   s.chi_right, s.d ), ]

            s.contracted_tensor_as_matrix_shape_pre_contraction = [ ( s.chi_right,                    s.chi_right ),
                                                                    ( s.chi_right*s.chi_centre,       s.d ),
                                                                    ( s.chi_centre,                   s.chi_right*s.d ), ]

            s.contracted_tensor_shape_after_matmul = [ ( s.chi_right,                    s.chi_centre, s.d ),
                                                       ( s.chi_right, s.chi_centre,      s.d ),
                                                       ( s.chi_centre,                   s.chi_centre ), ]

            s.contracted_tensor_as_matrix_shape_post_contraction = [ ( s.chi_right,                    s.chi_centre*s.d ),
                                                                     ( s.chi_right*s.chi_centre,       s.d ),
                                                                     ( s.chi_centre,                   s.chi_centre ), ]


    def get_shared_charges_sets_as_self_objects(s):
        """
        Define the following self object:
          - central_charges_sets[ matmul_step ] = charges_sets
        """
        s.row_col_charges_sets = [ ]

        for _ in range( s.number_tensor_contractions[ s.tensor_contractions_tag ] ):

            row_tensor_charges_sets = list( s.contracting_tensors_fused_row_legs_indices_per_charges_set[ _ ].keys( ) )
            col_tensor_charges_sets = list( s.contracting_tensors_fused_col_legs_indices_per_charges_set[ _ ].keys( ) )
            s.row_col_charges_sets.append( s.intersect_charges_sets( row_tensor_charges_sets, col_tensor_charges_sets ) )

        s.central_charges_sets = [ ]

        for _ in range( s.number_tensor_contractions[ s.tensor_contractions_tag ] ):

            cen_tensor_charges_sets = list( s.contracting_tensors_fused_cen_legs_indices_per_charges_set[ _ ].keys( ) )
            row_col_charges_sets = s.row_col_charges_sets[ _ ]
            s.central_charges_sets.append( s.intersect_charges_sets( cen_tensor_charges_sets, row_col_charges_sets ) )


    def get_tensor_blocks_shape(s):
        """
        Define the following self object:
          - contracted_tensor_blocks_shape[ matmul_step ][ charges_set ] = [ row_size, cen_size ]
        """
        s.contracted_tensor_blocks_shape = [ { } for _ in range( s.number_tensor_contractions[ s.tensor_contractions_tag ] ) ]
        s.row_col_tensor_blocks_shape = [ { } for _ in range( s.number_tensor_contractions[ s.tensor_contractions_tag ] ) ]

        for matmul_step in range( s.number_tensor_contractions[ s.tensor_contractions_tag ] ):

            for charges_set in s.central_charges_sets[ matmul_step ]:

                row_ind_charge = s.contracting_tensors_fused_row_legs_indices_per_charges_set[ matmul_step ][ charges_set ]
                cen_ind_charge = s.contracting_tensors_fused_cen_legs_indices_per_charges_set[ matmul_step ][ charges_set ]
                s.contracted_tensor_blocks_shape[ matmul_step ][ charges_set ] = [ len( row_ind_charge ), len( cen_ind_charge ) ]

            for charges_set in s.row_col_charges_sets[ matmul_step ]:

                row_ind_charge = s.contracting_tensors_fused_row_legs_indices_per_charges_set[ matmul_step ][ charges_set ]
                col_ind_charge = s.contracting_tensors_fused_col_legs_indices_per_charges_set[ matmul_step ][ charges_set ]
                s.row_col_tensor_blocks_shape[ matmul_step ][ charges_set ] = [ len( row_ind_charge ), len( col_ind_charge ) ]


    def get_indices_tms_contraction_first_step(s):
        """
        # =====================================================================
        # This function prepares the dictionary of the indices of the non-zero
        # 'abelian' entries of theta as before the matrix-matrix multiplication,
        # after the transposition.
        # =====================================================================
        """
        tensordot_step = 0

        # This is the number of columns of theta (matrix) before the matmul.
        number_cols_two_sites_matrix = s.contracted_tensor_as_matrix_shape_pre_contraction[ tensordot_step ][ 1 ]

        for charges_set in s.central_charges_sets[ tensordot_step ]:
            row_ind_charge = s.contracting_tensors_fused_row_legs_indices_per_charges_set[ tensordot_step ][ charges_set ]
            cen_ind_charge = s.contracting_tensors_fused_cen_legs_indices_per_charges_set[ tensordot_step ][ charges_set ]

            flattened_indices = ( row_ind_charge[:,None ]*number_cols_two_sites_matrix + cen_ind_charge[None,: ] ).flatten( ).astype( npuint64 )
            s.indices_from_dense_flattened_tensor_to_blocks_dict[ tensordot_step ][ charges_set ] = flattened_indices


    def get_contracted_tensor_sparse_dense_map(s):
        """
        Theta is built as a sparse tensor, merging together two neighboring
        MPS sites. This has to be mapped into a dense vector for the Lanczos
        minimization (and v.v.). This index does the job.
        """
        # Map from dense to sparse and viceversa.
        number_cols_two_sites_matrix = s.contracted_tensor_as_matrix_shape_post_contraction[ - 1 ][ 1 ]
        s.indices_map_sparse_and_dense_contracted_tensor = empty(0, dtype=npuint64)

        for charges_set in s.row_col_charges_sets[ - 1 ]:
            row_ind_charge = s.contracting_tensors_fused_row_legs_indices_per_charges_set[ - 1 ][ charges_set ]
            col_ind_charge = s.contracting_tensors_fused_col_legs_indices_per_charges_set[ - 1 ][ charges_set ]

            dummie = ( row_ind_charge[:,None ]*number_cols_two_sites_matrix + col_ind_charge[None,: ] ).flatten( ).astype( npuint64 )

            s.indices_map_sparse_and_dense_contracted_tensor \
                = npappend( s.indices_map_sparse_and_dense_contracted_tensor, dummie )



# =============================================================================
# Abelian svd. Define the elements for the block-diagonal svd.
# =============================================================================
    def get_two_sites_indices_per_charges_set(s):
        """
        Define the following self objects:
          - two_sites_fused_row_legs_indices_per_charges_set[ charges_set ] = nparray
          - two_sites_fused_cen_legs_indices_per_charges_set[ charges_set ] = nparray
          - two_sites_fused_col_legs_indices_per_charges_set[ charges_set ] = nparray
        """

        # =====================================================================
        # Get the indices to load the theta matrix blocks from the matrix theta.
        #
        #   - s.two_sites_fused_row_legs_indices_per_charges_set[ charges_set ] = nparray
        #   - s.two_sites_fused_cen_legs_indices_per_charges_set[ charges_set ] = nparray
        #   - s.two_sites_fused_col_legs_indices_per_charges_set[ charges_set ] = nparray
        #
        # Define the indices per charges sets to load the theta blocks drawing from the sparse matrix theta.

        # We start by identifying those rows and cols of theta that are to be
        # excluded due to the presence of local constraints.
        # The next function returns the indices of such rows and cols.

        # Get the following local object:
        #   - invalid_two_sites_fused_row_legs_indices
        #   - invalid_two_sites_fused_cen_legs_indices
        #   - invalid_two_sites_fused_col_legs_indices
        #
        # For practical purposes, we make them self objects and delete them
        # before the end of the function.
        if 'links_alignment' in s.LIST_SYMMETRIES_NAMES:
            s.invalid_two_sites_fused_row_legs_indices,  \
            s.invalid_two_sites_fused_col_legs_indices = \
                s.add_local_symmetries_two_sites( )

        # We continue by defining the indices per charges set, that serve to
        # identify and extract the blocks of the block diagonal matrix theta.

        # Get the following local objects:
        #
        #   - two_sites_row_indices_per_symmetry_per_charge[ symmetry ][ charge ]
        #   - two_sites_cen_indices_per_symmetry_per_charge[ symmetry ][ charge ]
        #   - two_sites_col_indices_per_symmetry_per_charge[ symmetry ][ charge ]
        #   - two_sites_charges_sets[ ( , , ), ( , , ), ( , , ), ( , , ) ]
        two_sites_row_indices_per_symmetry_per_charge, \
        two_sites_cen_indices_per_symmetry_per_charge, \
        two_sites_col_indices_per_symmetry_per_charge, \
        two_sites_charges_sets = \
            s.get_two_sites_indices_per_symmetry_per_charge( )

        # ... and then create the valid theta indices per charges sets.
        s.two_sites_fused_row_legs_indices_per_charges_set, \
        s.two_sites_fused_cen_legs_indices_per_charges_set, \
        s.two_sites_fused_col_legs_indices_per_charges_set, \
        s.svd_charges_sets = \
            s.build_two_sites_indices_per_charges_set( two_sites_charges_sets, \
                                                           two_sites_row_indices_per_symmetry_per_charge, \
                                                           two_sites_cen_indices_per_symmetry_per_charge, \
                                                           two_sites_col_indices_per_symmetry_per_charge  )

        if 'links_alignment' in s.LIST_SYMMETRIES_NAMES:
            del s.invalid_two_sites_fused_row_legs_indices
            del s.invalid_two_sites_fused_col_legs_indices


    def add_local_symmetries_two_sites(s):
        """
        Theta might have some local constraints that apply to the row legs but
        do not extend to the col legs and vv. Hence, it is about reducing the size
        of the fused legs rather than adding block diagonal structures.

        Return the following local objects:
          - invalid_two_sites_fused_row_legs_indices
          - invalid_two_sites_fused_col_legs_indices
        """
        def local_routine( charges_lists_dict, invalid_two_sites_fused_row_legs_indices, invalid_two_sites_fused_col_legs_indices ):
            """
            Return the following local objects:
              - invalid_two_sites_fused_row_legs_indices = nparray
              - invalid_two_sites_fused_col_legs_indices = nparray
            """
            # theta row and col legs
            two_sites_fused_row_legs_charges,  \
            two_sites_fused_cen_legs_charges,  \
            two_sites_fused_col_legs_charges = \
                s.get_two_sites_fused_legs_charges( charges_lists_dict )

            # Extract the indices that preserve the right- and left-most links alignment.
            valid_two_sites_fused_row_legs_indices = where( two_sites_fused_row_legs_charges == 0 )[ 0 ].astype( npuint64 )
            valid_two_sites_fused_col_legs_indices = where( two_sites_fused_col_legs_charges == 0 )[ 0 ].astype( npuint64 )

            # Obtain its complementary set.
            invalid_two_sites_fused_row_legs_indices = setdiff1d( invalid_two_sites_fused_row_legs_indices, valid_two_sites_fused_row_legs_indices )
            invalid_two_sites_fused_col_legs_indices = setdiff1d( invalid_two_sites_fused_col_legs_indices, valid_two_sites_fused_col_legs_indices )

            return invalid_two_sites_fused_row_legs_indices, invalid_two_sites_fused_col_legs_indices

        charges_lists_dict = { }

        invalid_two_sites_fused_row_legs_indices = arange( s.d*s.chi_left,  dtype=npint64 )
        invalid_two_sites_fused_col_legs_indices = arange( s.d*s.chi_right, dtype=npint64 )

        # alignment of the set of links on the left-hand (right-hand) side on the row (col) legs
        charges_lists_dict['virtual_charges_left'] =   s.virtual_charges_normalized['links_alignment'][  s.lambda_index_left ]
        charges_lists_dict['physical_charges_left'] =  s.physical_charges_normalized['links_set_left'][  s.physical_charges_index_left ]
        charges_lists_dict['virtual_charges_centre'] = s.virtual_charges_normalized['links_alignment'][  s.lambda_index_centre ]
        charges_lists_dict['physical_charges_right'] = s.physical_charges_normalized['links_set_right'][ s.physical_charges_index_right ]
        charges_lists_dict['virtual_charges_right'] =  s.virtual_charges_normalized['links_alignment'][  s.lambda_index_right ]

        invalid_two_sites_fused_row_legs_indices,  \
        invalid_two_sites_fused_col_legs_indices = \
            local_routine( charges_lists_dict,
                           invalid_two_sites_fused_row_legs_indices,
                           invalid_two_sites_fused_col_legs_indices )

        return invalid_two_sites_fused_row_legs_indices, invalid_two_sites_fused_col_legs_indices


    def get_two_sites_indices_per_symmetry_per_charge(s):
        """
        Return the following local objects, for the first matmul step:
          - two_sites_row_indices_per_symmetry_per_charge[ symmetry_tag ][ charge ] = nparray
          - two_sites_cen_indices_per_symmetry_per_charge[ symmetry_tag ][ charge ] = nparray
          - two_sites_col_indices_per_symmetry_per_charge[ symmetry_tag ][ charge ] = nparray
          - two_sites_charges_sets[ ( , , ),  ( , , ),  ( , , ),  ... ]
        """

        def local_routine( symmetry_tag, two_sites_row_indices_per_symmetry_per_charge, \
                                         two_sites_cen_indices_per_symmetry_per_charge, \
                                         two_sites_col_indices_per_symmetry_per_charge, \
                                         two_sites_charges_sets, charges_lists_dict     ):
            """
            Return the following local objects:
              - two_sites_row_indices_per_symmetry_per_charge[ symmetry_tag ][ charge ] = nparray
              - two_sites_cen_indices_per_symmetry_per_charge[ symmetry_tag ][ charge ] = nparray
              - two_sites_col_indices_per_symmetry_per_charge[ symmetry_tag ][ charge ] = nparray
              - two_sites_charges_sets = [ ( , , ),  ( , , ),  ( , , ),  ... ]
            """

            two_sites_row_indices_per_charge, \
            two_sites_cen_indices_per_charge, \
            two_sites_col_indices_per_charge, \
            unique_two_sites_charges = \
                s.get_two_sites_indices_per_charge( charges_lists_dict )

            two_sites_row_indices_per_symmetry_per_charge[ symmetry_tag ] = two_sites_row_indices_per_charge
            two_sites_cen_indices_per_symmetry_per_charge[ symmetry_tag ] = two_sites_cen_indices_per_charge
            two_sites_col_indices_per_symmetry_per_charge[ symmetry_tag ] = two_sites_col_indices_per_charge

            # After identifying the unique charges per symmetry, those can be combined to form
            # all the possible charges set.
            two_sites_charges_sets = s.combine_charges_sets_with_unique_charges( two_sites_charges_sets, unique_two_sites_charges )
            return two_sites_row_indices_per_symmetry_per_charge, two_sites_cen_indices_per_symmetry_per_charge, two_sites_col_indices_per_symmetry_per_charge, two_sites_charges_sets

        two_sites_charges_sets = [( ) ]
        two_sites_row_indices_per_symmetry_per_charge = { }
        two_sites_cen_indices_per_symmetry_per_charge = { }
        two_sites_col_indices_per_symmetry_per_charge = { }

        for symmetry in s.LIST_SYMMETRIES_NAMES:

            charges_lists_dict = { }

            if symmetry == 'links_alignment':

                charges_lists_dict['virtual_charges_left'] =   zeros( s.chi_left, dtype=npint64 )
                charges_lists_dict['physical_charges_left'] =  s.physical_charges_normalized['links_set_right'][ s.physical_charges_index_left ]
                charges_lists_dict['virtual_charges_centre'] =  s.virtual_charges_normalized['links_alignment'][ s.lambda_index_centre ]
                charges_lists_dict['physical_charges_right'] = s.physical_charges_normalized['links_set_left'][ s.physical_charges_index_right ]
                charges_lists_dict['virtual_charges_right'] =  zeros( s.chi_right, dtype=npint64 )
                symmetry_tag = 'alignment_central_links'

                two_sites_row_indices_per_symmetry_per_charge, \
                two_sites_cen_indices_per_symmetry_per_charge, \
                two_sites_col_indices_per_symmetry_per_charge, \
                two_sites_charges_sets = local_routine( symmetry_tag,
                                                        two_sites_row_indices_per_symmetry_per_charge,
                                                        two_sites_cen_indices_per_symmetry_per_charge,
                                                        two_sites_col_indices_per_symmetry_per_charge,
                                                        two_sites_charges_sets, charges_lists_dict )

            else:

                charges_lists_dict['virtual_charges_left'] =    s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
                charges_lists_dict['physical_charges_left'] =  s.physical_charges_normalized[ symmetry ][ s.physical_charges_index_left ]
                charges_lists_dict['virtual_charges_centre'] =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_centre ]
                charges_lists_dict['physical_charges_right'] = s.physical_charges_normalized[ symmetry ][ s.physical_charges_index_right ]
                charges_lists_dict['virtual_charges_right'] =   s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]
                symmetry_tag = symmetry

                two_sites_row_indices_per_symmetry_per_charge, \
                two_sites_cen_indices_per_symmetry_per_charge, \
                two_sites_col_indices_per_symmetry_per_charge, \
                two_sites_charges_sets = \
                    local_routine( symmetry_tag,
                                   two_sites_row_indices_per_symmetry_per_charge,
                                   two_sites_cen_indices_per_symmetry_per_charge,
                                   two_sites_col_indices_per_symmetry_per_charge,
                                   two_sites_charges_sets,
                                   charges_lists_dict )

        return two_sites_row_indices_per_symmetry_per_charge, two_sites_cen_indices_per_symmetry_per_charge, two_sites_col_indices_per_symmetry_per_charge, two_sites_charges_sets


    def build_two_sites_indices_per_charges_set(s, two_sites_charges_sets, two_sites_row_indices_per_symmetry_per_charge, two_sites_cen_indices_per_symmetry_per_charge, two_sites_col_indices_per_symmetry_per_charge ):
        """
        Return the following local objects:
          - two_sites_fused_row_legs_indices_per_charges_set[ charges_set ] = nparray
          - two_sites_fused_cen_legs_indices_per_charges_set[ charges_set ] = nparray
          - two_sites_fused_col_legs_indices_per_charges_set[ charges_set ] = nparray
        """
        two_sites_fused_row_legs_indices_per_charges_set = { }
        two_sites_fused_cen_legs_indices_per_charges_set = { }
        two_sites_fused_col_legs_indices_per_charges_set = { }
        svd_charges_sets = [ ]
        ordered_symmetries_names = list( two_sites_row_indices_per_symmetry_per_charge.keys( ) )

        for charges_set in two_sites_charges_sets:
            row_ind_charges_set = empty(0)
            cen_ind_charges_set = empty(0)
            col_ind_charges_set = empty(0)

            for i in range( len(charges_set) ):
                charge = charges_set[ i ]
                symmetry_tag = ordered_symmetries_names[ i ]
                row_ind_charge = two_sites_row_indices_per_symmetry_per_charge[ symmetry_tag ][ charge ]
                col_ind_charge = two_sites_col_indices_per_symmetry_per_charge[ symmetry_tag ][ charge ]

                try:
                    cen_ind_charge = two_sites_cen_indices_per_symmetry_per_charge[ symmetry_tag ][ charge ]

                except:
                    cen_ind_charge = empty(0)

                if i == 0:
                    row_ind_charges_set = row_ind_charge.copy( )
                    cen_ind_charges_set = cen_ind_charge.copy( )
                    col_ind_charges_set = col_ind_charge.copy( )

                else:
                    row_ind_charges_set = intersect1d( row_ind_charges_set, row_ind_charge )
                    cen_ind_charges_set = intersect1d( cen_ind_charges_set, cen_ind_charge )
                    col_ind_charges_set = intersect1d( col_ind_charges_set, col_ind_charge )

            if len( row_ind_charges_set ) > 0 and len( col_ind_charges_set ) > 0:
                two_sites_fused_row_legs_indices_per_charges_set[ charges_set ] = row_ind_charges_set
                two_sites_fused_cen_legs_indices_per_charges_set[ charges_set ] = cen_ind_charges_set
                two_sites_fused_col_legs_indices_per_charges_set[ charges_set ] = col_ind_charges_set
                svd_charges_sets.append( charges_set )

        return two_sites_fused_row_legs_indices_per_charges_set, two_sites_fused_cen_legs_indices_per_charges_set, two_sites_fused_col_legs_indices_per_charges_set, svd_charges_sets


    def get_two_sites_fused_legs_charges(s, charges_lists_dict):
        """
        Return the following local objects:
          - two_sites_row_indices_per_charge[ charge ]
          - two_sites_cen_indices_per_charge[ charge ]
          - two_sites_col_indices_per_charge[ charge ]
        """
        # ======================= two-sites tensor ========================== #
        #                  ___                            ___                 #
        #          0 -->--|___|-->-- 2            0 -->--|___|-->-- 2         #
        #                   ^                              ^                  #
        #                   |                              |                  #
        #                   1                              1                  #
        #                                                                     #
        # ======================= two-sites tensor ========================== #
        virtual_charges_left =   charges_lists_dict['virtual_charges_left'   ]
        physical_charges_left =  charges_lists_dict['physical_charges_left'  ]
        virtual_charges_centre = charges_lists_dict['virtual_charges_centre']
        physical_charges_right = charges_lists_dict['physical_charges_right']
        virtual_charges_right =  charges_lists_dict['virtual_charges_right'  ]

        # theta row legs
        two_sites_fused_row_legs_charges = s.generate_list_charges([ - virtual_charges_left ,
                                                                     - physical_charges_left ], dtype=npint64 )
        # theta cen legs
        two_sites_fused_cen_legs_charges = s.generate_list_charges([ + virtual_charges_centre ], dtype=npint64 )

        # theta col legs
        two_sites_fused_col_legs_charges = s.generate_list_charges([ - physical_charges_right ,
                                                                     + virtual_charges_right ], dtype=npint64 )

        two_sites_fused_row_legs_charges *= -1 # see WIKI CHARGE CONVENTION

        return two_sites_fused_row_legs_charges, two_sites_fused_cen_legs_charges, two_sites_fused_col_legs_charges


    def get_two_sites_indices_per_charge(s, charges_lists_dict):
        """
        Return the following local objects:
          - two_sites_row_indices_per_charge[ charge ] = nparray
          - two_sites_cen_indices_per_charge[ charge ] = nparray
          - two_sites_col_indices_per_charge[ charge ] = nparray
          - unique_two_sites_charges = nparray
        """
        two_sites_fused_row_legs_charges,  \
        two_sites_fused_cen_legs_charges,  \
        two_sites_fused_col_legs_charges = \
            s.get_two_sites_fused_legs_charges( charges_lists_dict )

        # The unique set of charges in row/cen/col legs.
        unique_two_sites_fused_row_legs_charges = unique( two_sites_fused_row_legs_charges )
        unique_two_sites_fused_cen_legs_charges = unique( two_sites_fused_cen_legs_charges )
        unique_two_sites_fused_col_legs_charges = unique( two_sites_fused_col_legs_charges )

        unique_two_sites_charges = intersect1d( unique_two_sites_fused_row_legs_charges, unique_two_sites_fused_col_legs_charges )

        # For QLM, the overlap of the links sets determines an additional local symmetry
        # that allows to further reduce the size of the fused row leg of theta.
        if 'links_alignment' in s.LIST_SYMMETRIES_NAMES:
            invalid_charge = choice( setdiff1d( arange( min(unique_two_sites_charges) - 1, max(unique_two_sites_charges) + 2 ), unique_two_sites_charges ) )

            # Set off the elements that do not belong in valid_two_sites_fused_row_legs_indices.
            two_sites_fused_row_legs_charges[ s.invalid_two_sites_fused_row_legs_indices ] = invalid_charge
            two_sites_fused_col_legs_charges[ s.invalid_two_sites_fused_col_legs_indices ] = invalid_charge

        # Extract the indices of the col leg with entries equal to charge.
        two_sites_row_indices_per_charge = { }
        two_sites_cen_indices_per_charge = { }
        two_sites_col_indices_per_charge = { }

        for charge in unique_two_sites_charges:
            two_sites_row_indices_per_charge[ charge ] = where( two_sites_fused_row_legs_charges == charge )[ 0 ].astype( npuint64 )
            two_sites_col_indices_per_charge[ charge ] = where( two_sites_fused_col_legs_charges == charge )[ 0 ].astype( npuint64 )

        for charge in unique_two_sites_fused_cen_legs_charges:
            two_sites_cen_indices_per_charge[ charge ] = where( two_sites_fused_cen_legs_charges == charge )[ 0 ].astype( npuint64 )

        return two_sites_row_indices_per_charge, two_sites_cen_indices_per_charge, two_sites_col_indices_per_charge, unique_two_sites_charges






# =============================================================================
# General functions.
# =============================================================================

    # =========================================================================
    # Some functions used for generating objects related to Abelian symmetries.
    def generate_list_charges(s, charge_legs, dtype=npint64):

        charge_index = nparray([ 0 ], dtype=dtype )

        for charge_leg in charge_legs:
            charge_index = (charge_index[:, None ] + charge_leg[None, : ]).flatten( )

        return charge_index

    def intersect_charges_sets(s, charges_sets_A, charges_sets_B):
        """
        # Identify the sets of charges that are shared by both theta and the Hamiltonian MPO.
        """
        shared_charges_sets = [ ]

        for charges_set in charges_sets_A:
            if charges_set in charges_sets_B:
                 shared_charges_sets.append( charges_set )

        return shared_charges_sets




# =============================================================================
# TDB.
# =============================================================================
    def compare_two_virtual_charges_configurations(s, conf1, conf2):
        same = True
        for i in range(3):
            same = same and array_equal(conf1[ i ][ 0 ], conf2[ i ][ 0 ])
            same = same and array_equal(conf1[ i ][ 1 ], conf2[ i ][ 1 ])
        return same


    def give_chain_centered_mixed_canonical_form(s):
        """If the chain is in mixed canonical form but the point X where the left-canonical
        side meets the right-canonical one does not correspond to the centre of the
        chain Y, we can perform iterative two-stes SVDs from X to Y in order to obtain
        the chain in a centered mixed canonical form.
        """

        while s.sweeps_counter < s.NUMBER_SWEEPS:
            s.iteration_time_elapsed = time( )
            # Update the sweep direction and the step index.
            s.update_sweep_direction_and_sweep_index( )
            # Update basic run indices.
            s.update_indices( pivot_index=s.sweep_index )
            # Close and open some elements according to the current pivot position.
            s.open_elements_in_use_for_DMRG( )
            s.close_elements_not_in_use( )
            # Prepare the site tensors to have lambda on the external side.
            s.give_addressed_sites_form_lg_gl( )

            if s.ABELIAN_SYMMETRIES:
                # Define the elements needed to build the dense two-sites tensor.
                s.get_two_sites_indices_per_charges_set( )
                # Map the theta indices per charges sets so that they draw from the dense array theta.
                s.tensor_contractions_tag = 'matvec'
                s.mpo_operator_tag = 'hamiltonian'
                s.get_objects_for_abelian_tensordot( )
                # Get the mapped indices, from sorted vector to matrix sectors.
                s.load_or_map_indices_for_dense_vector_representation( )
                # Define the sectors of the two-sites matrix, shaped for the svd.
                s.get_two_sites_svd_sectors( )
                # Get the sorted two-sites vector, shaped for the Krylov space.
                s.map_two_sites_svd_sectors_to_krylov_vector( )
            else:
                vec = tdot( s.gamma_list[ s.gamma_index_left ], diag( s.lambda_list[ s.lambda_index_centre ] ), axes=(2,0) )
                vec = tdot( vec, s.gamma_list[ s.gamma_index_right ], axes=(2,1) )
                s.dense_flattened_contracted_tensor = transpose( vec, (1,0,2,3) ).flatten( )

            # Update the elements of a two-sites cell.
            s.update_two_sites_and_transfer_matrices( )
            # Some last conclusive operations: if required, store the datafile at each step, display the timers, display the RAM.
            s.update_counters_DMRG( )
            s.conclude_iteration( )
            s.close_one_ltm( s.ltm_index_external )
            s.close_one_rtm( s.rtm_index_external )


    def get_transfer_matrices_list(s, transfer_matrix_list_purpose='', gamma_list_span=0):
        """Build the list of left and right transfer matrices for multiple purposes.
        Currently, this task serves the calculation of expectation values with
        MPOs and the generation of the Hamiltonian TMs for DMRG sweeps.

        Each list counts L-1 elements, the only ones that matter for (i)DMRG methods or to calculate
        physical quantities being those at the mixed canonical form centre.

        Hence, for simplicity, we first define lists of empty tranfer matrices
        and then load them only up to the centre of the mixed canonical form.
        This approach is convenient because it corresponsponds to the case of
        a tm update within a sweep step, hence it can exploit the architecture
        of methods thereby developed.

        Parameters
        ----------
        transfer_matrix_list_purpose : str
            - set to 'iDMRG_step', get the transfer matrices from the outmost site
                until the chain centre;
            - set to 'data_analysis', get the transfer matrices from the left/rightmost
                site until the chain centre, depending on the required span;
            - set to 'DMRG_step', get the transfer matrices from the outmost site
                until the other end of the chain

        """
        # If the gamma_list_span parameter is left as by default, then set it to the entire chain length.
        if gamma_list_span == 0:
            gamma_list_span = s.current_chain_length

        half_gamma_list_span = int( gamma_list_span / 2 )
        ltm_leftmost_pivot_index =  s.current_chain_centre_index - ( half_gamma_list_span - 1 )
        rtm_rightmost_pivot_index = s.current_chain_centre_index + ( half_gamma_list_span - 1 )

        # Set the required length of the transfer matrix list.
        if transfer_matrix_list_purpose in ['data_analysis', 'iDMRG_step']:
            ltm_rightmost_pivot_index = s.current_chain_centre_index
            rtm_leftmost_pivot_index =  s.current_chain_centre_index
        elif transfer_matrix_list_purpose == 'DMRG_sweep':
            ltm_rightmost_pivot_index = s.current_chain_centre_index + ( half_gamma_list_span - 1 )
            rtm_leftmost_pivot_index =  s.current_chain_centre_index - ( half_gamma_list_span - 1 )
        else:
            print('Unknown transfer_matrix_list_purpose specified.')
            sysexit(0)

        s.define_edge_mpo_transfer_matrix_list_left(  ltm_leftmost_pivot_index - 1  )
        s.define_edge_mpo_transfer_matrix_list_right( rtm_rightmost_pivot_index + 1 )
        if s.current_chain_length == 2:
            return

        s.internal_streamline_current_step_tag = 'mpo_transfer_matrix_list_buildup'

        # Create the list of left transfer matrices.
        s.print_only_post_run('Creating list of left transfer matrices for the operators:', s.NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_ALL_SITES)
        for pivot_index in range( ltm_leftmost_pivot_index, ltm_rightmost_pivot_index + 1 ):
            # Extend the left transfer matrix.
            s.add_element_to_left_mpo_transfer_matrix( )
            s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
            s.open_one_gamma( s.gamma_index_left )
            s.update_left_mpo_transfer_matrix( )
            s.close_one_ltm( s.ltm_index_external )
            s.close_one_gamma( s.gamma_index_left )

        # Create the list of right transfer matrices.
        s.print_only_post_run('Creating list of right transfer matrices for the operators:' , s.NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_ALL_SITES )
        for pivot_index in range( rtm_rightmost_pivot_index, rtm_leftmost_pivot_index - 1, -1 ):
            # Extend the right transfer matrix.
            s.add_element_to_right_mpo_transfer_matrix( )
            s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
            s.open_one_gamma( s.gamma_index_right )
            s.update_right_mpo_transfer_matrix( )
            s.close_one_rtm( s.rtm_index_external )
            s.close_one_gamma( s.gamma_index_right )

        s.close_all_ltms( )
        s.close_all_rtms( )


    def get_empty_transfer_matrix_list(s, transfer_matrix_list_length):
        V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ]
        # Create the list of left/right transfer matrices.
        empty_ltm = zeros([ 1, 1, V_mpo ])
        empty_rtm = zeros([ 1, 1, V_mpo ])
        for _ in range( transfer_matrix_list_length - 1 ):
            s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ].append( empty_ltm )
            s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ].insert( 0, empty_rtm )


    def remove_empty_transfer_matrices(s):
        to_remove = int( s.current_chain_length/2 -2 )
        for _ in range(to_remove ):
            del s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ -1 ]
            del s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ][ 0 ]


    def define_edge_mpo_transfer_matrix_list_left(s, leftmost_lambda_index):
        V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ]
        # Create the list of left/right transfer matrices.
        leftmost_lambda = s.lambda_list[ leftmost_lambda_index ]
        leftmost_chi = len( leftmost_lambda )
        leftmost_tm = zeros([V_mpo, leftmost_chi, leftmost_chi ])
        leftmost_tm[ 0 ] = diag( leftmost_lambda )
        leftmost_tm = transpose( leftmost_tm, (1,2,0) )
        #
        s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag ] = [ leftmost_tm ]


    def define_edge_transfer_matrices_list(s, gamma_list_span=0):
        s.define_edge_mpo_transfer_matrix_list_left( gamma_list_span=gamma_list_span )
        s.define_edge_mpo_transfer_matrix_list_right( gamma_list_span=gamma_list_span )


    def define_edge_mpo_transfer_matrix_list_right(s, rightmost_lambda_index):
        V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag ]
        # Create the list of left/right transfer matrices.
        rightmost_lambda = s.lambda_list[ rightmost_lambda_index ]
        rightmost_chi = len( rightmost_lambda )
        rightmost_tm = zeros([V_mpo, rightmost_chi, rightmost_chi ])
        rightmost_tm[ -1 ] = diag( rightmost_lambda )
        rightmost_tm = transpose( rightmost_tm, (1,2,0) )
        #
        s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag ] = [ rightmost_tm ]


    def give_addressed_sites_form_lg_gl(s):
        """
        The addressed sites have both left- or right-canonical form, hence they both have
        lg- or gl-form. The subsequent get_two_sites_gamma_sparse_tensor( ) functions
        requires them to be in lg-gl form.
        """

        if s.sweep_direction == 'R':
            # The gamma tensor to be inserted on the right-hand side of the new pair of central sites.
            gamma_left = s.gamma_list[ s.gamma_index_left ] # 102
            gamma_left = tdot( gamma_left, diag( s.lambda_list[ s.lambda_index_left ]), axes=(1,1) ) # 201
            gamma_left = tdot( gamma_left, diag( s.lambda_list[ s.lambda_index_centre ]**(-1) ), axes=(1,0) ) # 102
            s.gamma_list[ s.gamma_index_left ] = gamma_left

        elif s.sweep_direction == 'L':
            # The gamma tensor to be inserted on the left-hand side of the new pair of central sites.
            gamma_right = s.gamma_list[ s.gamma_index_right ] # 102
            gamma_right = tdot( gamma_right, diag( s.lambda_list[ s.lambda_index_centre ]**(-1) ), axes=(1,1) ) # 201
            gamma_right = tdot( gamma_right, diag( s.lambda_list[ s.lambda_index_right ]), axes=(1,0) ) # 102
            s.gamma_list[ s.gamma_index_right ] = gamma_right


    def get_two_sites_svd_sectors(s):

        gamma_left = tdot( s.gamma_list[ s.gamma_index_left ], diag( s.lambda_list[ s.lambda_index_centre ]), axes=(2,0) ) # 1.0.2
        gamma_left = reshape( transpose( gamma_left, (1,0,2) ), ( s.d*s.chi_left, s.chi_centre ) ) # 01.2

        gamma_right = reshape( transpose( s.gamma_list[ s.gamma_index_right ], (1,0,2) ), (s.chi_centre, s.d*s.chi_right) ) # 0.12

        s.two_sites_svd_sectors = { }

        # Load the sectors of the two central sites, to build the two-sites matrix.
        for charges_set in s.svd_charges_sets:

            row_ind_charge = s.two_sites_fused_row_legs_indices_per_charges_set[ charges_set ]
            cen_ind_charge = s.two_sites_fused_cen_legs_indices_per_charges_set[ charges_set ]
            col_ind_charge = s.two_sites_fused_col_legs_indices_per_charges_set[ charges_set ]

            # Populate the dense block with the corresponding entries of s.two_sites_gamma_dense_matrix.
            if len( cen_ind_charge ) > 0:

                gamma_left_block =  zeros( [ len( row_ind_charge ), len( cen_ind_charge ) ] )
                gamma_right_block = zeros( [ len( cen_ind_charge ), len( col_ind_charge ) ] )

                cython_extract( gamma_left,  gamma_left_block,  row_ind_charge, cen_ind_charge )
                cython_extract( gamma_right, gamma_right_block, cen_ind_charge, col_ind_charge )

                s.two_sites_svd_sectors[ charges_set ] = dot( gamma_left_block, gamma_right_block )

            else:

                s.two_sites_svd_sectors[ charges_set ] = zeros( [ len( row_ind_charge ), len( col_ind_charge ) ] )


    def map_two_sites_svd_sectors_to_krylov_vector(s):

        # Load the two-sites vector, sorted as (01,23) for the svd.
        sorted_two_sites_vector = empty(0)

        for charges_set in s.svd_charges_sets:

            sorted_two_sites_vector = npappend( sorted_two_sites_vector, s.two_sites_svd_sectors[ charges_set ].flatten( ) )

        s.dense_flattened_contracted_tensor = empty(0)

        for charges_set in s.row_col_charges_sets[ - 1 ]:

            sector_rows = len( s.contracting_tensors_fused_row_legs_indices_per_charges_set[ - 1 ][ charges_set ] )
            sector_cols = len( s.contracting_tensors_fused_col_legs_indices_per_charges_set[ - 1 ][ charges_set ] )

            matrix_diagonal_block = zeros( [ sector_rows, sector_cols ] )

            cython_extract_from_vector_to_matrix( sorted_two_sites_vector, matrix_diagonal_block, s.indices_from_svd_vector_to_Krylov_sectors[ charges_set ] )

            s.dense_flattened_contracted_tensor = npappend( s.dense_flattened_contracted_tensor, matrix_diagonal_block.flatten( ) )


    def check_homogeneity_central_pair(s):

        s.update_indices( )
        lambda_left =   s.lambda_list[ s.lambda_index_left ]
        lambda_centre = s.lambda_list[ s.lambda_index_centre ]
        lambda_right =  s.lambda_list[ s.lambda_index_right ]

        chi_left =   len( lambda_left )
        chi_centre = len( lambda_centre )
        chi_right =  len( lambda_right )

        if chi_left != chi_right or chi_left != chi_centre:
            s.print_only_post_run('\n************* Bond dimensions not identical: %d, %d, %d\n' % (chi_left, chi_centre, chi_right) )
            # s.print_only_post_run('\n'.join( str(_) + '  ' + str(len( s.lambda_list[_ ]) ) for _ in range(len( s.lambda_list) ) ) )
            return 'different_bond_dimensions'

        for symmetry in s.LIST_SYMMETRIES_NAMES:
            virtual_left =  s.virtual_charges_normalized[ symmetry ][ s.lambda_index_left ]
            virtual_right = s.virtual_charges_normalized[ symmetry ][ s.lambda_index_right ]

            if not array_equal(virtual_left, virtual_right):
                s.print_only_post_run('\n************* Virtual charges for symmetry %s not identical.' % symmetry )
                s.print_only_post_run( unique(virtual_left, return_counts=True ) )
                s.print_only_post_run( unique(virtual_right, return_counts=True ) )
                s.print_only_post_run( )
                return 'different_virtual_charges'

        if not allclose(lambda_left, lambda_right):
            delta = abs( lambda_left - lambda_right )
            delta_rel = divide( delta, lambda_right )
            s.print_only_post_run('\n************* Lambdas are not close. \t max absolute delta: %.2E \t max relative delta: %.2E\n' % (max(delta), max(delta_rel) ) )
            return 'lambdas_not_allclose'

        return 'allright'


    def get_dense_flattened_contracted_tensor(s):

        if s.tensor_contractions_tag == 'ltm_mpo_update':
            V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag  ]
            ltm_tensor = transpose( s.left_mpo_transfer_matrix_list[ s.mpo_operator_tag  ][ s.ltm_index_external ], (1, 2, 0) )
            s.dense_flattened_contracted_tensor = reshape( ltm_tensor, (V_mpo*s.chi_left, s.chi_left) ).flatten( )

        elif s.tensor_contractions_tag == 'rtm_mpo_update':
            V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag  ]
            rtm_tensor = transpose( s.right_mpo_transfer_matrix_list[ s.mpo_operator_tag  ][ s.rtm_index_external ], (1, 2, 0) )
            s.dense_flattened_contracted_tensor = reshape( rtm_tensor, (V_mpo*s.chi_right, s.chi_right) ).flatten( )

        elif s.tensor_contractions_tag == 'ltm_opt_update':
            ltm_tensor = transpose( s.left_opt_transfer_matrix_list[ s.mpo_operator_tag  ][ s.ltm_index_external ], (1, 0) )
            s.dense_flattened_contracted_tensor = reshape( ltm_tensor, ( s.chi_left, s.chi_left) ).flatten( )

        elif s.tensor_contractions_tag == 'rtm_opt_update':
            rtm_tensor = transpose( s.right_opt_transfer_matrix_list[ s.mpo_operator_tag  ][ s.rtm_index_external ], (1, 0) )
            s.dense_flattened_contracted_tensor = reshape( rtm_tensor, ( s.chi_right, s.chi_right) ).flatten( )


    def map_dense_to_sparse_contracted_tensor(s):

        if s.tensor_contractions_tag in ['ltm_mpo_update', 'rtm_mpo_update']:

            V_mpo = s.mpo_virtual_leg_dimension[ s.mpo_operator_tag  ]

            transfer_matrix_sparse_vector = zeros([ s.chi_centre**2*V_mpo ])
            transfer_matrix_sparse_vector[ s.indices_map_sparse_and_dense_contracted_tensor ] = s.dense_flattened_contracted_tensor
            transfer_matrix_sparse_tensor = reshape( transfer_matrix_sparse_vector, ( s.chi_centre, V_mpo, s.chi_centre ) )
            transfer_matrix_sparse_tensor = transpose( transfer_matrix_sparse_tensor, (0,2,1) )

            return transfer_matrix_sparse_tensor

        elif s.tensor_contractions_tag in ['ltm_opt_update', 'rtm_opt_update']:

            transfer_matrix_sparse_vector = zeros( [ s.chi_centre**2 ] )
            transfer_matrix_sparse_vector[ s.indices_map_sparse_and_dense_contracted_tensor ] = s.dense_flattened_contracted_tensor
            transfer_matrix_sparse_tensor = reshape( transfer_matrix_sparse_vector, ( s.chi_centre, s.chi_centre ) )

            return transfer_matrix_sparse_tensor


    def get_charges_Schmidt_spectrum(s):
        # The charges at the chain centre, sorted according to the Schmidt eigenvalues decreasing order.

        s.update_indices( )

        order = argsort( 10 - s.lambda_list[ s.lambda_index_centre ] )
        first_lambdas = s.lambda_list[ s.lambda_index_centre ][ order ][ :10 ]
        charges_centre_sorted = s.virtual_charges_normalized['n'][ s.lambda_index_centre ][ order ][ :10 ]

        s.print_only_post_run( ['%.3E' %_ for _ in first_lambdas ] )
        s.print_only_post_run( charges_centre_sorted )


    def measure_and_print_gamma_list_weight(s):

        x = 0
        for elem in s.gamma_list:
            x += elem.count_nonzero()
        s.printb('Gamma should weight around %d Mb' % int( x*8/1.E6 ) )


    def display_gamma_type(s):

        for _ in s.gamma_list:
            s.printb( type(_) )

        s.printb()


    def display_tms_type(s):

        s.printb('\tms')

        for _ in s.left_mpo_transfer_matrix_list['hamiltonian']:
            s.printb(type(_) )

        s.printb()

        for _ in s.right_mpo_transfer_matrix_list['hamiltonian']:
            s.printb(type(_) )

        s.printb()


# =============================================================================
#
# =============================================================================
    def get_entanglement_entropy(s):

        s.refresh_current_chain_indices( )

        central_lambda = s.lambda_list[ s.current_chain_centre_index ]
        entanglement_entropy = npsum( - 2*central_lambda**2*log( central_lambda ) )

        return entanglement_entropy


    def get_fidelity_susceptibility(s, other_gamma_list, other_lambda_centre, susceptibility, gamma_list_span=0):

        s.refresh_current_chain_indices( )

        # If the gamma_list_span parameter is left as by default, then set it to the entire chain length.
        if gamma_list_span == 0:
            gamma_list_span = s.current_chain_length

        half_gamma_list_span = int( gamma_list_span / 2 )

        leftmost_gamma_index =  s.current_chain_centre_index - half_gamma_list_span
        rightmost_gamma_index = s.current_chain_centre_index + half_gamma_list_span - 1

        s.open_one_gamma( leftmost_gamma_index )
        other_gamma = conj( other_gamma_list[ leftmost_gamma_index ] )
        other_gamma = s.open_this_gamma( leftmost_gamma_index, other_gamma )

        ltm = tdot( s.gamma_list[ leftmost_gamma_index ], other_gamma, axes=([ 0,1 ],[ 0,1 ]) )
        s.close_one_gamma( leftmost_gamma_index )

        for index in range( leftmost_gamma_index + 1, rightmost_gamma_index ):

            s.open_one_gamma( index )

            other_gamma = conj( other_gamma_list[ index ] )
            other_gamma = s.open_this_gamma( index, other_gamma )

            ltm = tdot( ltm, s.gamma_list[ index ], axes=(0,1) )
            ltm = tdot( ltm, other_gamma, axes=( [ 0,1 ], [ 1,0 ] ) )
            s.close_one_gamma( index )

            if index == s.current_chain_centre_index - 1:

                ltm = tdot( ltm, diag( s.lambda_list[ s.current_chain_centre_index ] ), axes=(0,0) )
                ltm = tdot( ltm, diag( other_lambda_centre ), axes=(0,0) )

        s.open_one_gamma( rightmost_gamma_index )
        other_gamma = conj( other_gamma_list[ rightmost_gamma_index ] )
        other_gamma = s.open_this_gamma( rightmost_gamma_index, other_gamma )

        ltm = tdot( ltm, s.gamma_list[ rightmost_gamma_index ], axes=(0,1) )
        result = asscalar( squeeze( tdot( ltm, other_gamma, axes=( [ 0,1,2 ], [ 1,0,2 ] ) ) ) )

        if result != 0:
            result = - 2 * log( abs( result ) ) / susceptibility**2 / s.current_chain_length
        else:
            s.printb('No overlap at all!'); sysexit( 0 )


        s.close_one_gamma( rightmost_gamma_index )

        return result


    def avg_sum_expectation_value_mpo(s):
        """

        """
        return s.sum_expectation_value_mpo( ) / s.REQUIRED_CHAIN_LENGTH


    def sum_expectation_value_mpo(s, gamma_list_span=0):
        """We might want to calculate the sum of the expectation values of an mpo
        operator calculated over a centered chunck of the chain (even all of it).
        In this case, we load the transfer matrices starting from the edges of
        the chunck of the chain till the central site.

        """
        s.refresh_current_chain_indices( )

        if not s.mpo_operator_tag in s.left_mpo_transfer_matrix_list:
            s.generating_transfer_matrices_list = True
            s.get_transfer_matrices_list( transfer_matrix_list_purpose='data_analysis', gamma_list_span=gamma_list_span )
            del s.generating_transfer_matrices_list

        s.update_indices( gamma_list_span=gamma_list_span )
        s.open_one_ltm( s.ltm_index_internal )
        s.open_one_rtm( s.rtm_index_internal )
        expv = s.calculate_mpo_chain_expectation_value( )

        return expv


    def refresh_current_chain_indices(s):
        """

        """
        s.get_current_chain_length( )
        s.get_current_chain_centre_index( )


    def get_string_or_parity_order(s, **kwargs):
        """

        """
        s.mpo_operator_tag = 'parity_order_leg_' + kwargs['leg']
        s.refresh_current_chain_indices( )
        return s.get_expectation_value_product_onsite_operators(
            gamma_list_span = kwargs['gamma_list_span'],
            border_opt_tag = kwargs['border_opt_tag']
            )

    def get_parity_order(s, **kwargs):
        """The parity order is the expectation value of the parity operator
        applied on all the sites of a certain portion of the chain.

        """
        kwargs['border_opt_tag'] = ''
        return s.get_string_or_parity_order( **kwargs )


    def get_string_order(s, **kwargs):
        """The string order is the expectation value of the parity operator
        applied on all the sites of a certain portion of the chain, plus the
        string operators on the two edges.

        """
        kwargs['border_opt_tag'] = 'string_order_leg_' + kwargs['leg']
        return s.get_string_or_parity_order( **kwargs )


    def get_magnetization_leg(s, **kwargs):
        """

        """
        if kwargs['method'] == 'mpo':
            s.get_magnetization_leg_mpo(leg='0')
        elif kwargs['method'] == 'opt':
            s.get_magnetization_leg_opt(**kwargs)


    def get_magnetization_leg_mpo(s, **kwargs):
        """

        """
        s.refresh_current_chain_indices( )
        s.mpo_operator_tag = 'magnetization_leg_' + kwargs['leg']
        magnetization = s.sum_expectation_value_mpo( )
        return magnetization


    def get_magnetization_leg_opt(s, **kwargs):
        """

        """
        kwargs['operators_name'] = ['magnetization_leg_' + kwargs['leg']]
        return s.get_avg_abs_list_evoo( **kwargs )


    def get_energy(s, **kwargs):
        """

        """
        s.refresh_current_chain_indices( )
        s.mpo_operator_tag = 'hamiltonian'
        energy = s.avg_sum_expectation_value_mpo( )
        print( energy )
        return energy


    def get_list_expvonop(s, **kwargs):
        """

        """
        s.refresh_current_chain_indices( )
        list_evs = s.get_list_expectation_value_operators( **kwargs )
        return list_evs


    def get_avg_abs_list_evoo(s, **kwargs):
        """

        """
        gamma_list_span = kwargs['gamma_list_span']
        list_evs = s.get_list_expvonop(**kwargs)
        abs_list_evs = [ abs(_) for _ in list_evs ]
        avg_abs_list_evs = sum( abs_list_evs ) / gamma_list_span
        return avg_abs_list_evs


    def get_magnetization_rung(s, **kwargs):
        """

        """
        kwargs['operators_name'] = ['magnetization_rung']
        return s.get_avg_abs_list_evoo( **kwargs )


    def get_particle_density_imbalance(s, **kwargs):
        """

        """
        kwargs['operators_name'] = ['particle_density_imbalance']
        return s.get_avg_abs_list_evoo( **kwargs )


    def get_all_horizontal_gauge_field_sz(s, **kwargs):
        N = s.number_legs_ladder
        L = kwargs['gamma_list_span']
        all_horizontal_gauge_field_sz = zeros( [N, L] )
        for leg in range( N ):
            kwargs['operators_name'] = [ 'horizontal_gauge_field_sz_leg_%d' %leg ]
            all_horizontal_gauge_field_sz[leg] = nparray( s.get_list_expvonop( **kwargs ) )
        return all_horizontal_gauge_field_sz


    def get_all_particle_density(s, **kwargs):
        N = s.number_legs_ladder
        L = kwargs['gamma_list_span']
        all_particle_density = zeros( [N, L] )
        for leg in range( N ):
            kwargs['operators_name'] = [ 'particle_density_leg_%d' %leg ]
            all_particle_density[leg] = nparray( s.get_list_expvonop( **kwargs ) )
        return all_particle_density


    def get_all_vertical_gauge_field_sz(s, **kwargs):
        N = s.number_active_rungs
        L = kwargs['gamma_list_span']
        all_vertical_gauge_field_sz = zeros( [N, L] )
        for rung in range( N ):
            kwargs['operators_name'] = [ 'vertical_gauge_field_sz_rung_%d' %rung ]
            all_vertical_gauge_field_sz[rung] = nparray( s.get_list_expvonop( **kwargs ) )
        return all_vertical_gauge_field_sz


    def get_all_flippable_clock(s, **kwargs):
        N = s.number_active_rungs
        L = kwargs['gamma_list_span'] - 1
        all_flippable_clock = zeros( [N, L] )
        for rung in range( N ):
            kwargs['operators_name'] = [ 'flippable_clock_left_rung_%d'  %rung,
                                         'flippable_clock_right_rung_%d' %rung ]
            all_flippable_clock[rung] = nparray( s.get_list_expvonop( **kwargs ) )
        return all_flippable_clock


    def get_all_flippable_anticlock(s, **kwargs):
        N = s.number_active_rungs
        L = kwargs['gamma_list_span'] - 1
        all_flippable_anticlock = zeros( [N, L] )
        for rung in range( N ):
            kwargs['operators_name'] = [ 'flippable_anticlock_left_rung_%d'  %rung,
                                         'flippable_anticlock_right_rung_%d' %rung ]
            all_flippable_anticlock[rung] = nparray( s.get_list_expvonop( **kwargs ) )
        return all_flippable_anticlock


    def get_all_RokhsarKivelson(s, **kwargs):
        N = s.number_active_rungs
        L = kwargs['gamma_list_span'] - 1
        all_RokhsarKivelson = zeros( [N, L] )
        for rung in range( N ):
            kwargs['operators_name'] = [ 'RokhsarKivelson_left_rung_%d'  %rung,
                                         'RokhsarKivelson_right_rung_%d' %rung ]
            all_RokhsarKivelson[rung] = nparray( s.get_list_expvonop( **kwargs ) )
        return all_RokhsarKivelson


    def get_expectation_value_product_onsite_operators(s, gamma_list_span=0, border_opt_tag=''):
        """The mpo_operator_tag is supposed to be defined beforehand.

        """
        # If the gamma_list_span parameter is left as by default, then set it to the entire chain length.
        if gamma_list_span == 0:
            gamma_list_span = s.current_chain_length

        half_gamma_list_span = int( gamma_list_span / 2 )
        leftmost_pivot_index =  s.current_chain_centre_index - ( half_gamma_list_span - 1 )
        rightmost_pivot_index = s.current_chain_centre_index + ( half_gamma_list_span - 1 )

        if border_opt_tag == '':
            ltm = eye( len( s.lambda_list[ leftmost_pivot_index  - 1 ] ) )
            rtm = eye( len( s.lambda_list[ rightmost_pivot_index + 1 ] ) )

        else:
            # The left border.
            pivot_index = leftmost_pivot_index - 1
            s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
            s.open_one_gamma( s.gamma_index_left )
            gamma = s.gamma_list[ s.gamma_index_left ]
            ltm = tdot( s.nparray_operator[ border_opt_tag ][ s.mpo_index_left ], gamma, axes=(1,0) )
            ltm = tdot( ltm, conj( gamma ), axes=([ 0,1 ],[ 0,1 ]) )
            s.close_one_gamma( s.gamma_index_left )
            # The right border.
            pivot_index = rightmost_pivot_index + 1
            s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
            s.open_one_gamma( s.gamma_index_right )
            gamma = s.gamma_list[ s.gamma_index_right ]
            rtm = tdot( s.nparray_operator[ border_opt_tag ][ s.mpo_index_right ], gamma, axes=(1,0) )
            rtm = tdot( rtm, conj( gamma ), axes=([ 0,2 ],[ 0,2 ]) )
            s.close_one_gamma( s.gamma_index_right )

        s.left_opt_transfer_matrix_list[ s.mpo_operator_tag ] =  [ ltm, ltm ]
        s.right_opt_transfer_matrix_list[ s.mpo_operator_tag ] = [ rtm, rtm ]

        s.internal_streamline_current_step_tag = 'product_onsite_opt_transfer_matrix_buildup'

        for pivot_index in range( leftmost_pivot_index, s.current_chain_centre_index + 1 ):
            s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
            s.open_one_gamma( s.gamma_index_left )
            s.left_opt_transfer_matrix_list[ s.mpo_operator_tag  ][ s.ltm_index_external ] = s.left_opt_transfer_matrix_list[ s.mpo_operator_tag ][ s.ltm_index_internal ]
            s.update_left_opt_transfer_matrix( )
            s.close_one_gamma( s.gamma_index_left )

        for pivot_index in range( rightmost_pivot_index, s.current_chain_centre_index - 1, -1 ):
            s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
            s.open_one_gamma( s.gamma_index_right )
            s.right_opt_transfer_matrix_list[ s.mpo_operator_tag ][ s.rtm_index_external ] = s.right_opt_transfer_matrix_list[ s.mpo_operator_tag ][ s.rtm_index_internal ]
            s.update_right_opt_transfer_matrix( )
            s.close_one_gamma( s.gamma_index_right )

        ltm = s.left_opt_transfer_matrix_list[ s.mpo_operator_tag ][ s.ltm_index_internal ]
        l   = diag( s.lambda_list[ s.current_chain_centre_index ] )
        rtm = s.right_opt_transfer_matrix_list[ s.mpo_operator_tag ][ s.rtm_index_internal ]

        expectation_value = s.contract_opt_ltm_l_rtm(ltm, l, rtm)

        del s.left_opt_transfer_matrix_list[ s.mpo_operator_tag ]
        del s.right_opt_transfer_matrix_list[ s.mpo_operator_tag ]

        return expectation_value


    def get_list_expectation_value_operators(s, **kwargs):
        """We might want to obtain the list of expectation values of an operator,
        to perform some operations different from the sum of expectation values.
        To do that, in addition to the operator transfer matrices until
        the central site, we also load the bare transfer matrices from the
        'non-canonical' side to account for the lack of isometry.

        The operator_name is supposed to be defined beforehand.
        """
        gamma_list_span = kwargs['gamma_list_span']
        operator_span = len( kwargs['operators_name'] )

        s.get_current_chain_length( )
        s.get_current_chain_centre_index( )

        if hasattr(s, 'idm_transfer_matrix_list', ):
            if s.ticket_idm_transfer_matrix_list != s.state_ticket:
                s.get_list_identity_transfer_matrix( gamma_list_span=gamma_list_span )
            if not len(s.idm_transfer_matrix_list) == gamma_list_span + 1:
                print('Warning! The list of identity transfer matrices does not have the correct number of elements.'); sysexit(0)
        else:
            s.get_list_identity_transfer_matrix( gamma_list_span=gamma_list_span )

        # If the gamma_list_span parameter is left as by default, then set it to the entire chain length.
        if gamma_list_span == 0:
            gamma_list_span = s.current_chain_length

        half_gamma_list_span = int( gamma_list_span / 2 )
        leftmost_pivot_index = s.current_chain_centre_index - ( half_gamma_list_span - 1 )
        rightmost_pivot_index = s.current_chain_centre_index + ( half_gamma_list_span - 1 )

        list_expectation_values = [ ]

        if operator_span == 1:

            operator_name = kwargs['operators_name'][0]

            # Left part of the chain chunck
            for pivot_index in range( leftmost_pivot_index, s.current_chain_centre_index + 1 ):
                s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
                s.open_one_gamma( s.gamma_index_left )
                gamma = s.gamma_list[ s.gamma_index_left ]
                opt = s.nparray_operator[ operator_name ][ s.mpo_index_left ]
                rtm = s.idm_transfer_matrix_list[ s.rtm_index_internal ]
                expv = s.contract_opt_lg_opt_rtm( gamma, opt, rtm )
                list_expectation_values.append( expv )
                s.close_one_gamma( s.gamma_index_left )

            # Right part of the chain chunck
            for pivot_index in range( s.current_chain_centre_index, rightmost_pivot_index + 1 ):
                s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
                s.open_one_gamma( s.gamma_index_right )
                gamma = s.gamma_list[ s.gamma_index_right ]
                opt = s.nparray_operator[ operator_name ][ s.mpo_index_right ]
                ltm = s.idm_transfer_matrix_list[ s.ltm_index_internal ]
                expv = s.contract_opt_gl_opt_ltm( ltm, opt, gamma )
                list_expectation_values.append( expv )
                s.close_one_gamma( s.gamma_index_right )

        elif operator_span == 2:

            operator_name_left = kwargs['operators_name'][0]
            operator_name_right = kwargs['operators_name'][1]

            # Left part of the chain chunck
            for pivot_index in range( leftmost_pivot_index, s.current_chain_centre_index ):
                s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
                s.open_one_gamma( s.gamma_index_left )
                s.open_one_gamma( s.gamma_index_right )
                gamma_left = s.gamma_list[ s.gamma_index_left ]
                gamma_right = s.gamma_list[ s.gamma_index_right ]
                opt_left = s.nparray_operator[ operator_name_left ][ s.mpo_index_left ]
                opt_right = s.nparray_operator[ operator_name_right ][ s.mpo_index_right ]
                rtm = s.idm_transfer_matrix_list[ s.rtm_index_external ]
                #
                expv = s.contract_opt_lg_lg_opt1_opt2_rtm( gamma_left, gamma_right, opt_left, opt_right, rtm )
                #
                list_expectation_values.append( expv )
                s.close_one_gamma( s.gamma_index_left )
                s.close_one_gamma( s.gamma_index_right )

            # Central sites of the chain chunck
            pivot_index = s.current_chain_centre_index
            s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
            s.open_one_gamma( s.gamma_index_left )
            s.open_one_gamma( s.gamma_index_right )
            gamma_left = s.gamma_list[ s.gamma_index_left ]
            gamma_right = s.gamma_list[ s.gamma_index_right ]
            lambda_schmeigv = diag( s.lambda_list[ s.lambda_index_centre ] )
            opt_left = s.nparray_operator[ operator_name_left ][ s.mpo_index_left ]
            opt_right = s.nparray_operator[ operator_name_right ][ s.mpo_index_right ]
            #
            expv = s.contract_opt_lg_l_gl_opt1_opt2( gamma_left, gamma_right, opt_left, opt_right, lambda_schmeigv )
            #
            list_expectation_values.append( expv )
            s.close_one_gamma( s.gamma_index_left )
            s.close_one_gamma( s.gamma_index_right )

            # Right part of the chain chunck
            for pivot_index in range( s.current_chain_centre_index + 1, rightmost_pivot_index + 1 ):
                s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
                s.open_one_gamma( s.gamma_index_left )
                s.open_one_gamma( s.gamma_index_right )
                gamma_left = s.gamma_list[ s.gamma_index_left ]
                gamma_right = s.gamma_list[ s.gamma_index_right ]
                opt_left = s.nparray_operator[ operator_name_left ][ s.mpo_index_left ]
                opt_right = s.nparray_operator[ operator_name_right ][ s.mpo_index_right ]
                ltm = s.idm_transfer_matrix_list[ s.ltm_index_external ]
                #
                expv = s.contract_opt_gl_gl_opt1_opt2_ltm( gamma_left, gamma_right, opt_left, opt_right, ltm )
                #
                list_expectation_values.append( expv )
                s.close_one_gamma( s.gamma_index_left )
                s.close_one_gamma( s.gamma_index_right )

        else:
            print('Warning! operator_span is neither 1 nor 2.'); sysexit(0)

        return list_expectation_values


    def get_list_identity_transfer_matrix(s, gamma_list_span=0):
        """The list has gamma_list_span + 1 entries.

        Example: if gamma_list_span = 4, len(idm_transfer_matrix_list) = 5.

            -----       -----       -----       -----
            | 1 |       | 2 |       | 3 |       | 4 |
            -----       -----       -----       -----
        --o         --o         o         o--         o--
          |           |         |         |           |
          |           |         |         |           |
          |           |         |         |           |
        --o         --o         o         o--         o--

        The left- and right-most identity matrices are useless but necessary for
        consistency with the indexing convention - see update_indices.

        """

        # If the gamma_list_span parameter is left as by default, then set it to the entire chain length.
        if gamma_list_span == 0:
            gamma_list_span = s.current_chain_length

        half_gamma_list_span = int( gamma_list_span / 2 )

        leftmost_pivot_index =  s.current_chain_centre_index - ( half_gamma_list_span - 1 )
        rightmost_pivot_index = s.current_chain_centre_index + ( half_gamma_list_span - 1 )

        s.idm_transfer_matrix_list = [ diag( s.lambda_list[ s.current_chain_centre_index ]**2 ) ]

        for pivot_index in range( s.current_chain_centre_index, rightmost_pivot_index+1 ):
            s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
            s.open_one_gamma( s.gamma_index_right )
            gamma = s.gamma_list[ s.gamma_index_right ]
            new_tm = tdot( s.idm_transfer_matrix_list[ -1 ], gamma, axes=(1,1) )
            new_tm = tdot( new_tm, conj(gamma), axes=([ 0,1 ],[ 1,0 ]) )
            s.idm_transfer_matrix_list.append( new_tm )
            s.close_one_gamma( s.gamma_index_right )

        for pivot_index in range( s.current_chain_centre_index, leftmost_pivot_index-1, -1 ):
            s.update_indices( pivot_index=pivot_index, gamma_list_span=gamma_list_span )
            s.open_one_gamma( s.gamma_index_left )
            gamma = s.gamma_list[ s.gamma_index_left ]
            new_tm = tdot( s.idm_transfer_matrix_list[ 0 ], gamma, axes=(1,2) )
            new_tm = tdot( new_tm, conj(gamma), axes=([ 0,1 ],[2,0 ]) )
            s.idm_transfer_matrix_list.insert( 0, new_tm )
            s.close_one_gamma( s.gamma_index_left )

        s.ticket_idm_transfer_matrix_list = s.state_ticket


    def get_maps_from_unsorted_matrix_to_sorted_vector(s, matrix_shape, charges_sets, row_sectors_indices, col_sectors_indices):
        """
        The scope of this function is to obtain 4 maps, whose linear combination permits
        to ultimately map the index of an element of an unsorted matrix onto its correspondent
        index in the sorted vector.

        In order to do this we need to identify 4 features of the index:
            A: the origin index of its charge sector
            B: its relative col index on its charge sector
            C: its relative row index on its charge sector
            D: the number of columns of its charge sector

        These can depend on the row/column index or any of the two;
        in the latter case we pick the column, which is usually smaller.

        The complete equation reads:
            A( col_ind ) + B( col_ind ) + C( row_ind ) * D( col_ind )

        which can be simplified as
            [ A + B ]( col_ind ) + C( row_ind ) * D( col_ind )

        Summarizing, here we generate four vectors of length equal to the number of rows or cols of the matrix.
        Of those, A and B can be summed, and by so doing one saves resources.

        Then, given any row/col pair of indices of an element x in the unsorted matrix (or arrays of them),
        all it is to do is to slice out of A+B, C and D the elements in that position, and combined them as
        prescribed above.
        """

        matrix_number_of_rows = matrix_shape[ 0 ]
        matrix_number_of_cols = matrix_shape[ 1 ]

        A_sector_origin_index =       zeros( matrix_number_of_cols, dtype=npuint64 )
        B_col_indices_within_sector = zeros( matrix_number_of_cols, dtype=npuint64 )
        C_row_indices_within_sector = zeros( matrix_number_of_rows, dtype=npuint64 )
        D_sector_number_columns =     zeros( matrix_number_of_cols, dtype=npuint64 )

        dummie = 0

        for charges_set in charges_sets:

            sector_row_indices = row_sectors_indices[ charges_set ]
            sector_col_indices = col_sectors_indices[ charges_set ]

            sector_number_rows = len( sector_row_indices )
            sector_number_cols = len( sector_col_indices )

            # =================================================================
            # Map A.
            A_sector_origin_index[ sector_col_indices ] = dummie
            dummie += sector_number_cols * sector_number_rows

            # =================================================================
            # Map B.
            B_col_indices_within_sector[ sector_col_indices ] = arange( sector_number_cols, dtype=npuint64 )

            # =================================================================
            # Map C.
            C_row_indices_within_sector[ sector_row_indices ] = arange( sector_number_rows, dtype=npuint64 )

            # =================================================================
            # Map D.
            D_sector_number_columns[ sector_col_indices ] = sector_number_cols

        map_AB = A_sector_origin_index + B_col_indices_within_sector

        return map_AB, C_row_indices_within_sector, D_sector_number_columns


    def get_qsi_vectorial_form(s, qsi_matrix_form_row, qsi_matrix_form_col, querying_matrix_shape, tensor_shape, number_tensor_legs, tensor_legs_transposition_order, source_matrix_shape, charges_set, map_AB, map_C, map_D):
        # =====================================================================
        # qsi = querying sector indices
        # qsim = querying sector indices mapped after reshape transpose reshape
        #
        # source tensor: the tensor output of the tensordot, before res-tra-res
        # querying tensor: the tensor after the res-tra-res
        # =====================================================================

        # The following indices refer to the tensor pre-contraction, unsorted, in vectorial form.
        querying_matrix_number_of_columns = querying_matrix_shape[ 1 ]

        qsi_vectorial_form = s.map_indices_from_matrix_to_vectorial_form( qsi_matrix_form_row, qsi_matrix_form_col, querying_matrix_number_of_columns )

        # The indices are mapped onto their relative counterpart in the source tensor, unsorted, in vectorial form.
        qsim_vectorial_form = s.map_indices_through_reshape_transpose_reshape( qsi_vectorial_form, tensor_shape, number_tensor_legs, tensor_legs_transposition_order )

        # The indices are ultimately mapped onto the charge-wise sorted vectorial form.
        qsim_sorted_vectorial_form = s.map_indices_onto_sorted_vector( qsim_vectorial_form, source_matrix_shape, map_AB, map_C, map_D )

        return qsim_sorted_vectorial_form


    def map_indices_onto_sorted_vector(s, indices_vectorial_form, matrix_shape, map_AB, map_C, map_D):
        """

        """

        # Get the row and column indices that identify the charge sector in the unsorted matrix.
        indices_matrix_form_row, indices_matrix_form_col = s.map_indices_from_vector_to_matrix_form( indices_vectorial_form, matrix_shape )

        indices_sorted_vectorial_form = zeros( len( indices_vectorial_form ), dtype=npuint64 )

        # This slicing might be costly. Warning!
        indices_sorted_vectorial_form += map_AB[ indices_matrix_form_col ]
        indices_sorted_vectorial_form += map_C[ indices_matrix_form_row ] * map_D[ indices_matrix_form_col ]

        return indices_sorted_vectorial_form


    def map_indices_from_vector_to_matrix_form(s, indices_vectorial_form, matrix_shape):
        """

        """
        number_of_columns = matrix_shape[ 1 ]
        indices_matrix_form_row, indices_matrix_form_col = npdivmod( indices_vectorial_form, number_of_columns )

        return indices_matrix_form_row, indices_matrix_form_col


    def map_indices_through_reshape_transpose_reshape(s, indices_vectorial_form, tensor_shape, number_tensor_legs, tensor_legs_transposition_order):
        """

        """
        # Define the containers for the quotient and the remainder.
        degeneracies = ones( number_tensor_legs, dtype = npuint64 )
        transposition_order = argsort( tensor_legs_transposition_order )

        # Reshape from tensor to vector.
        for i in range( number_tensor_legs - 2, -1, -1 ):
            degeneracies[ transposition_order[ i ] ] = degeneracies[ transposition_order[ i + 1 ] ] * tensor_shape[ transposition_order[ i + 1 ] ]

        # Reshape from vector to tensor.
        indices_mapped = zeros( len( indices_vectorial_form ), dtype=npuint64 )

        for i in range( number_tensor_legs - 1, -1, -1 ):
            indices_vectorial_form, dummie_remainder = npdivmod( indices_vectorial_form, tensor_shape[ i ] )
            indices_mapped += dummie_remainder*degeneracies[ i ]

        # Overwirte the indices with those mapped by transposition.
        return indices_mapped


    def map_indices_from_matrix_to_vectorial_form(s, indices_matrix_form_row, indices_matrix_form_col, number_of_columns):
        """

        """
        indices_vectorial_form = ( indices_matrix_form_row[ :, None ]*number_of_columns + indices_matrix_form_col[ None, : ] ).flatten( ).astype( npuint64 )

        return indices_vectorial_form


    def get_chain_bond_dimensions(s):
        return [ len(_) for _ in s.lambda_list ]


class H_effective_abelian(object):
    " Class of the effective DMRG hamiltonian "

    def __init__( self, \
                  central_charges_sets, \
                  row_col_charges_sets, \
                  contracted_tensor_blocks_shape, \
                  row_col_tensor_blocks_shape, \
                  indices_from_dense_flattened_tensor_to_blocks_dict, \
                  hltm_blocks, \
                  hmpo_blocks_for_matvec_left, \
                  hmpo_blocks_for_matvec_right, \
                  hrtm_blocks, \
                  shape, \
                  dtype = npfloat64 ):

        self.central_charges_sets = central_charges_sets
        self.row_col_charges_sets = row_col_charges_sets

        self.contracted_tensor_blocks_shape = contracted_tensor_blocks_shape
        self.row_col_tensor_blocks_shape = row_col_tensor_blocks_shape

        self.indices_vector_to_blocks = indices_from_dense_flattened_tensor_to_blocks_dict

        self.hltm_blocks = hltm_blocks
        self.hmpo_blocks_for_matvec_left = hmpo_blocks_for_matvec_left
        self.hmpo_blocks_for_matvec_right = hmpo_blocks_for_matvec_right
        self.hrtm_blocks = hrtm_blocks

        self.shape = shape
        self.dtype = dtype

        self.filter_central_charges( )


    def filter_central_charges(self):

        tdot_step = 0
        for charges_set in copy( self.central_charges_sets[ tdot_step ] ):
            if npall( self.hltm_blocks[ charges_set ] == 0 ):
                self.central_charges_sets[ tdot_step ].remove( charges_set )

        tdot_step = 1
        for charges_set in copy( self.central_charges_sets[ tdot_step ] ):
            if npall( self.hmpo_blocks_for_matvec_left[ charges_set ] == 0 ):
                self.central_charges_sets[ tdot_step ].remove( charges_set )

        tdot_step = 2
        for charges_set in copy( self.central_charges_sets[ tdot_step ] ):
            if npall( self.hmpo_blocks_for_matvec_right[ charges_set ] == 0 ):
                self.central_charges_sets[ tdot_step ].remove( charges_set )

        tdot_step = 3
        for charges_set in copy( self.central_charges_sets[ tdot_step ] ):
            if npall( self.hrtm_blocks[ charges_set ] == 0 ):
                self.central_charges_sets[ tdot_step ].remove( charges_set )


    def matvec(self, vector):
        vector = self.abelian_tensordot( vector, 0, self.hltm_blocks )                  # 1) Contraction theta-ltm.
        vector = self.abelian_tensordot( vector, 1, self.hmpo_blocks_for_matvec_left  ) # 2) Contraction theta-hmpo1.
        vector = self.abelian_tensordot( vector, 2, self.hmpo_blocks_for_matvec_right ) # 3) Contraction theta-hmpo2.
        vector = self.abelian_tensordot( vector, 3, self.hrtm_blocks )                  # 4) Contraction theta-rtm.

        if(self.dtype==npfloat64):
            return vector
        else:
            return vector

    def abelian_tensordot(self, vector, tdot_step, other_blocks_set):
        new_vector = empty(0)

        for charges_set in self.row_col_charges_sets[ tdot_step ]:
            if charges_set not in self.central_charges_sets[ tdot_step ]:
                matrix_diagonal_block = zeros( self.row_col_tensor_blocks_shape[ tdot_step ][ charges_set ] )

            else:
                matrix_diagonal_block = zeros( self.contracted_tensor_blocks_shape[ tdot_step ][ charges_set ] )
                cython_extract_from_vector_to_matrix( vector, \
                                                      matrix_diagonal_block, \
                                                      self.indices_vector_to_blocks[ tdot_step ][ charges_set ] )
                matrix_diagonal_block = dot( matrix_diagonal_block, \
                                             other_blocks_set[ charges_set ] )

            new_vector = npappend( new_vector, matrix_diagonal_block.flatten( ) )

        return new_vector



class H_effective(object):
    def __init__(self, Lp, Rp, M1, M2, dtype=float):
        self.Lp = Lp
        self.Rp = Rp
        self.M1 = M1
        self.M2 = M2
        self.d = M1.shape[3]
        self.chi1 = Lp.shape[0]
        self.chi2 = Rp.shape[0]
        self.shape = nparray( [self.d**2*self.chi1*self.chi2, self.d**2*self.chi1*self.chi2] )
        self.dtype = dtype

    def matvec(self,x):
        x = reshape( x, (self.chi1, self.d, self.d, self.chi2) )
        x = tdot( self.Lp, x, axes=(0,0) )
        x = tdot( x, self.M1, axes=([1,2], [0,3]) )
        x = tdot( x, self.M2, axes=([3,1], [0,3]) )
        x = tdot( x, self.Rp, axes=([1,3], [0,2]) )
        x = x.flatten( )
        return x

"""
Some useful functions, to sort out.

    def stretch_and_copy_row_n_final_copies(s, cell, n):
        zerodummy = zeros( n )
        cell = (cell[:,None] + zerodummy[None,:]).flatten( )
        return cell


    def copy_and_append_row_n_final_copies(s, cell, n):
        zerodummy = zeros( n )
        cell = (cell[None,:] + zerodummy[:,None]).flatten( )
        return cell


    def get_array_idm_ith_position_combination(s, cell, i, N):
        D = len(cell)
        cell = stretch_and_copy_row_n_final_copies(cell, D**(i-1))
        cell = copy_and_append_row_n_final_copies(cell, D**(N-i))
        return cell


    def ias_get_unique_mat_charges(s, cell, i, N):
        num_syms = len(s.LIST_SYMMETRIES_NAMES)
        unique_mat_charges = zeros([num_syms, 2**(num_syms)]) # matrix all unique combinations of populated charges
        for iter_index in range(num_syms):
            cell = Cmat[2*index : 2*index + 2]
            unique_mat_charges[iter_index] = get_array_idm_ith_position_combination(cell, iter_index+1, num_syms)
        return unique_mat_charges


    def ias_get_linear_system_mat(s, cell, i, N):
        num_syms = len(s.LIST_SYMMETRIES_NAMES)
        unique_mat_charges = zeros([2*num_syms, 2**(num_syms)]) # matrix all unique combinations of populated charges
        for iter_index in range(num_syms):
            cell = nparray([1,0])
            local_index = 2*iter_index
            unique_mat_charges[local_index] = get_array_idm_ith_position_combination(cell, local_index, num_syms)
            #
            cell = nparray([0,1])
            local_index = 2*iter_index + 1
            unique_mat_charges[local_index] = get_array_idm_ith_position_combination(cell, local_index, num_syms)
        return unique_mat_charges
"""

