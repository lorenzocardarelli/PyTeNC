# =============================================================================
#
# Then the master is loaded. At the moment, if a simulation is started in
# reduced-ram mode, it can not be switched. This is because the
# master is loaded even before the simulation parameters are passed in, hence
# there can't be a selection. Also, there is no way to distinguish the masters,
# meaning that there should be only one master --> only one RAM_CONSUMPTION
# mode for the first .pkls stored --> same when they are loaded and stored
# again.
#
# =============================================================================

from universal_methods import get_initial_state_filename

from algorithms import DMRG
from hamiltonians import (
                           Spin_Half_Hamiltonian,
                           Spin_One_Hamiltonian,
                           Extended_Bose_Hubbard_Hamiltonian,
                           Spin_Matter_Hamiltonian,
                         )
from glob import glob
from numpy import ndarray
from pickle import dump, load, HIGHEST_PROTOCOL
from scipy.sparse import csr_matrix
from sys import exit as sysexit
from time import time
from warnings import showwarning


class Basic_Loader:

    def __init__(s):
        pass


    @staticmethod
    def check_master_file_existence( SIM_PARAMS ):

        # Check whether the master has been defined already or not.
        MASTER_DATAFILE_PATH = SIM_PARAMS['OUTPUT_PARAMS']['MASTER_DATAFILE_PATH']
        number_master_files = len( glob( MASTER_DATAFILE_PATH ) )

        if number_master_files == 0:
            # If the master does not exist yet, define it.
            print( 'There is no master file. It is being created now.' )
            define_master_file = True

        elif number_master_files == 1:
            # If the master exists, do no create it again.
            define_master_file = False

        elif number_master_files > 1:
            showwarning("Houston, we have a problem. Two masters have been idetified.", category = UserWarning, filename = '', lineno = 0)

        return define_master_file


    def mother_class_initialization(s, hamiltonian_class, algorithm_class, SIM_PARAMS):

        define_master_file = Basic_Loader.check_master_file_existence( SIM_PARAMS )
        initial_state_filename = get_initial_state_filename( SIM_PARAMS )

        # Load the simulation parameters, passed-in from the main script.
        s.initialize_and_update_simulation_parameters( SIM_PARAMS )
        s.initialize_info_string_and_timers( )

        s.SIMULATION_VARIABLES = list(s.__dict__.keys( )).copy( )

        # Load or create the master associated to the specified Hamiltonian.
        s.create_or_load_master( define_master_file, hamiltonian_class )

        # MPS_STATE_VARIABLES is defined after that MASTER_VARIABLES and SIMULATION_VARIABLES have been defined.
        s.MPS_STATE_VARIABLES = [
                                # The following variables must be stored, they contain the data.
                                'lambda_list',
                                'gamma_list',
                                'virtual_charges_normalized',
                                # The following variables must be stored, they contain the info for resuming the run.
                                'centered_mixed_canonical_form',
                                'sweep_direction',
                                'sweep_index',
                                'warmup_step',
                                # The following variables could in principle be retrieved but it's simpler to just store them as well.
                                'chi_left',
                                'chi_right',
                                'current_mixed_canonical_form_centre_index',
                                # This is just to print info before the warmup starts, when reloaded
                                'lambda_index_centre',
                                ]

        # Define the Hamiltonian operators: site_opts, bond_opts, asym_site
        s.define_hamiltonian( )

        # Instantiate an object of the class "algorithm": DMRG, most likely.
        algorithm_class.__init__(s, initial_state_filename)


    def create_or_load_master(s, define_master_file, hamiltonian_class):

        if define_master_file:
            # Store the master file.
            s.printb("Storing the master: ", s.MASTER_DATAFILE_PATH)

            # Instantiate the hamiltonian class.
            hamiltonian_class.__init__(s)

            # Create the master .pkl file and dump the hamiltonian operators onto it.
            s.store_the_master( )

        else:
            # Load an existing master file.
            s.printb('\nLoading the master: ', s.MASTER_DATAFILE_PATH, '\n')

            if s.POST_RUN_INSPECTION:
                print('Loading the master: ', s.MASTER_DATAFILE_PATH)

            with open(s.MASTER_DATAFILE_PATH, 'rb') as my_file:
                master = load(my_file)
                s.MASTER_VARIABLES = master.MASTER_VARIABLES

                for variable_name in s.MASTER_VARIABLES:
                    s.__dict__[variable_name] = master.__dict__[variable_name]


    def store_the_master(s):
        """Define as 'MASTER_VARIABLES' those variables of the self object that
        do not belong in the list 'SIMULATION_VARIABLES'. Essentially, it includes
        those defined in hamiltonian_class.__init__(s).

        """
        s.MASTER_VARIABLES = [x for x in s.__dict__.keys( ) if x not in s.SIMULATION_VARIABLES if x != 'SIMULATION_VARIABLES']

        for key, arg in s.model_operators.items( ):
            if type( s.model_operators[key] ) == list:
                if type( s.model_operators[key][0] ) == ndarray:
                    s.model_operators[key] = [ csr_matrix(_) for _ in s.model_operators[key] ]

        # Function that stores the class object into .pkl file.
        if s.STORE_MASTER:
            with open(s.MASTER_DATAFILE_PATH, 'wb') as my_file:
                dump(s, my_file, HIGHEST_PROTOCOL)


    def set_standard_initial_state_matrix(s, SIM_PARAMS):
        """

        """
        try:
            model = SIM_PARAMS['H_PARAMS']['Gauge_Theory']
            legs = SIM_PARAMS['H_PARAMS']['Number_Legs_Ladder']
            BCs = SIM_PARAMS['H_PARAMS']['Boundary_Conditions']
        except:
            model = SIM_PARAMS['HAMILTONIAN']

        if model == 'Extended_Bose_Hubbard':

            s.INITIAL_STATE_MATRIX = [1, 1]

        elif model == 'quantum_link':

            if legs == 2:

                if BCs == 'OBC':
                    """
                        +     |
                        |     +
                    --> x <-- o -->
                        |     +
                        +     |
                    <-- o --> x <--
                        +     |
                        |     +

                    state   n_0     n_1     sz_left_0   sz_left_1   sz_right_0  sz_right_1  sz_rung_0
                    6       +1      +0      -0.5        +0.5        +0.5        -0.5        +0.5
                    11      +0      +1      +0.5        -0.5        -0.5        +0.5        -0.5
                    """
                    s.INITIAL_STATE_MATRIX = [6, 11]

                elif BCs == 'PBC':
                    """
                    state   n_0     n_1     sz_left_0   sz_left_1   sz_right_0  sz_right_1  sz_rung_0   sz_rung_1
                    11      +1      +0      -0.5        +0.5        +0.5        -0.5        +0.5        -0.5
                    16      +0      +1      +0.5        -0.5        -0.5        +0.5        -0.5        +0.5
                    """
                    s.INITIAL_STATE_MATRIX = [11, 16]

            elif legs == 3:

                if BCs == 'OBC':
                    """
                        |     +
                        +     |
                    <-- o --> x <--
                        +     |
                        |     +
                    --> x <-- o -->
                        |     +
                        +     |
                    <-- o --> x <--
                        +     |
                        |     +

                    state   n_0     n_1     n_2     sz_left_0   sz_left_1   sz_left_2   sz_right_0  sz_right_1  sz_right_2  sz_rung_0   sz_rung_1
                    21      +1      +0      +1      -0.5        +0.5        -0.5        +0.5        -0.5        +0.5        +0.5        -0.5
                    68      +0      +1      +0      +0.5        -0.5        +0.5        -0.5        +0.5        -0.5        -0.5        +0.5
                    """
                    s.INITIAL_STATE_MATRIX = [21, 68]

                elif BCs == 'PBC':
                    """
                        +     |
                        |     +
                    <-- o <-- x <--
                        +     |
                        |     +
                    --> x --> o -->
                        +     |
                        |     +
                    <-- o <-- x <--
                        +     |
                        |     +

                    state   n_0     n_1     n_2     sz_left_0   sz_left_1   sz_left_2   sz_right_0  sz_right_1  sz_right_2  sz_rung_0   sz_rung_1   sz_rung_2
                    54      +1      +0      +1      -0.5        +0.5        -0.5        -0.5        +0.5        -0.5        -0.5        -0.5        -0.5
                    89      +0      +1      +0      -0.5        +0.5        -0.5        -0.5        +0.5        -0.5        +0.5        +0.5        +0.5
                    """
                    s.INITIAL_STATE_MATRIX = [54, 89]

            elif legs == 4:

                if BCs == 'OBC':
                    """
                        +     |
                        |     +
                    --> x <-- o -->
                        |     +
                        +     |
                    <-- o --> x <--
                        +     |
                        |     +
                    --> x <-- o -->
                        |     +
                        +     |
                    <-- o --> x <--
                        +     |
                        |     +

                    state   n_0     n_1     n_2     n_3     sz_left_0   sz_left_1   sz_left_2   sz_left_3   sz_right_0  sz_right_1  sz_right_2  sz_right_3  sz_rung_0   sz_rung_1   sz_rung_2
                    146     +1      +0      +1      +0      -0.5        +0.5        -0.5        +0.5        +0.5        -0.5        +0.5        -0.5        +0.5        -0.5        +0.5
                    321     +0      +1      +0      +1      +0.5        -0.5        +0.5        -0.5        -0.5        +0.5        -0.5        +0.5        -0.5        +0.5        -0.5

                    PERIODIC BOUNDARY CONDITIONS
                    Configuration ferro: [243, 436]
                    0 A 0 A
                    V 0 V 0
                    0 A 0 A
                    Configuration antiferro: [246, 465]
                    V A V A
                    A V A V
                    V A V A
                    """
                    s.INITIAL_STATE_MATRIX = [146, 321]

                elif BCs == 'PBC':
                    """
                        +     |
                        |     +
                    --> x <-- o -->
                        |     +
                        +     |
                    <-- o --> x <--
                        +     |
                        |     +
                    --> x <-- o -->
                        |     +
                        +     |
                    <-- o --> x <--
                        +     |
                        |     +

                    state   n_0     n_1     n_2     n_3     sz_left_0   sz_left_1   sz_left_2   sz_left_3   sz_right_0  sz_right_1  sz_right_2  sz_right_3  sz_rung_0   sz_rung_1   sz_rung_2   sz_rung_2
                    271     +1      +0      +1      +0      -0.5        +0.5        -0.5        +0.5        +0.5        -0.5        +0.5        -0.5        +0.5        -0.5        +0.5        +0.5
                    440     +0      +1      +0      +1      +0.5        -0.5        +0.5        -0.5        -0.5        +0.5        -0.5        +0.5        -0.5        +0.5        -0.5        +0.5
                    """
                    s.INITIAL_STATE_MATRIX = [271, 440]


    def initialize_and_update_simulation_parameters(s, SIM_PARAMS):
        """Define (many) simulation parameters as self objects, from SIM_PARAMS.
        Store the algorithm and Hamiltonian input parameters as self objects.

        """

        # ----------------------------------

        # Initial state parameters.
        if type(SIM_PARAMS['INIT_STATE_PARAMS']) is list:
            s.INITIAL_STATE_MATRIX = SIM_PARAMS['INIT_STATE_PARAMS']

        elif SIM_PARAMS['INIT_STATE_PARAMS'] == 'fixed':
            s.set_standard_initial_state_matrix( SIM_PARAMS )

        elif SIM_PARAMS['INIT_STATE_PARAMS'] == 'random':
            s.INITIAL_STATE_MATRIX = []

        # ----------------------------------

        # Data output, data storing, filenames related parameters.
        OUTPUT_PARAMS = {
            'LOCAL_RUN':               True,
            # This stores the state at every warmup step and every completed sweep step.
            'STORE_STATE':             False,
            # This stores the master file. Convenient for systems with very large Hilbert space.
            'STORE_MASTER':            False,
            # This displays info and stores the state at every iteration of the sweep step.
            'INFO_EVERY_SWEEP_STEP':   True,
            'DISPLAY_RAM':             False,
            'DISPLAY_TIMERS':          False,
            # In the cluster run, this sets a minimum time interval to store the state ( eventually skipping some step/iteration ).
            'PKL_STORE_TIME_INTERVAL':    1,
            # In the cluster run, this sets a minimum time interval to print to file the stdout information.
            'STDOUT_FLUSH_TIME_INTERVAL': 1,
        }
        OUTPUT_PARAMS.update( SIM_PARAMS['OUTPUT_PARAMS'] )
        for key in OUTPUT_PARAMS.keys( ):
            setattr(s, key, OUTPUT_PARAMS[key])

        # ----------------------------------

        # Algorithm parameters.
        ALG_PARAMS = {
            'POST_RUN_INSPECTION': False,
            'INFINITE_SYSTEM_WARMUP': True,
            'REQUIRED_CHAIN_LENGTH': 30,
            'NUMBER_SWEEPS':         2,
            'BOND_DIMENSION':        50,
            'SCHMIDT_TOLERANCE':     10.**(-15),
            # Minimization-related parameters.
            'LANCZOS_ALGORITHM':      'SCIPY', # SCIPY, HOMEMADE
            'SCIPY_EIGSH_TOLERANCE':  0, # 10.**(-15)
            'KRYLOV_SPACE_DIMENSION': 200,
            'ALWAYS_MINIMIZE':        True,
            # Printout-related parameters.
            'SELF_ATTRIBUTES': {
                'rek_value': '%.0E',
                'rek_vector': '%.0E',
            },
            # 'n': '%.3f', also 'hamiltonian' is available
            # 'magnetization_rung': '%s'
            # This, only for the RUN routine.
            'INFOSTREAM_OPERATORS_SUMMED_OVER_ALL_SITES': { },
            'INFOSTREAM_OPERATORS_ACTING_ON_CENTRAL_SITES': [ ],
            # This, only for the ANALYSIS routine.
            'LOCAL_OPERATORS_SUMMED_OVER_ALL_SITES': [],
            'NON_LOCAL_OPERATORS_OR_LIST_LOCAL_EXPECTATION_VALUES': [],
            # This for both.
            'NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_ALL_SITES': [],
            'NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_CENTRAL_SITES': [],
            'NAMES_NORMAL_MATRIX_OPERATORS_FOR_CORRELATIONS_AND_LOCAL_EXPECTATION_VALUES': [],
        }
        ALG_PARAMS.update( SIM_PARAMS['ALG_PARAMS'] )
        for key in ALG_PARAMS.keys( ):
            setattr(s, key, ALG_PARAMS[key])

        # ----------------------------------

        if ALG_PARAMS['INFINITE_SYSTEM_WARMUP']:
            s.INITIAL_STATE_LENGTH = 2 # initial_state_length
        else:
            s.INITIAL_STATE_LENGTH = s.REQUIRED_CHAIN_LENGTH # initial_state_length

        # ----------------------------------

        # Make homogeneous different inputs from RUN and ANALYSIS scripts.
        s.SELF_ATTRIBUTES_TAGS = list( s.SELF_ATTRIBUTES.keys() )

        if s.POST_RUN_INSPECTION:
            s.NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_ALL_SITES = s.LOCAL_OPERATORS_SUMMED_OVER_ALL_SITES
            s.NAMES_NORMAL_MATRIX_OPERATORS_FOR_CORRELATIONS_AND_LOCAL_EXPECTATION_VALUES = s.NON_LOCAL_OPERATORS_OR_LIST_LOCAL_EXPECTATION_VALUES
        else:
            s.INFOSTREAM_OPERATORS_SUMMED_OVER_ALL_SITES['hamiltonian'] = '%.10f'
            #
            s.NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_ALL_SITES = list( s.INFOSTREAM_OPERATORS_SUMMED_OVER_ALL_SITES.keys() )
            s.NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_CENTRAL_SITES = s.INFOSTREAM_OPERATORS_ACTING_ON_CENTRAL_SITES
            # A list for the information stream screen print.
            s.DATA_COLUMNS_TAG = [ ]
            s.DATA_COLUMNS_TAG += s.NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_ALL_SITES
            s.DATA_COLUMNS_TAG += [_ + '_mid' for _ in s.NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_CENTRAL_SITES]
            s.DATA_COLUMNS_TAG += s.SELF_ATTRIBUTES_TAGS

        s.NAMES_ALL_ACTIVE_MATRIX_PRODUCT_OPERATORS = list( set( s.NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_ALL_SITES  ) | set( s.NAMES_MATRIX_PRODUCT_OPERATORS_ACTING_ON_CENTRAL_SITES ) )

        # ----------------------------------

        # Hamiltonian parameters.
        s.H_PARAMS = SIM_PARAMS['H_PARAMS']

        # ----------------------------------

        # Regarding Abelian symmetries.
        if len( s.H_PARAMS['Commuting_Operators'].keys() ) == 0:
            s.ABELIAN_SYMMETRIES = False
        else:
            s.ABELIAN_SYMMETRIES = True
            s.TOTAL_CHARGE = {}
            s.AVERAGE_CHARGE_PER_SITE = {}
            s.LIST_SYMMETRIES_NAMES = list( s.H_PARAMS['Commuting_Operators'].keys() )
            s.LIST_SYMMETRIC_OPERATORS_NAMES = []

            for symmetry_name in s.LIST_SYMMETRIES_NAMES:
                if symmetry_name == 'links_alignment':
                    s.LIST_SYMMETRIC_OPERATORS_NAMES.append( 'links_set_left' )
                    s.AVERAGE_CHARGE_PER_SITE[ 'links_set_left' ] = 0
                    s.LIST_SYMMETRIC_OPERATORS_NAMES.append( 'links_set_right' )
                    s.AVERAGE_CHARGE_PER_SITE[ 'links_set_right' ] = 0

                else:
                    s.LIST_SYMMETRIC_OPERATORS_NAMES.append( symmetry_name )
                    try:
                        s.AVERAGE_CHARGE_PER_SITE[ symmetry_name ] = s.H_PARAMS[ 'Commuting_Operators' ][ symmetry_name ][ 'Average_Charge' ]
                        s.TOTAL_CHARGE[ symmetry_name ] = int( s.AVERAGE_CHARGE_PER_SITE[ symmetry_name ]*s.REQUIRED_CHAIN_LENGTH )
                    except:
                        s.TOTAL_CHARGE[ symmetry_name ] = s.H_PARAMS[ 'Commuting_Operators' ][ symmetry_name ][ 'Total_Charge' ]
                        s.AVERAGE_CHARGE_PER_SITE[ symmetry_name ] = s.TOTAL_CHARGE[ symmetry_name ] / s.REQUIRED_CHAIN_LENGTH

        # Number of tensor contractions, per case.
        s.number_tensor_contractions = {}
        s.number_tensor_contractions['matvec'] = 4
        s.number_tensor_contractions['ltm_mpo_update'] = 3
        s.number_tensor_contractions['rtm_mpo_update'] = 3
        s.number_tensor_contractions['ltm_opt_update'] = 3
        s.number_tensor_contractions['rtm_opt_update'] = 3
        s.number_tensor_contractions['two_sites_svd'] = 1
        s.HALF_REQUIRED_CHAIN_LENGTH = int(s.REQUIRED_CHAIN_LENGTH / 2)


    def initialize_info_string_and_timers(s):

        # These elements have to be initialized here in order to correctly display
        # some initial information, previous to the run.
        s.list_stdout_strings = []
        s.PKL_STORE_TIMER = time()
        s.STDOUT_FLUSH_TIMER = time()



class Spin_Half_DMRG(Basic_Loader, Spin_Half_Hamiltonian, DMRG):
    def __init__(s, SIM_PARAMS):
        # super(Bose_Hubbard_Finite_DMRG, s).__init__() # Why does this only inherit Basic_Loader?? Has to do with MRO..
        Basic_Loader.__init__(s)
        s.mother_class_initialization(
            Spin_Half_Hamiltonian,
            DMRG,
            SIM_PARAMS )


class Spin_One_DMRG(Basic_Loader, Spin_One_Hamiltonian, DMRG):
    def __init__(s, SIM_PARAMS):
        # super(Bose_Hubbard_Finite_DMRG, s).__init__() # Why does this only inherit Basic_Loader?? Has to do with MRO..
        Basic_Loader.__init__(s)
        s.mother_class_initialization(
            Spin_One_Hamiltonian,
            DMRG,
            SIM_PARAMS )


class Extended_Bose_Hubbard_DMRG(Basic_Loader, Extended_Bose_Hubbard_Hamiltonian, DMRG):
    def __init__(s, SIM_PARAMS):
        # super(Bose_Hubbard_Finite_DMRG, s).__init__() # Why does this only inherit Basic_Loader?? Has to do with MRO..
        Basic_Loader.__init__(s)
        s.mother_class_initialization(
            Extended_Bose_Hubbard_Hamiltonian,
            DMRG,
            SIM_PARAMS )


class Spin_Matter_DMRG(Basic_Loader, Spin_Matter_Hamiltonian, DMRG):
    def __init__(s, SIM_PARAMS):
        # super(Bose_Hubbard_Finite_DMRG, s).__init__() # Why does this only inherit Basic_Loader?? Has to do with MRO..
        Basic_Loader.__init__(s) # Some the simulation parameters can/must always be reloaded.
        s.mother_class_initialization(
            Spin_Matter_Hamiltonian,
            DMRG,
            SIM_PARAMS )



