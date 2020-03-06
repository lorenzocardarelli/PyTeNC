import socket
from universal_methods import (
                                load_initialize,
                                update_from_command_line_inputs,
                                get_SOURCE_DIRECTORY_PATH,
                              )
source_directory_path = get_SOURCE_DIRECTORY_PATH( )

from subprocess import call
call( "python " + source_directory_path + "setup.py build_ext --inplace", shell=True )


# =============================================================================
# Define the simulation parameter.
# =============================================================================
# Parameters to initialize the MPS: 'fixed', a list of two [12, 3] or 'random'.
INIT_STATE_PARAMS = 'fixed'

# Hamiltonian coefficients.
H_PARAMS = {
            'm': 0,
            'jy': 1.11,
            'k': 0,
            'Gauge_Theory': 'quantum_link',  # 'quantum_link', 'spin_ice'
            'Number_Legs_Ladder': 3,
            'Boundary_Conditions': 'PBC', # 'OBC', 'PBC', 'TORUS' only with REQUIRED_CHAIN_LENGTH: 4
            'Commuting_Operators': {
                'n': {'Average_Charge': 1.5},
                'links_alignment': None,
                },
            }

# Algorithm parameters.
ALG_PARAMS={
            'POST_RUN_INSPECTION': False,
            #
            'REQUIRED_CHAIN_LENGTH': 10,
            'NUMBER_SWEEPS':         0,
            'BOND_DIMENSION':        40,
            'INFOSTREAM_OPERATORS_SUMMED_OVER_ALL_SITES': { }, #{'n': '%.3f'},
            'INFOSTREAM_OPERATORS_ACTING_ON_CENTRAL_SITES': [ ], #['magnetization_rung']
            }

# Printout / data storing parameters.
OUTPUT_PARAMS= {
                'LOCAL_RUN':               True,
                'STORE_STATE':             False,
                'STORE_MASTER':            True,
                'INFO_EVERY_SWEEP_STEP':   True,
                }


# =============================================================================
# Read command-line inputs and update parameters accordingly.
# =============================================================================
ALG_PARAMS, OUTPUT_PARAMS, H_PARAMS, CORES_PER_NODE = \
    update_from_command_line_inputs( ALG_PARAMS, OUTPUT_PARAMS, H_PARAMS )


# Simulation parameters.
SIM_PARAMS = {
                # Algorithm parameters.
                'ALGORITHM': 'DMRG',
                'ALG_PARAMS': ALG_PARAMS,
                # Hamiltonian-related parameters.
                'HAMILTONIAN': 'Spin_Matter',
                'H_PARAMS': H_PARAMS,
                # Initial state parameters.
                'INIT_STATE_PARAMS': INIT_STATE_PARAMS,
                # Data output parameters.
                'OUTPUT_PARAMS': OUTPUT_PARAMS,
                }


# =============================================================================
# Try to load initial state, otherwise initiate a new MPS.
# =============================================================================
MPS = load_initialize( SIM_PARAMS )


# =============================================================================
# Print initial state and some initial information.
# =============================================================================
MPS.printb( 'Number of cores used: ' + str( CORES_PER_NODE ) )
MPS.printb( 'Job running on: '       + str( socket.gethostname( ) ) )
MPS.printb( 'Physical dimension: '   + str( MPS.d ) )
for key in H_PARAMS:   MPS.printb( key, '\t', H_PARAMS[ key ] )
for key in ALG_PARAMS: MPS.printb( key, '\t', ALG_PARAMS[ key ] )
if len( MPS.gamma_list ) == 2:
    MPS.printb( 'INITIAL_STATE_MATRIX', '\t', MPS.INITIAL_STATE_MATRIX )


# =============================================================================
# Run.
# =============================================================================
MPS.run( )

