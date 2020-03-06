from glob import glob
from numpy import array as nparray
from numpy import where
from numpy.random import randint
from os import environ
from os import path
from os import sep as ossep
from os.path import normpath
from re import findall
from re import sub
from sys import argv
from sys import exit as sysexit


def get_args( ):
    """

    """
    args = []
    for i in range(1, len(argv)):
        args.append(argv[i])
    return args


def get_DATA_SUBDIRECTORY_NAME( ):
    """

    """
    args = get_args( )

    try:
        i = args.index('-subfolder') + 1
        DATA_SUBDIRECTORY_NAME = polish_directory_path(str(args[i]))

    except ValueError:
        DATA_SUBDIRECTORY_NAME = ''
        print('Warning! DATA_SUBDIRECTORY_NAME was not specified.')

    return DATA_SUBDIRECTORY_NAME


def retrieve_bcs_from_datafoldername( ):
    """

    """
    DATA_SUBDIRECTORY_NAME = get_DATA_SUBDIRECTORY_NAME( )
    if 'OBC' in DATA_SUBDIRECTORY_NAME:
        return 'OBC'
    elif 'PBC' in DATA_SUBDIRECTORY_NAME:
        return 'PBC'
    else:
        return ''


def retrieve_parameters_from_datafoldername(DATA_SUBDIRECTORY_NAME, key, numeric_format):
    """

    """
    if DATA_SUBDIRECTORY_NAME == '':
        return 0

    DATA_SUBDIRECTORY_NAME = normpath(DATA_SUBDIRECTORY_NAME).split(ossep)[-1]
    search_key = '_' + key + numeric_format
    parameter = findall( search_key, DATA_SUBDIRECTORY_NAME )[0]
    parameter = findall( numeric_format, parameter )[0]
    if '.' in numeric_format:
        parameter = float(parameter)
    else:
        parameter = int(parameter)
    return parameter


def get_PLOT_DIRECTORY_PATH( ):
    """Get the path to the plots directory.

    """
    if environ['HOME'] == '/home/itp/lcarda':
        PLOT_DIRECTORY_PATH = '/bigwork/lcarda/plots/'
    elif environ['HOME'] == '/home/lorenzo':
        PLOT_DIRECTORY_PATH = '/media/sf_lcarda/bigwork/plots/'
    elif environ['HOME'] == '/home/nhbbcard':
        PLOT_DIRECTORY_PATH = '/bigwork/nhbbcard/plots/'

    return PLOT_DIRECTORY_PATH


def get_DATA_ROOT_DIRECTORY_PATH( ):
    """Get the path to the data directory.

    """
    if environ['HOME'] == '/home/itp/lcarda':
        DATA_ROOT_DIRECTORY_PATH = '/bigwork/lcarda/data/finite/'
    elif environ['HOME'] == '/home/lorenzo':
        DATA_ROOT_DIRECTORY_PATH = '/media/sf_lcarda/bigwork/data/finite/'
    elif environ['HOME'] == '/home/nhbbcard':
        DATA_ROOT_DIRECTORY_PATH = '/bigwork/nhbbcard/data/finite/'

    return DATA_ROOT_DIRECTORY_PATH


def get_SOURCE_DIRECTORY_PATH( ):
    """Get the path to the source directory.

    """
    if environ['HOME'] == '/home/itp/lcarda':
        SOURCE_DIRECTORY_PATH = '/bigwork/lcarda/TensorNetworks/code/'
    elif environ['HOME'] == '/home/lorenzo':
        SOURCE_DIRECTORY_PATH = '/media/sf_lcarda/bigwork/TensorNetworks/code/'
    elif environ['HOME'] == '/home/nhbbcard':
        SOURCE_DIRECTORY_PATH = '/bigwork/nhbbcard/TensorNetworks/code/'

    return SOURCE_DIRECTORY_PATH


def load_initialize( SIM_PARAMS ):
    """This is equivalent to: Bose_Hubbard_DMRG.__init__(SIM_PARAMS)

    """
    return getattr( __import__("mother_classes"), SIM_PARAMS['HAMILTONIAN'] + "_" + SIM_PARAMS['ALGORITHM'] ) \
            ( SIM_PARAMS )


def get_MPS_datafile_attributes(H_PARAMS, ALG_PARAMS):
    """

    """
    STATE_DATAFILE_ATTRIBUTES = ''
    L = ALG_PARAMS[ 'REQUIRED_CHAIN_LENGTH' ]
    sorted_dict_keys = sorted( H_PARAMS.keys( ) )

    for key in sorted_dict_keys:
        arg = H_PARAMS[ key ]

        if key == 'Commuting_Operators':
            for Operator_Name in H_PARAMS[ 'Commuting_Operators' ].keys( ):
                try:
                    string = Operator_Name + str( '%.2f' %H_PARAMS[ 'Commuting_Operators' ][ Operator_Name ][ 'Average_Charge' ] )
                    STATE_DATAFILE_ATTRIBUTES += '_' + string
                except:
                    pass
        else:
            if key in ['jy', 'V']:
                STATE_DATAFILE_ATTRIBUTES += '_' + key + str( '%.6f'%arg )
            elif key in ['Gauge_Theory', 'Number_Legs_Ladder']:
                pass
            else:
                try:
                    STATE_DATAFILE_ATTRIBUTES += '_' + key + str( '%.2f'%arg )
                except:
                    STATE_DATAFILE_ATTRIBUTES += '_' + str( arg )

    STATE_DATAFILE_ATTRIBUTES += '_' + 'L%d' %L

    # Add the bond dimension as attribute to the state filename for the RUN routine.
    # No problem in case of a POST_RUN_ANALYSIS; if not needed, it is discarded/overwritten.
    STATE_DATAFILE_ATTRIBUTES += '_' + 'chi%d' %ALG_PARAMS['BOND_DIMENSION']

    try:
        MASTER_FILE_ATTRIBUTES = '_' + H_PARAMS[ 'Boundary_Conditions' ]
    except:
        MASTER_FILE_ATTRIBUTES = ''

    return STATE_DATAFILE_ATTRIBUTES, MASTER_FILE_ATTRIBUTES


def polish_directory_path(string):
    """Add a slash '/' at the end, if not provided.

    """
    if string[-1] != '/':
        string += '/'
    return string


def update_from_command_line_inputs( ALG_PARAMS, OUTPUT_PARAMS, H_PARAMS ):
    """Parse command line inputs and assign them to their relative variables.

    """
    INDEX = randint( 10000, 100000 )

    args = [ ]
    for i in range( 1,len( argv ) ):
        args.append( argv[ i ] )

    try:
        i = args.index('-bc') + 1
        H_PARAMS['Boundary_Conditions'] = str( args[ i ] )
    except ValueError:
        pass
    # =================
    try:
        i = args.index('-m') + 1
        H_PARAMS['m'] = float( args[ i ] )
    except ValueError:
        pass
    # =================
    try:
        i = args.index('-jy') + 1
        H_PARAMS[ 'jy' ] = float( args[ i ] )
    except ValueError:
        pass
    #=================
    try:
        i = args.index('-t') + 1
        H_PARAMS['t'] = float(args[i])
    except ValueError:
        pass
    #=================
    try:
        i = args.index('-k') + 1
        H_PARAMS['k'] = float(args[i])
    except ValueError:
        pass
    #=================
    try:
        i = args.index('-U') + 1
        H_PARAMS['U'] = float(args[i])
    except ValueError:
        pass
    #=================
    try:
        i = args.index('-V') + 1
        H_PARAMS['V'] = float(args[i])
    except ValueError:
        pass
    #=================
    try:
        i = args.index('-N') + 1
        H_PARAMS['N'] = float(args[i])
    except ValueError:
        pass
    #=================
    try:
        i = args.index('-D') + 1
        H_PARAMS['D'] = float(args[i])
    except ValueError:
        pass
    # ====================================================================
    try:
        i = args.index('-length') + 1
        ALG_PARAMS['REQUIRED_CHAIN_LENGTH'] = int( args[ i ] )
    except ValueError:
        pass
    # =================
    try:
        i = args.index('-sweeps') + 1
        ALG_PARAMS[ 'NUMBER_SWEEPS' ] = int( args[ i ] )
    except ValueError:
        pass
    # =================
    try:
        i = args.index('-chi') + 1
        ALG_PARAMS[ 'BOND_DIMENSION' ] = int( args[ i ] )
    except ValueError:
        pass
    # ====================================================================
    try:
        i = args.index('-eigshtol') + 1
        ALG_PARAMS[ 'SCIPY_EIGSH_TOLERANCE' ] = int( args[ i ] )
    except ValueError:
        pass
    # =================
    try:
        i = args.index('-krylov') + 1
        ALG_PARAMS[ 'KRYLOV_SPACE_DIMENSION' ] = int( args[ i ] )
    except ValueError:
        pass
    # =================
    try:
        i = args.index('-alwayslanczos') + 1
        ALG_PARAMS[ 'ALWAYS_MINIMIZE' ] = bool( int( args[ i ] ) )
    except ValueError:
        pass
    # ====================================================================
    try:
        i = args.index('-loc') + 1
        OUTPUT_PARAMS[ 'LOCAL_RUN' ] = bool( int( args[ i ] ) )
    except ValueError:
        pass
    # =================
    try:
        i = args.index('-store') + 1
        OUTPUT_PARAMS[ 'STORE_STATE' ] = bool( int( args[ i ] ) )
    except ValueError:
        pass
    # =================
    try:
        i = args.index('-storesweep') + 1
        OUTPUT_PARAMS[ 'INFO_EVERY_SWEEP_STEP' ] = bool( int( args[ i ] ) )
    except ValueError:
        pass
    # =================
    try:
        i = args.index('-displayram') + 1
        OUTPUT_PARAMS[ 'DISPLAY_RAM' ] = bool( int( args[ i ] ) )
    except ValueError:
        pass
    # =================
    try:
        i = args.index('-displaytimer') + 1
        OUTPUT_PARAMS[ 'DISPLAY_TIMERS' ] = bool( int( args[ i ] ) )
    except ValueError:
        pass
    # =================
    try:
        i = args.index('-psi') + 1
        OUTPUT_PARAMS[ 'PKL_STORE_TIME_INTERVAL' ] = int( args[ i ] )
    except ValueError:
        pass
    # =================
    try:
        i = args.index('-sfi') + 1
        OUTPUT_PARAMS[ 'STDOUT_FLUSH_TIME_INTERVAL' ] = int( args[ i ] )
    except ValueError:
        pass
    # ====================================================================
    try:
        i = args.index('-datasub') + 1
        DATA_SUBDIRECTOY_PATH = polish_directory_path(str( args[ i ] ))
    except ValueError:
        DATA_SUBDIRECTOY_PATH = ''
        pass
    # =================
    try:
        i = args.index('-ind') + 1
        INDEX = str( args[ i ] )
    except ValueError:
        pass
    # =================
    CORES_PER_NODE = 1
    try:
        i = args.index('-cpn') + 1
        CORES_PER_NODE = int( args[ i ] )
    except ValueError:
        pass

    # Filename attributes.
    STATE_DATAFILE_ATTRIBUTES, MASTER_FILE_ATTRIBUTES = get_MPS_datafile_attributes( H_PARAMS, ALG_PARAMS )

    # =========================================================================
    # Define automatically the destination folder for data files.
    # =========================================================================
    DATA_ROOT_DIRECTORY_PATH = get_DATA_ROOT_DIRECTORY_PATH( )

    # File name used for stdout files and .pkl data files.
    SCRIPT_NAME = sub( '.py', '', argv[0] )
    OUTPUT_PARAMS['MASTER_DATAFILE_PATH'] = DATA_ROOT_DIRECTORY_PATH + 'master_' + SCRIPT_NAME + MASTER_FILE_ATTRIBUTES + '.pkl'
    OUTPUT_PARAMS['STATE_DATAFILE_PATH'] =  DATA_ROOT_DIRECTORY_PATH + DATA_SUBDIRECTOY_PATH + '' + SCRIPT_NAME + STATE_DATAFILE_ATTRIBUTES + '_' + str( INDEX )

    return ALG_PARAMS, OUTPUT_PARAMS, H_PARAMS, CORES_PER_NODE


def remove_dummy_index_from_filename_and_add_pkl(FILENAME):
    # This might as well be \d*, in case there are no other substrings in the form ' undercore + digits '
    FILENAME = sub( '_\d\d\d\d\d', '', FILENAME ) + '_*.pkl'
    return FILENAME


def get_initial_state_filename(SIM_PARAMS):
    """ Search possible initial states to load. If any is found, assign its
    filename to the variable 'initial_state_filename'.

    """
    FILENAME = remove_dummy_index_from_filename_and_add_pkl( SIM_PARAMS['OUTPUT_PARAMS']['STATE_DATAFILE_PATH'] )
    print('Globbing ', FILENAME)

    try:
        initial_state_filename = sorted( glob( FILENAME ), key=path.getmtime )[-1]
        print( 'An MPS wave function with the required parameters was found. It wil be loaded and used as initial ansatz.\n%s\n' %initial_state_filename )

    except:

        if SIM_PARAMS['ALG_PARAMS']['POST_RUN_INSPECTION']:
            print('The required MPS state datafile does not exist. Return an empty initial_state_filename.')
            initial_state_filename = ''

        else:
            FILENAME = sub( '_L\d*', '*', FILENAME )
            print( '\nNo MPS ansatz with the required parameters and length was stored. We look for an existing state of shorter length.\n%s\n' %FILENAME )

            try:
                sorted_filenames = sorted( glob( FILENAME ), key=path.getmtime )
                lengths = nparray( [ int( findall( r'_L\d*', filename )[0][2:] ) for filename in sorted_filenames ] )
                L = SIM_PARAMS['ALG_PARAMS']['REQUIRED_CHAIN_LENGTH']
                print('These are all the filenames found:')

                for filename in sorted_filenames:
                    print(filename)

                # Look for the lastest filename with chain length equal to the required chain length.
                found = False
                mask = lengths == L
                wh = where(mask)[0]
                if len(wh) > 0:
                    pos = wh[-1]
                    found = True

                # If no file with the required length was found, look for one with shorter chain.
                if not found:
                    mask = lengths < L
                    wh = where(mask)[0]
                    if len(wh) > 0:
                        pos = wh[-1]
                        found = True

                # If a file was found, load it.
                if found:
                    initial_state_filename = sorted_filenames[pos]
                    print( 'An MPS wave function with the required parameters is stored in the file:\n\t\"%s\".\nThe state will be loaded and used as initial ansatz.\n' %initial_state_filename )

                else:
                    initial_state_filename = ''
                    print( '\nNo MPS ansatz with the required parameters was stored yet. Create a new trivial MPS wave function ansatz,' )

            except IndexError:
                initial_state_filename = ''
                print( 'There was a problem with loading the datafiles. Anyway, no MPS ansatz with the required parameters was loaded. Create a new trivial MPS wave function ansatz,' )

    return initial_state_filename




