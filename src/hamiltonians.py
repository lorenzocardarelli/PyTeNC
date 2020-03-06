from warnings import warn
from sys import exit as sysexit
from numpy import float as npfloat
from numpy import dot

from lattices import (
                       Chain,
                       Bipartite_Chain,
                       Ladder,
                       Triangular_Ladder,
                       Bipartite_Ladder
                      )

from models import (
                     Spin_Model,
                     Ising_Pauli_Model,
                     Bose_Hubbard_Model,
                     Spin_Matter_Model,
                    )


def warn_the_user(parameters_to_set_in_H_PARAMS, H_PARAMS):
    hamiltonian_keys = list( H_PARAMS.keys( ) )
    try:
        hamiltonian_keys.remove('Commuting_Operators')
    except:
        pass
    if not sorted(hamiltonian_keys)==sorted(parameters_to_set_in_H_PARAMS):
        warn("\n\nIf you read this message, the required parameters in H_PARAMS have not been correctly set."+
             "\nPlease, specify all and only the following parameters: "+
             str(parameters_to_set_in_H_PARAMS)+
             "\nFor further information, please find in hamiltonian.py the class that you specified in SIM_PARAMS['HAMILTONIAN'].",
             UserWarning)
        sysexit(0)



class Hamiltonian:

    def __init__(s):
        s.H_dtype = npfloat # ?
        s.site_opts = []
        s.two_sites_operators = []
        s.asym_site = []

    #==========================================================================
    # Build Hamiltonian terms
    #==========================================================================
    # opt_span contains 2 instructions:
    # - the first flags whether it has to be a 2- or a 3-sites thing
    # - the second says which element of the H_mpo_cell block it belongs to
    def make_onsite_operators(s, opt_span, c, o1):
        if c!=0:
            s.site_opts.append( (opt_span, c*o1.toarray( ) ) )


    def make_asym_site(s, asym_pos, c, asym_op):
        if c!=0:
            s.asym_site.append( (asym_pos, c*asym_op.toarray( ) ) )


    def make_two_sites_operators(s, opt_span, c, o1, o2):
        if c!=0:
            s.two_sites_operators.append( (opt_span, o1.toarray( ), c*o2.toarray( ) ) )



class Extended_Bose_Hubbard_Hamiltonian(Hamiltonian, Chain, Bose_Hubbard_Model):
    def __init__(s):
        #======================================================================
        # Initiate the Hamiltonian features.
        #======================================================================
        parameters_to_set_in_H_PARAMS = ['t', 'U', 'V', 'N'] # Lists all the variables that must be specified in H_PARAMS for this Hamiltonian.
        warn_the_user(parameters_to_set_in_H_PARAMS, s.H_PARAMS)
        #======================================================================
        # Define the lattice  --->  for key in s.lattice_keys: printa(key)
        #======================================================================
        Chain.__init__(s)
        #======================================================================
        # Define the model  --->  for key in s.model_operators: printa(key)
        #======================================================================
        Bose_Hubbard_Model.__init__(s, s.H_PARAMS['N'])
        #======================================================================
        # Initiate the Hamiltonian features
        #======================================================================
        Hamiltonian.__init__(s)

    #==========================================================================
    # Define the Hamiltonian and its operators
    #==========================================================================
    def define_hamiltonian(s):

        t = s.H_PARAMS['t']
        U = s.H_PARAMS['U']
        V = s.H_PARAMS['V']

        #----------------------------------------

        # Kinetic energy
        if t != 0:
            s.printb("Kinetic energy operator t:\t%.2f" %-t)
            s.make_two_sites_operators( s.lattice_keys['nearest_n'], -t, s.model_operators['b'][0],     s.model_operators['b_dag'][0] )
            s.make_two_sites_operators( s.lattice_keys['nearest_n'], -t, s.model_operators['b_dag'][0], s.model_operators['b'][0] )

        #----------------------------------------

        # On-site interaction
        if U != 0:
            s.printb("On-site interaction operator U:\t%.2f" %U)
            s.make_onsite_operators( s.lattice_keys['site'], U/2., s.model_operators['n(n-1)'][0] )

        #----------------------------------------

        # Inter-site interaction
        if V != 0:
            s.printb("Inter-site interaction operator V:\t%.2f" %V)
            s.make_two_sites_operators( s.lattice_keys['nearest_n'], V, s.model_operators['n'][0], s.model_operators['n'][0] )

        s.printb( )



class Spin_Matter_Hamiltonian(Hamiltonian, Bipartite_Chain, Spin_Matter_Model):
    """The quantum link model Hamiltonian on a ladder lattice.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note
    ----
    asdsad

    Parameters
    ----------
    None

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """

    def __init__(s):
        #======================================================================
        # Initiate the Hamiltonian features.
        #======================================================================
        parameters_to_set_in_H_PARAMS = ['jy', 'm', 'k', 'Gauge_Theory', 'Number_Legs_Ladder', 'Boundary_Conditions'] # Lists all the variables that must be specified in H_PARAMS for this Hamiltonian.
        warn_the_user(parameters_to_set_in_H_PARAMS, s.H_PARAMS)
        #======================================================================
        # Define the lattice  --->  for key in s.lattice_keys: printa(key)
        #======================================================================
        Bipartite_Chain.__init__(s)
        #======================================================================
        # Define the model  --->  for key in s.model_operators: printa(key)
        #======================================================================
        Spin_Matter_Model.__init__(s, s.H_PARAMS['Gauge_Theory'], s.H_PARAMS['Number_Legs_Ladder'], s.H_PARAMS['Boundary_Conditions'])
        #======================================================================
        # Initiate the Hamiltonian features
        #======================================================================
        Hamiltonian.__init__(s)

    #==========================================================================
    # Define the Hamiltonian and its operators
    #==========================================================================
    def define_hamiltonian(s):

        if s.H_PARAMS['Gauge_Theory'] == 'quantum_link':
            s.define_qlm_hamiltonian( )
        elif s.H_PARAMS['Gauge_Theory'] == 'spin_ice':
            s.define_spinice_hamiltonian( )

        s.printb( )

    def define_qlm_hamiltonian(s):

        m  = s.H_PARAMS['m']
        jy = s.H_PARAMS['jy']

        s.number_internal_rungs = s.number_legs_ladder - 1
        if s.H_PARAMS['Boundary_Conditions'] == 'OBC':
            s.number_active_rungs = s.number_legs_ladder - 1
        elif s.H_PARAMS['Boundary_Conditions'] == 'PBC':
            s.number_active_rungs = s.number_legs_ladder
        elif s.H_PARAMS['Boundary_Conditions'] == 'TORUS':
            number_active_rungs = s.H_PARAMS['Number_Legs_Ladder']

        #----------------------------------------

        # Mass
        if m != 0:
            s.printb("Mass operator on: %.2f" %m)

            # rung 0
            mass_opt = s.model_operators['null']
            for leg_index in range(s.number_legs_ladder):
                mass_opt += (-1)**(leg_index+1)*s.model_operators['n_%d' %leg_index][0]
            s.make_onsite_operators( s.lattice_keys['first_site'], m, mass_opt )

            # rung 1
            mass_opt = s.model_operators['null']
            for leg_index in range(s.number_legs_ladder):
                mass_opt += (-1)**(leg_index)*s.model_operators['n_%d' %leg_index][1]
            s.make_onsite_operators( s.lattice_keys['second_site'], m, mass_opt )

        #----------------------------------------

        # Kinetic energy on the rungs
        if jy != 0:
            s.printb("Kinetic energy rungs on: %.2f" %-jy)

            # rung 0
            hopping_along_rung = s.model_operators['null']
            for rung_index in range(s.number_active_rungs):
                hopping_along_rung += s.model_operators['rung_%d_hop_down' %rung_index][0] + s.model_operators['rung_%d_hop_up' %rung_index][0]
            s.make_onsite_operators( s.lattice_keys['first_site'], -jy, hopping_along_rung )

            # rung 1
            hopping_along_rung = s.model_operators['null']
            for rung_index in range(s.number_active_rungs):
                hopping_along_rung += s.model_operators['rung_%d_hop_down' %rung_index][1] + s.model_operators['rung_%d_hop_up' %rung_index][1]
            s.make_onsite_operators( s.lattice_keys['second_site'], -jy, hopping_along_rung )

        #----------------------------------------

        # Kinetic energy on the legs
        s.printb("Kinetic energy legs on: %.2f" %-1.)

        # Leftmost operator acting on rung 0
        for leg_index in range(s.number_legs_ladder):
            s.make_two_sites_operators( s.lattice_keys['nearest_n_sublattice_0'], -1., s.model_operators['hop_off_leg_%d_towards_right' %leg_index][0], s.model_operators['hop_in_leg_%d_from_left'     %leg_index][1] )
            s.make_two_sites_operators( s.lattice_keys['nearest_n_sublattice_0'], -1., s.model_operators['hop_in_leg_%d_from_right'     %leg_index][0], s.model_operators['hop_off_leg_%d_towards_left' %leg_index][1] )
            if s.H_PARAMS['Boundary_Conditions'] == 'TORUS':
                s.make_two_sites_operators( s.lattice_keys['second_nnn_sublattice_0'], -1., s.model_operators['hop_off_leg_%d_towards_left' %leg_index][0], s.model_operators['hop_in_leg_%d_from_right'     %leg_index][1] )
                s.make_two_sites_operators( s.lattice_keys['second_nnn_sublattice_0'], -1., s.model_operators['hop_in_leg_%d_from_left'     %leg_index][0], s.model_operators['hop_off_leg_%d_towards_right' %leg_index][1] )

        # Leftmost operator acting on rung 1
        for leg_index in range(s.number_legs_ladder):
            s.make_two_sites_operators( s.lattice_keys['nearest_n_sublattice_1'], -1., s.model_operators['hop_off_leg_%d_towards_right' %leg_index][1], s.model_operators['hop_in_leg_%d_from_left'     %leg_index][0] )
            s.make_two_sites_operators( s.lattice_keys['nearest_n_sublattice_1'], -1., s.model_operators['hop_in_leg_%d_from_right'     %leg_index][1], s.model_operators['hop_off_leg_%d_towards_left' %leg_index][0] )
            if s.H_PARAMS['Boundary_Conditions'] == 'TORUS': # this should go
                s.make_two_sites_operators( s.lattice_keys['second_nnn_sublattice_1'], -1., s.model_operators['hop_off_leg_%d_towards_left' %leg_index][1], s.model_operators['hop_in_leg_%d_from_right'     %leg_index][0] )
                s.make_two_sites_operators( s.lattice_keys['second_nnn_sublattice_1'], -1., s.model_operators['hop_in_leg_%d_from_left'     %leg_index][1], s.model_operators['hop_off_leg_%d_towards_right' %leg_index][0] )


    def define_spinice_hamiltonian(s):

        s.number_internal_rungs = s.number_legs_ladder - 1
        if s.H_PARAMS['Boundary_Conditions'] == 'OBC':
            s.number_active_rungs = s.number_legs_ladder - 1
        elif s.H_PARAMS['Boundary_Conditions'] == 'PBC':
            s.number_active_rungs = s.number_legs_ladder
        elif s.H_PARAMS['Boundary_Conditions'] == 'TORUS':
            number_active_rungs = s.H_PARAMS['Number_Legs_Ladder']


class Spin_One_Hamiltonian(Hamiltonian, Chain, Spin_Model):

    def __init__(s):
        #======================================================================
        # Initiate the Hamiltonian features.
        #======================================================================
        parameters_to_set_in_H_PARAMS = ['D'] # Lists all the variables that must be specified in H_PARAMS for this Hamiltonian.
        warn_the_user(parameters_to_set_in_H_PARAMS, s.H_PARAMS)
        #======================================================================
        # Define the lattice  --->  for key in s.lattice_keys: printa(key)
        #======================================================================
        Chain.__init__(s)
        #======================================================================
        # Define the model  --->  for key in s.model_operators: printa(key)
        #======================================================================
        Spin_Model.__init__(s, 1)
        #======================================================================
        # Initiate the Hamiltonian features
        #======================================================================
        Hamiltonian.__init__(s)

    #==========================================================================
    # Define the Hamiltonian and its operators
    #==========================================================================
    def define_hamiltonian(s):

        D =  s.H_PARAMS['D']
        B =  s.H_PARAMS['B']
        Sx = s.H_PARAMS['Sx']
        Sy = s.H_PARAMS['Sy']
        Sz = s.H_PARAMS['Sz']

        # Anisotropy
        if D != 0:
            s.printb("On-site Sz^2 anisotropy operator: %.2f"%D)
            s.make_onsite_operators( keys['site'], D, dot(s.model_operators['Sz'],s.model_operators['Sz']) )

        # Spin-Spin
        if Sx!=0:
            s.printb("Sx-Sx operator: %.2f"%Sx)
            s.make_two_sites_operators( keys['nearest_n'], Sx, s.model_operators['Sx'], s.model_operators['Sx'])

        if Sy!=0:
            s.printb("Sy-Sy operator: %.2f"%Sy)
            s.make_two_sites_operators( keys['nearest_n'], Sy, s.model_operators['Sy'], s.model_operators['Sy'])

        if Sz!=0:
            s.printb("Sz-Sz operator: %.2f"%Sz)
            s.make_two_sites_operators( keys['nearest_n'], Sz, s.model_operators['Sz'], s.model_operators['Sz'])

        # Magnetic field
        if B != 0:
            s.printb("Magnetic field operator: %.2f"%B)
            s.make_onsite_operators( keys['site'], B, s.model_operators['Sx'] )

        print( )



class Spin_Half_Hamiltonian(Hamiltonian):

    def __init__(s, H_PARAMS):
        #======================================================================
        # Initiate the Hamiltonian features.
        #======================================================================
        parameters_to_set_in_H_PARAMS = ['J1', 'J2'] # Lists all the variables that must be specified in H_PARAMS for this Hamiltonian.
        warn_the_user(parameters_to_set_in_H_PARAMS, s.H_PARAMS)
        #======================================================================
        # Define the lattice  --->  for key in s.lattice_keys: printa(key)
        #======================================================================
        Chain.__init__(s)
        #======================================================================
        # Define the model  --->  for key in s.model_operators: printa(key)
        #======================================================================
        Spin_Model.__init__(s, .5)
        #======================================================================
        # Initiate the Hamiltonian features
        #======================================================================
        Hamiltonian.__init__(s)

    #==========================================================================
    # Define the Hamiltonian and its operators
    #==========================================================================
    def define_hamiltonian(s):

        J1 =  s.H_PARAMS['J1']
        J2 =  s.H_PARAMS['J2']

        # Nearest-Neighbours spin-spin coupling.
        if J1 != 0:
            s.printb("Nearest-neighbours spin coupling operator: %.2f"%J1)
            s.make_two_sites_operators( keys['nearest_n'], Sx, s.model_operators['Sx'], s.model_operators['Sx'])
            s.make_two_sites_operators( keys['nearest_n'], Sy, s.model_operators['Sy'], s.model_operators['Sy'])
            s.make_two_sites_operators( keys['nearest_n'], Sz, s.model_operators['Sz'], s.model_operators['Sz'])

        # Next-Nearest-Neighbours spin-spin coupling.
        if J2 != 0:
            s.printb("Next-nearest-neighbours spin coupling operator: %.2f"%J1)
            s.make_two_sites_operators( keys['nearest_n'], Sx, s.model_operators['Sx'], s.model_operators['Sx'])
            s.make_two_sites_operators( keys['nearest_n'], Sy, s.model_operators['Sy'], s.model_operators['Sy'])
            s.make_two_sites_operators( keys['nearest_n'], Sz, s.model_operators['Sz'], s.model_operators['Sz'])

        print( )



class AKLT(Hamiltonian):
    def __init__(s, H_PARAMS):
        #======================================================================
        # Initiate the Hamiltonian features
        #======================================================================
        Hamiltonian.__init__(s)
        #======================================================================
        # Define the lattice
        # for key in lattice.keys: printa(key)
        #======================================================================
        from lattices import Chain
        lattice = Chain( )
        keys = lattice.keys
        s.LATTICE_UNIT_CELL_SIZE = lattice.LATTICE_UNIT_CELL_SIZE
        #======================================================================
        # Define the model
        # for key in model.opts: printa(key)
        #======================================================================
        from models import Spin
        model = Spin(1)
        s.d = model.d
        s.model_operators = model.opts
        #======================================================================
        # Initiate the Hamiltonian features
        #======================================================================
        Hamiltonian.__init__(s, model, lattice, H_PARAMS)
        #======================================================================
        # Define the Hamiltonian and its operators
        #======================================================================
        g=H_PARAMS['g']
        # Spin-Spin
        s.make_two_sites_operators( keys['nearest_n'], 1, s.model_operators['Sx'], s.model_operators['Sx'])
        s.make_two_sites_operators( keys['nearest_n'], 1, s.model_operators['Sy'], s.model_operators['Sy'])
        s.make_two_sites_operators( keys['nearest_n'], 1, s.model_operators['Sz'], s.model_operators['Sz'])
        # Spin-squared
        s.make_two_sites_operators( keys['nearest_n'], g, dot(s.model_operators['Sx'],s.model_operators['Sx']), dot(s.model_operators['Sx'],s.model_operators['Sx']) )
        s.make_two_sites_operators( keys['nearest_n'], g, dot(s.model_operators['Sy'],s.model_operators['Sy']), dot(s.model_operators['Sy'],s.model_operators['Sy']) )
        s.make_two_sites_operators( keys['nearest_n'], g, dot(s.model_operators['Sz'],s.model_operators['Sz']), dot(s.model_operators['Sz'],s.model_operators['Sz']) )

        s.make_two_sites_operators( keys['nearest_n'], g, dot(s.model_operators['Sx'],s.model_operators['Sy']), dot(s.model_operators['Sx'],s.model_operators['Sy']) )
        s.make_two_sites_operators( keys['nearest_n'], g, dot(s.model_operators['Sy'],s.model_operators['Sx']), dot(s.model_operators['Sy'],s.model_operators['Sx']) )

        s.make_two_sites_operators( keys['nearest_n'], g, dot(s.model_operators['Sx'],s.model_operators['Sz']), dot(s.model_operators['Sx'],s.model_operators['Sz']) )
        s.make_two_sites_operators( keys['nearest_n'], g, dot(s.model_operators['Sz'],s.model_operators['Sx']), dot(s.model_operators['Sz'],s.model_operators['Sx']) )

        s.make_two_sites_operators( keys['nearest_n'], g, dot(s.model_operators['Sz'],s.model_operators['Sy']), dot(s.model_operators['Sz'],s.model_operators['Sy']) )
        s.make_two_sites_operators( keys['nearest_n'], g, dot(s.model_operators['Sy'],s.model_operators['Sz']), dot(s.model_operators['Sy'],s.model_operators['Sz']) )

