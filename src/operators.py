import numpy as np
from scipy.linalg import expm


def projector(d):
    P_list = []
    for i in range(d):
        P_i = np.zeros((d,d))
        P_i[i,i]=1
        P_list.append(P_i)
    return P_list


class Operator:
    def __init__(s):
        s.ONE = np.eye(s.d)       # Identity matrix.
        s.P_list = projector(s.d) # List of projector operators.


class Particle_Operators(Operator):
    def __init__(s):
        a = Angular_Momentum_Operators(0.5)
        s.d = 2
        s.psi_dag = a.LP    # Creation operator.
        s.psi = a.LM        # Destruction operator.
        s.n = np.dot(s.psi_dag, s.psi)  # Particle number, decreasingly sorted.
        s.n2 = np.dot(s.psi, s.psi_dag) # Particle number, increasingly sorted.
        s.phi = 2*a.LX                  # Particle-Hole inversion.
        Operator.__init__(s)


class Pauli_Operators(Operator):
    def __init__(s):
        a = Angular_Momentum_Operators(0.5)
        s.d = 2
        s.sigX = 2*a.LX     # sigma_x matrix.
        s.sigY = 2*a.LY     # sigma_y matrix.
        s.sigZ = 2*a.LZ     # sigma_z matrix.
        s.sigP = a.LP       # Raising operator.
        s.sigM = a.LM       # Lowering operator.
        Operator.__init__(s)


class Angular_Momentum_Operators(Operator):
    def __init__(s, l):
        s.d = int(2*l+1)
        null_opt = np.zeros((s.d,s.d))
        s.LP  = np.copy(null_opt)      # Raising operator.
        s.LM  = np.copy(null_opt)      # Lowering operator.
        s.LZ  = np.copy(null_opt)      # L_Z matrix
        # Make the operators L_X and L_Y.
        ind = [(i,j) for i in range(s.d) for j in range(s.d)]
        for i, j in ind:
            m = l-j
            if i==j:
                s.LZ[i,i] = l-i
            elif i+1==j:
                s.LP[i,j] = np.sqrt((l+m+1)*(l-m))
            elif i-1==j:
                s.LM[i,j] = np.sqrt((l-m+1)*(l+m))
        s.LX = (s.LP + s.LM)/np.sqrt(2)/np.sqrt(2)
        s.LY = (-s.LP + s.LM)/np.sqrt(2)/np.sqrt(2)*1.j
        # Make operators for pi rotations in the Bloch sphere.
        s.pi_rotation_x = np.real_if_close(expm(1.j*np.pi*s.LX))
        s.pi_rotation_y = np.real_if_close(expm(1.j*np.pi*s.LY))
        s.pi_rotation_z = np.real_if_close(expm(1.j*np.pi*s.LZ))
        # Make L_X,Y,Z squared and L squared.
        s.LX_sqrd = np.dot(s.LX, s.LX)
        s.LY_sqrd = np.real_if_close(np.dot(s.LY, s.LY))
        s.LZ_sqrd = np.dot(s.LZ, s.LZ)
        s.L_sqrd = s.LX_sqrd + s.LY_sqrd + s.LZ_sqrd
        Operator.__init__(s)


class Boson_Operators(Operator):
    def __init__(s, n):
        s.d = n+1
        # Define the operators.
        s.b      = np.zeros([s.d, s.d])
        s.b_dag  = np.zeros([s.d, s.d])
        s.n      = np.zeros([s.d, s.d])
        s.n2n    = np.zeros([s.d, s.d])
        # Populate the creaton, annihilation, particle density and n(n-1) operators.
        all_possible_tuples_of_rows_cols = [(i,j) for i in range(s.d) for j in range(s.d)]
        for i, j in all_possible_tuples_of_rows_cols:
            if i==j:
                s.n[i,i] = i
                s.n2n[i,i] = i*(i-1)
            elif i+1==j:
                s.b[i,j] = np.sqrt(j)
            elif i-1==j:
                s.b_dag[i,j] = np.sqrt(i)
        Operator.__init__(s)


class Fermion_Operators(Operator):
    def __init__(s):
        p = Pauli()
        s.d = 2
        s.c = p.SP
        s.c_dag = p.SM
        s.n = np.dot(s.c_dag,s.c)
        Operator.__init__(s)

