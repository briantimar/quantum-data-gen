""" Some hamiltonian constructors in qutip"""
import qutip as qt
import numpy as np
from qutip_utils import embed, n
import itertools as it

def dist1d(i, j, L, bc='open'):
    """Distance between two lattice sites on a 1d chain.
        i, j : site labels.
        L: length of the chain.
        bc: boundary conditions for the chain. 'open' or 'closed'.
        """
    if bc == 'open':
        return abs(i-j)
    elif bc == 'closed':
        return min((abs(i-j), abs(i-j + L), abs(i-j-L)))
    raise NotImplementedError


def rydberg_hamiltonian_1d(param_dict):
    """ A 1d rydberg hamiltonian implemented as QObj.
        L: the number of qubits.
        Delta: the laser detuning / long. field
        Omega: the Rabi frequency or transverse field.
        V: the nearest-neighbor interaction strength.
        ktrunc: the range at which to truncate interactions.
        bc: the boundary conditions.

        Returns: QObj representing the hamiltonian"""

    L = param_dict['L']
    Delta = param_dict['Delta']
    Omega = param_dict['Omega']
    V = param_dict['V']
    ktrunc = param_dict['ktrunc']
    bc = param_dict.get('bc', 'closed')

    #start with the laser terms
    h = sum([-Delta * embed([n()], L, [i]) for i in range(L)])
    h += sum([-(Omega/2.0) * embed([qt.sigmax()], L, [i]) for i in range(L)])

    #add pairwise interactions up to ktrunc.
    # the lazy way
    for i in range(L):
        for j in range(i+1, L):
            d = dist1d(i, j, L, bc=bc)
            if d <= ktrunc:
                coupling = V / d**6
                h += coupling * embed([n(), n()], L, [i, j])
    return h

def tfim_1d(param_dict):
    """ A transverse-field ising model in one dimension.
    H = -hz sum(z) -hx sum(x) - Jsum(z_i z_i+1) """

    L = param_dict['L']
    hz = param_dict['hz']
    hx = param_dict['hx']
    J = param_dict['J']
    bc = param_dict.get('bc', 'closed')

    Hz = sum( [ -hz * embed([qt.sigmaz()], L, [i]) for i in range(L)])
    Hx = sum( [ -hx * embed([qt.sigmax()], L, [i]) for i in range(L)])
    H = Hz + Hx

    coupmax = L-1 if bc=='open' else L
    if bc =='closed' and L==2:
        J = J/2
    for i in range(coupmax):
        H += -J * embed([qt.sigmaz(), qt.sigmaz()], L, [i, (i+1)%L])
    return H

def heisenberg_1d(param_dict):
    """ A heisenberg model in one dimension.
        H = -J sum si . s_i+1
    """

    L = param_dict['L']
    J = param_dict['J']
    bc = param_dict.get('bc', 'closed')

    coupmax = L-1 if bc=='open' else L
    pauli_x = [ embed([qt.sigmax(), qt.sigmax()], L, [i, (i+1)%2]) for i in range(coupmax)]
    pauli_y = [ embed([qt.sigmay(), qt.sigmay()], L, [i, (i+1)%2]) for i in range(coupmax)]
    pauli_z = [ embed([qt.sigmaz(), qt.sigmaz()], L, [i, (i+1)%2]) for i in range(coupmax)]

    if bc=='closed' and L==2:
        J = J/2

    return -J * (1/4) * (sum(pauli_x) + sum(pauli_y) + sum(pauli_z))


def get_H_XY_terms(N):
    """ Returns local terms, without coupling coefficients, that make up the XY hamiltonian.
        returns: Hx, Hz, HPM
        Hx = list of pauli X's
        Hz = list of Pauli Z's
        HPM = list of sigma_plus * sigma_minus """

    I=qt.tensor([qt.qeye(2)] * N)
    c=list(it.combinations(range(N), 2))
    Hz,Hx, HPM=[0 * I for x in range(N)],[0 * I for x in range(N)], [0 * I for x in c]
    for i in range(N):
        l=[qt.qeye(2)] * N
        l[i]=qt.sigmaz()
        Hz[i]=qt.tensor(l)
        l[i]=qt.sigmax()
        Hx[i]=qt.tensor(l)
    for s in range(len(c)):
        i, j=c[s]
        l=[qt.qeye(2)] * N
        l[i]=qt.sigmap()
        l[j]=qt.sigmam()
        HPM[s]=qt.tensor(l)
        HPM[s]+=HPM[s].dag()
    return Hx,Hz, HPM


def construct_XY_Hamiltonian(param_dict):
    """Construct a matrix representation of the XY hamiltonian.
    The interaction strengths are loaded from Jij_10.npy; hence system size can be at most 10"""
    N = param_dict['N']
    if N > 10:
        raise ValueError("coupling strengths for N > 10 not provided")
        
    __, Hz, HPM = get_H_XY_terms(N)   # returns lists containing the individual terms of the Hamiltonian
                            # Hz: local sigma_z terms
                            # HPM: flip flop (between every to sites)

    disorder=[0.]*N  # optional disorder potential (here put to zero)

    ### Construct the Jij matrix
    Jij = np.load("Jij_10.npy")

    ### Construct the Hamiltonian
    H = sum([disorder[i]  * h for i,h in enumerate(Hz)], 0) # Local terms
    c=list(it.combinations(range(N), 2)) # all combinations (but lower triangular)
    for s, cs in enumerate(c):
        i, j = cs
        H+=HPM[s]*(Jij[i,j]+Jij[j,i])

    return H



hamiltonian_constructors = { 'rydberg': rydberg_hamiltonian_1d, 
                              'tfim': tfim_1d, 
                              'heisenberg': heisenberg_1d, 
                              'xy': construct_XY_Hamiltonian }

def get_hamiltonian(name, *args):
    if name not in hamiltonian_constructors.keys():
        raise NotImplementedError
    return hamiltonian_constructors[name](*args)