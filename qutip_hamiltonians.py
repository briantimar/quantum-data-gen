""" Some hamiltonian constructors in qutip"""
import qutip as qt
import numpy as np
from qutip_utils import embed, n

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


def rydberg_hamiltonian_1d(param_dict, bc='open'):
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

def tfim_1d(param_dict, bc='open'):
    """ A transverse-field ising model in one dimension.
    H = -hz sum(z) -hx sum(x) - Jsum(z_i z_i+1) """

    L = param_dict['L']
    hz = param_dict['hz']
    hx = param_dict['hx']
    J = param_dict['J']

    Hz = sum( [ -hz * embed([qt.sigmaz()], L, [i]) for i in range(L)])
    Hx = sum( [ -hx * embed([qt.sigmax()], L, [i]) for i in range(L)])
    H = Hz + Hx

    coupmax = L-2 if bc=='open' else L-1
    for i in range(coupmax):
        H += -J * embed([qt.sigmaz(), qt.sigmaz()], L, [i, (i+1)%L])
    return h

def heisenberg_1d(param_dict, bc='open'):
    """ A heisenberg model in one dimension.
        H = -J sum si . s_i+1
    """

    L = param_dict['L']
    J = param_dict['J']

    coupmax = L-2 if bc=='open' else L-1
    pauli_x = [ embed([qt.sigmax(), qt.sigmax()], L, [i, i+1]) for i in range(coupmax)]
    pauli_y = [ embed([qt.sigmay(), qt.sigmay()], L, [i, i+1]) for i in range(coupmax)]
    pauli_z = [ embed([qt.sigmaz(), qt.sigmaz()], L, [i, i+1]) for i in range(coupmax)]

    return -J * (1/4) * (sum(pauli_x) + sum(pauli_y) + sum(pauli_z))


hamiltonian_constructors = { 'rydberg': rydberg_hamiltonian_1d, 
                              'tfim': tfim_1d, 
                              'heisenberg': heisenberg_1d }

def get_hamiltonian(name, *args):
    if name not in hamiltonian_constructors.keys():
        raise NotImplementedError
    return hamiltonian_constructors[name](*args)