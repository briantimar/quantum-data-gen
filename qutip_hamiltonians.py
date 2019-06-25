""" Some hamiltonian constructors in qutip"""
import qutip as qt
import numpy as np
from qutip utils import embed, n

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


def rydberg_hamiltonian_1d(L, Delta, Omega, V, ktrunc, bc='open'):
    """ A 1d rydberg hamiltonian implemented as QObj.
        L: the number of qubits.
        Delta: the laser detuning / long. field
        Omega: the Rabi frequency or transverse field.
        V: the nearest-neighbor interaction strength.
        ktrunc: the range at which to truncate interactions.
        bc: the boundary conditions.

        Returns: QObj representing the hamiltonian"""

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

def tfim_1d(L, hz, hz, J, bc='open'):
    """ A transverse-field ising model in one dimension.
    H = -hz sum(z) -hx sum(x) - Jsum(z_i z_i+1) """
    hz = sum( [ -hz * embed([qt.sigmaz()], L, [i]) for i in range(L))
    hx = sum( [ -hx * embed([qt.sigmax()], L, [i]) for i in range(L))
    h = hx + hz

    coupmax = L-2 if bc=='open' else L-1
    for i in range(coupmax):
        h += -J * embed([qt.sigmaz(), qt.sigmaz()], L, [i, (i+1)%L])
    return h