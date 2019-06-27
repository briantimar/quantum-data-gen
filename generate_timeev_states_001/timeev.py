import numpy as np
import sys
import json
sys.path.append('..')
from qutip_hamiltonians import get_hamiltonian
import qutip as qt


def do_evolution(rho0, H, times):
    return qt.mesolve(H, rho0, times)

def generate_xy_states(L, times):
    H = get_hamiltonian('xy', dict(L=L))
    psi0 = qt.tensor([qt.basis(2,0), qt.basis(2,1)]*(L//2) + [qt.basis(2,0)] * (L%2))
    return do_evolution(psi0, H, times)