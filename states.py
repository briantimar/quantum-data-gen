import numpy as np
import qutip as qt
from qutip_hamiltonians import get_hamiltonian


def get_ghz_state(param_dict):
    L = param_dict['L']
    psi1 = qt.tensor([qt.basis(2, 0), qt.basis(2, 1)] * (L//2) + [qt.basis(2, 0)] * (L%2))
    psi2 = qt.tensor([qt.basis(2, 1), qt.basis(2, 0)] * (L//2) + [qt.basis(2, 1)] * (L%2))
    return (1.0/np.sqrt(2) * (psi1 + psi2))

def get_ground_state(model_name, param_dict):
    h = get_hamiltonian(model_name, param_dict)
    evals, evecs = h.eigenstates(eigvals=1)
    return evecs[0]

