import numpy as np
import qutip as qt
from qutip_hamiltonians import get_hamiltonian


def get_ground_state(model_name, param_dict):
    h = get_hamiltonian(model_name, param_dict)
    evals, evecs = h.eigenstates(eigvals=1)
    return evecs[0]