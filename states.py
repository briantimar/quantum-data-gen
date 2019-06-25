import numpy as np
import qutip as qt
from qutip_hamiltonians import get_hamiltonian


def get_ground_state(model_name, *args):
    h = get_hamiltonian(model_name, *args)
    evals, evecs = h.eigenstates(eigvals=1)
    return evecs[0]