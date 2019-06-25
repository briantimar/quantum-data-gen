import numpy as np
from states import get_ground_state, get_ghz_state
import os
import json
import qutip as qt

state_names = ['tfim_ground', 'rydberg_ground', 'heisenberg_ground', 'ghz']
system_sizes = range(2, 12, 2)


### Set hamiltonian parameters here
rydberg_params = { 'V': 1.0, 'Omega': .2, 'Delta': .2, 'ktrunc': 2, 'bc': 'closed'}
heisenberg_params = {'J': -1.0, 'bc': 'closed'}
tfim_params = {'J': 1.0, 'hz': .5, 'hx': .6, 'bc': 'closed'}
ghz_params = {}

params = {'rydberg': rydberg_params, 'heisenberg': heisenberg_params, 
            'tfim': tfim_params, 'ghz': ghz_params}

SAVEDIR = "/Users/btimar/Dropbox/data/states/qutip"

def get_state(state_name, system_size):
    if state_name == 'ghz':
        param_dict = {**params['ghz'], 'L': system_size}
        return get_ghz_state(param_dict)
    else:
        model_name = state_name.split('_')[0]
        param_dict = {**params[model_name], 'L': system_size}
        return get_ground_state(model_name, param_dict)

with open(os.path.join(SAVEDIR, 'settings.json'), 'w') as f:
    settings = dict(params=params, system_sizes = list(system_sizes), state_names=state_names)
    json.dump(settings, f)

for L in system_sizes:
    for state_name in state_names:
        print("computing state {0} for system size {1}".format(state_name,
                                                        L))
        psi = get_state(state_name, L)
        qt.qsave(psi, 
                    os.path.join(SAVEDIR, "{0}_L={1}".format(state_name, L)))