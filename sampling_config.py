import qutip as qt
import numpy as np
import os
import json
from qutip_utils import get_random_local_measurements

numpy_seed = 4
Nsamp = int(1E4)

STATEDIR = "/home/btimar/data/states/qutip"
DATADIR = "/home/btimar/data/random_unitary_samples/from_qutip_states/seed_4"

with open( os.path.join(STATEDIR, 'settings.json')) as f:
    settings = json.load(f)

system_sizes = settings['system_sizes']
state_names = settings['state_names']

def get_state_path(state_name, L):
    return os.path.join(STATEDIR, "{0}_L={1}".format(state_name, L))

def get_state(state_name, L):
    return qt.qload(get_state_path(state_name, L))

def get_output_path(state_name, L):
    return os.path.join(DATADIR, "{0}_L={1}".format(state_name, L))

def write_config():
    with open(os.path.join(DATADIR, 'settings.json'), 'w') as f:
        data_settings = dict(params=settings['params'], system_sizes=system_sizes, 
                        state_names=state_names, state_dir=STATEDIR, Nsamp=Nsamp
                        numpy_seed=numpy_seed)
        json.dump(data_settings, f)

def sample_from_state(L, state_name):
    print("Generating data for system size {0}, state {1}".format(
                    L, state_name ))
    np.random.seed(numpy_seed)
    psi = get_state(state_name, L)
    settings, samples = get_random_local_measurements(psi, Nsamp)

    path = get_output_path(state_name, L)
    np.save(path + "_settings", settings)
    np.save(path + "_samples", samples)

if __name__=='__main__':
    write_config()
