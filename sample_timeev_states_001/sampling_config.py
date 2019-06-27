import qutip as qt
import numpy as np
import os
import json
from qutip_utils import get_random_local_measurements

numpy_seed = 4
Nsamp = int(1E4)

STATEDIR = "/Users/btimar/Dropbox/data/states/qutip/timeev/generate_timeev_states_001"
DATADIR = "/Users/btimar/Dropbox/data/random_unitary_data/from_qutip_states/sample_timeev_states_001"

with open( os.path.join(STATEDIR, 'settings.json')) as f:
    settings = json.load(f)

system_sizes = settings['system_sizes']
times = settings['times']
nt = len(times)

def get_state_path(system_size, time_index):
    return os.path.join(STATEDIR, "xy_timeev_L={0}_tindex={1}".format(system_size, time_index))

def get_state(system_size, time_index):
    return qt.qload(get_state_path(system_size, time_index))

def get_output_path(system_size, time_index):
    return os.path.join(DATADIR, "xy_timeev_L={0}_tindex={1}".format(system_size, time_index))

def write_config():
    with open(os.path.join(DATADIR, 'settings.json'), 'w') as f:
        data_settings = dict(times=times,system_sizes=system_sizes, 
                             state_dir=STATEDIR, Nsamp=Nsamp,
                        numpy_seed=numpy_seed)
        json.dump(data_settings, f)

def sample_from_state(L, tindex):
    print("Generating data for system size {0}, time index {1}".format(
                    L, tindex ))
    np.random.seed(numpy_seed)
    psi = get_state(L, tindex)
    settings, samples = get_random_local_measurements(psi, Nsamp)

    path = get_output_path(L, tindex)
    np.save(path + "_settings", settings)
    np.save(path + "_samples", samples)

if __name__=='__main__':
    write_config()
