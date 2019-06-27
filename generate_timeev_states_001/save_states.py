import numpy as np
import sys
import json
import os
import qutip as qt
sys.path.append('..')

ROOT = "/Users/btimar/Code/quantum-data-gen"
sys.path.append(ROOT)

from timeev import generate_xy_states

system_sizes = [2, 4, 6, 8, 10]
times = list(np.linspace(0, .01, 5))

SAVEDIR = "/Users/btimar/Dropbox/data/states/qutip/timeev/generate_timeev_states_001"

settings = {'model': 'xy', 
            'system_sizes': system_sizes,
            'times': times}
with open(os.path.join(SAVEDIR, 'settings.json'), 'w') as f:
    json.dump(settings, f)

for L in system_sizes:
    states = generate_xy_states(L, times).states
    for i in range(len(states)):
        qt.qsave(states[i], os.path.join(SAVEDIR, 'xy_timeev_L={0}_tindex={1}'.format(
                                                    L,i
        )))
