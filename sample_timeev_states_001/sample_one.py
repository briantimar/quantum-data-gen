from sampling_config import sample_from_state, system_sizes, times
import sys
import datetime

if __name__ == '__main__':
    L = int(sys.argv[1])
    time_index = int(sys.argv[2])
    N = int(sys.argv[3])
    seed = int(sys.argv[4])
    sample_from_state(L, time_index, N, seed)

