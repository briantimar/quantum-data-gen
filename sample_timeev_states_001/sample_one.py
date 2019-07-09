from sampling_config import sample_from_state, system_sizes, times
import sys
import datetime

if __name__ == '__main__':
    L = sys.argv[0]
    time_index = sys.argv[1]
    N = sys.argv[2]
    seed = sys.argv[3]
    sample_from_state(L, time_index, N, seed)

