from sampling_config import sample_from_state
import sys


if __name__=='__main__':
    L = sys.argv[1]
    state_name = sys.argv[2]
    sample_from_state(L, state_name)