from sampling_config import sample_from_state
import sys
import datetime


if __name__=='__main__':
    L = sys.argv[1]
    state_name = sys.argv[2]
    print(datetime.datetime.now().strftime("%H-%M-%S %Y-%m-%d"))
    sample_from_state(L, state_name)
    print(datetime.datetime.now().strftime("%H-%M-%S %Y-%m-%d"))