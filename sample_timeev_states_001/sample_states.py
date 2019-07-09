from sampling_config import sample_from_state, system_sizes, times
import sys
import datetime
from multiprocessing import Pool


if __name__=='__main__':
    args = []
    for L in system_sizes:
        for it in range(len(times)):
            args.append((L, it))
            
    def sample(args):
        sample_from_state(args[0], args[1],int(1e4),4)
    
    p = Pool(4)
    p.map(sample, args)