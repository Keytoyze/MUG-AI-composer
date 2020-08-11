import random
import os
from RhythmGenerator import get_item

def run(processes):
    for p in processes:
        p.start()
    for p in processes:
        p.join()

if __name__ == "__main__":
    
    import json
    import random
    import sys
    config_path
    lock = multiprocessing.Lock()

    p1 = multiprocessing.Process(target = evaluate, args = (m1, m2, lock))
    p2 = multiprocessing.Process(target = evaluate, args = (m2, m1, lock))
    run(p1, p2)
            