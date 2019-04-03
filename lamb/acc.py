import sys
import numpy as np

print 'Usage: python acc.py Filename'
filename = sys.argv[1];
config = ""
best_acc = 0.0
with open(filename) as f:
    for line in f:
        if "batch_size" in line:
            config = line
        if "accuracy" in line:
            mystr = line.split();
            acclist = mystr[-1].split('}');
            acc = float(acclist[0]);
            if acc > best_acc:
                best_acc = acc
                print(config)
                print("best accuracy: " + str(best_acc))
