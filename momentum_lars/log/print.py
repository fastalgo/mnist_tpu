import sys
import numpy as np

# batch_size: 60000; init_lr: 10.0; poly_power: 0.5; warm_up: 17
print 'Usage: python acc.py Filename'
filename = sys.argv[1];
config = ""
best_acc = 0.0
warm_up = 0.0
learing_rate = 0.0
w_list = []
l_list = []
a_list = []
with open(filename) as f:
    for line in f:
        if "batch_size" in line:
            print(line)
            config = line
            mystr = line.split();
            warmlist = mystr[-1]
            warm_up = float(warmlist)
            lrlist = mystr[3].split(';')
            learning_rate = float(lrlist[0])
            print(warm_up)
            print(learning_rate)
            w_list.append(warm_up)
            l_list.append(learning_rate)
        if "accuracy" in line:
            print(line)
            mystr = line.split();
            acclist = mystr[-1].split('}');
            acc = float(acclist[0]);
            print(acc)
            a_list.append(acc)
            if acc > best_acc:
                best_acc = acc
                # print(config)
                # print("best accuracy: " + str(best_acc))
print(w_list)
print(l_list)
print(a_list)
