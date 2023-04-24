
import sys
import numpy as np
from DCENet import DCENet


helpstr = \
"Script for training a DCEnet from the command line.\n \
\n \
Arguments\n \
----------------------------------------------------------------------------------\n \
--help/-h:      Print this information.\n \
--infile/-i:    Filename from which to read data used in DCEnet.fit()\n \
\n \
Corresponding to DCEnet constructor parameters:\n \
\n \
--seed/-s, --dim/-m, --hdims/-H, --loadfile/-l\n \
\n \
Corresponding to DCEnet.fit() parameters other than data:\n \
\n \
--delay/-d, --dt/-D, --savefreq/-f, --savename/-n, --batchsize/-B, --epochs/-e, --lookahead/-a, --lookbehind/-b"


infile = "no infile specified"
seed = 0
edim = 7
hdims = []
loadfile = None
delay = dt = lookahead = lookbehind = None
savefreq = 25
savename = None
batchsize = 32
epochs = 1000


for n in range(len(sys.argv)):

    if sys.argv[n] in ('--help', '-h'):
        print(helpstr)
        exit(0)

    elif sys.argv[n] in ('--infile', '-i'):
        infile = sys.argv[n + 1]

    elif sys.argv[n] in ('--seed', '-s'):
        seed = int(sys.argv[n + 1])

    elif sys.argv[n] in ('--dim', '-m'):
        edim = int(sys.argv[n + 1])

    elif sys.argv[n] in ('--hdims', '-H'):
        while 1:
            try:
                n += 1
                hdims.append(int(sys.argv[n]))
            except ValueError or IndexError:
                break

    elif sys.argv[n] in ('--loadfile', '-l'):
        loadfile = sys.argv[n + 1]

    elif sys.argv[n] in ('--delay', '-d'):
        delay = int(sys.argv[n + 1])

    elif sys.argv[n] in ('--dt', '-D'):
        dt = float(sys.argv[n + 1])

    elif sys.argv[n] in ('--savefreq', '-f'):
        savefreq = int(sys.argv[n + 1])

    elif sys.argv[n] in ('--savename', '-n'):
        savename = sys.argv[n + 1]

    elif sys.argv[n] in ('--batchsize', '-B'):
        batchsize = int(sys.argv[n + 1])

    elif sys.argv[n] in ('--epochs', '-e'):
        epochs = int(sys.argv[n + 1])

    elif sys.argv[n] in ('--lookahead', '-a'):
        lookahead = int(sys.argv[n + 1])

    elif sys.argv[n] in ('--lookbehind', '-b'):
        lookbehind = int(sys.argv[n + 1])


data = np.load(infile)
DCENet(seed, edim, hdims, loadfile).fit(data, delay, dt, savefreq, savename, batchsize, epochs, lookahead, lookbehind)
