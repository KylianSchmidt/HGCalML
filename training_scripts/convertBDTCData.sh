#!/bin/bash
# Script wrapping convertFromSource which normalizes the input data for better
# performance. The normalization is kept in the same directory as the original

rm -rf /ceph/kschmidt/beamdump/alps/simulation/Testing
convertFromSource.py -i /ceph/kschmidt/beamdump/alps/simulation/Alp.txt \
                     -o /ceph/kschmidt/beamdump/alps/simulation/Testing \
                     -c TrainData_TrackReco.