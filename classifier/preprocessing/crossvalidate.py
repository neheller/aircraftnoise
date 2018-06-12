from preprocessor import Preprocessor
import sys

# Proportions of data in each resulting set
TRPROP = 0.8 # Training
TEPROP = 0.0 # Testing
# VALIDATION SET IS REST

# Names of files containing the raw data
input_files = ['../raw_data/oml_final.csv', '../raw_data/400_community_events.csv']


# IDs of events in the first set that kill it
bad = [53905373, 53906999, 53907026, 53907026, 53907030, 53905373,
       53905426, 53907014, 53905400, 53905433, 53905397, 53905371,
       53905392]


if __name__ == "__main__":
    # Handle invalid calls
    if (len(sys.argv) < 4):
        print "Please provide dimensionality, output location, and number of folds"
        sys.exit()
    elif (len(sys.argv) > 4):
        print "Expecting exactly three arguments (dimensionality, location, and folds)"
        sys.exit()

    # Get provided dimesnionality
    dim = int(sys.argv[1])
    location = sys.argv[2]
    folds = int(sys.argv[3])
    print
    print '\033[92m' + "*************************************************" + '\033[0m'
    print
    print "Provided dimensionality:", dim
    print "Provided location:      ", location
    print "Provided folds:         ", folds

    print
    print '\033[94m' + "*************************************************" + '\033[0m'
    print
    print "Constructing preprocesor object..."
    print
    ppr = Preprocessor()

    print
    print '\033[94m' + "*************************************************" + '\033[0m'
    print
    print "Fetching data from storage..."
    print
    X, Y, events_found = ppr.get_raw_data(dim, input_files, bad)
    print
    print "Number of raw events found in database:            ", events_found
    print "Number of raw events that script was able to parse:", X.shape[0]

    print
    print '\033[94m' + "*************************************************" + '\033[0m'
    print
    print "Removing invalid and outlying events from dataset..."
    print
    X, Y = ppr.remove_outliers(X, Y)

    print
    print '\033[94m' + "*************************************************" + '\033[0m'
    print
    print "Normalizing data to have mean zero..."
    print
    X, Y = ppr.normalize(X, Y)

    print
    print '\033[94m' + "*************************************************" + '\033[0m'
    print
    print "Partitioning the data into folds"
    print
    lst = ppr.partition_for_cross_validation(X, Y, folds)
    for i in range(0,folds):
        print "fold:", i, lst[i][0].shape[0], "events"

    print
    print '\033[94m' + "*************************************************" + '\033[0m'
    print
    print "Storing the data at", location
    print
    ppr.store_cv_folds(lst, location)

    print '\033[92m' + "*************************************************" + '\033[0m'
    print
