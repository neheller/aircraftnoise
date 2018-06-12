from preprocessor import Preprocessor
import sys

# Proportions of data in each resulting set
TRPROP = 1.0 # Training
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
    if (len(sys.argv) < 3):
        print "Please provide dimensionality and output location"
        sys.exit()
    elif (len(sys.argv) > 3):
        print "Expecting exactly two arguments (dimensionality and location)"
        sys.exit()

    # Get provided dimesnionality
    dim = int(sys.argv[1])
    location = sys.argv[2]
    print
    print '\033[92m' + "*************************************************" + '\033[0m'
    print
    print "Provided dimensionality:", dim
    print "Provided location:      ", location

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
    print "Partitioning the data into training, testing, and validation sets"
    print
    trX, trY, teX, teY, vaX, vaY = ppr.partition_for_training(X, Y, TRPROP, TEPROP)

    print
    print '\033[94m' + "*************************************************" + '\033[0m'
    print
    print "Storing the data at", location
    print
    ppr.store_training_partitions(trX, trY, teX, teY, vaX, vaY, location)

    print '\033[92m' + "*************************************************" + '\033[0m'
    print
