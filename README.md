# aircraftnoise

The code to accompany our technical report on aircraft noise monitoring.

## Data
The raw data can be found in two csv files the `data` subdir, namely
`oml_final.csv` and `400_community_events.csv`. I read these into `event2d`
objects where the 1/3 octave data is available as event.rawmat. See
`preprocess.py` for an example of this.

## Requirements

TensorFlow > 1.0.0, Python 2, numpy, matplotlib

## Usage

If you would like to run training and see results on a hold-out set,
navigate to the `classifier` subdir and run
```
make cvconvnet
```
This will train on 9/10 folds about 50 times and log the results.

## Contact

Please contact Nicholas Heller at helle246@umn.edu if you have any questions
about this work.
