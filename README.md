kaggle-emc
==========

This repository contains the code I used to compete in the the EMC Data Science Challenge on Kaggle.  You can find details of the competition [here](http://www.kaggle.com/c/emc-data-science/).  The first step in using these scripts is to grab the data files test_data.csv, train_data.csv, and train_labels.csv from the kaggle competition site and dump them in the data directory.  From there, do:

python scripts/emc_convert_to_mat_file.py
python scripts/emc_row_normalize_data.py

And then, you're off!

