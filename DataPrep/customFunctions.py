# Importing Library
import gzip
import shutil

# Function to Unzip the ".txt.gz" into tsv file
def unzip(filename):

    # Open the gzip
    with gzip.open(filename, 'rb') as f_in:

        # Make the custom name and save
        with open(filename.split("_")[0] +'_rawCounts.tsv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
