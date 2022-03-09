import pandas as pd
from datetime import datetime

import zipfile

from data_functions import load_datasets

from constants import nsample, rseed, models_path, submissions_path

class base_model():

    def __init__(self, class_name, object_key, users_key, nsample=None):

        ## For user later when writing submissions
        self.class_name = class_name

        # Number of transactions to take from sales data (if none - take all)
        self.nsample = nsample

        # Path (directory) for models
        self.models = models_path

        # Get the datasets loaded up
        self.objects, self.users, self.sales = load_datasets(self.nsample)

        # Strings - names of columns
        self.object_key = object_key
        self.users_key = users_key

        # Init empty predictions dataframe
        self.predictions = pd.DataFrame()

    def save_model(self, name_prefix):
        """ Save the model to a pickled object"""

        # Now in YYYYMMDD_HHMMSS format
        now = datetime.today().strftime('%Y%m%d_%H%M%S')

        # Filename to output
        fname = f"{name_prefix}_{now}.pkl"

        start = datetime.now()
        # Open up binary file and dump object
        with open(fname, 'wb') as f: pickle.dump(self, f)
        print(f"Saved model with name {fname}")
        print(f"Processing time: {datetime.now() - start}")

    def zip_file(self, file):

        """ .zip the predictions once they have been made"""
        pass

    def write_predictions(self):

        """ Base class output of predictions """
        now = datetime.today().strftime('%Y%m%d_%H%M%S')

        fpath = f"{submissions_path}/submission_{self.class_name}_{now}.csv"
        print(f"Writing submissions to file: {fpath}")

        start = datetime.now()
        with open(fpath, 'wt') as f: self.predictions.to_csv(f, index=False)
        print(f"Completed writing file took {datetime.now()-start}")


class random_selections(base_model):

    def __init__(self, class_name, object_key, users_key, nsample=None):

        # Run the init method from the base model class
        super().__init__(class_name, object_key, users_key, nsample)

    def select_items(self, seed=None, num=12, selection_type='random'):

        """ Return num items from the objects dataframe
        """
        if selection_type == 'random':
            return self.objects[self.object_key].sample(num, random_state=seed)

    def all_items_same(self, seed=None, num=12):

        # Get some items
        items = self.select_items(seed, num)

        # Turn items into a space delimited string
        items_space = ' '.join(items.astype('str').to_list())

        # Initalise predictions dataframe
        self.predictions = self.users[self.users_key].to_frame().copy()

        self.predictions['prediction'] = items_space

    def make_predictions(self, method='same', seed=None, num=12):

        print(f"Make predictions with method = {method}")
        # Randomly select articles
        if method == 'same':
            self.all_items_same(seed, num)




object_key = 'article_id'
users_key = 'customer_id'

if __name__ == "__main__":

    # Initialise model type
    rs = random_selections('random_selections', object_key, users_key, nsample=100)
    rs.make_predictions('same', 0)