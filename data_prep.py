from datetime import datetime
from scipy import sparse

import pickle

from sklearn.decomposition import NMF
from constants import nsample, rseed, verbose, models_path

from data_functions import load_datasets

class nmf_model():

    def __init__(self, object_key, users_key, nsample=None):

        # Number of transactions to take from sales data (if none - take all)
        self.nsample = nsample

        # Path (directory) for models
        self.models = models_path

        # Get the datasets loaded up
        self.objects, self.users, self.sales = load_datasets(self.nsample)

        # Strings - names of columns
        self.object_key = object_key
        self.users_key = users_key

        # For each user get volume of sales per objet
        self.X_counts = self.get_counts()

        # For all objects (rows) and users (column) populate matrix with counts of sales
        self.X = self.create_X_matrix(self.X_counts)


    def get_counts(self):
        """

        Calculates the counts of purchases made by 'users' of 'objects'
        The resulting dataframe has a single row per article and customer with the total number of purchases made.
            The row_idx and col_idx values provide the co-ordinates for article X (row) for customer Y (column)'s cell value.

        :param objects: dataframe representing the objects available for sale (e.g. df_articles)
        :param object_key: string - unique identifier for the objects (e.g. article_id)
        :param users: dataframe representing the users/customers available for making purchaes (e.g. df_customers)
        :param users_key: string - unique identifier for the customers/users (e.g. customer_id)
        :param sales: dataframe holding actual training - transactions made
        :return: dataframe with columns: object_key | users_key | count | row_idx | col_idx
        """

        n_unique_users = self.users[self.users_key].nunique()
        print(f"For each of the {n_unique_users:,} users - count up volume of transactions for each of the objects")

        # Summarise sales (counts)
        sales_summary = self.sales.groupby([self.object_key, self.users_key]).size().reset_index().rename(columns={0: 'volume'})

        # Get index position for rows and columns
        sales_summary = sales_summary.merge(self.objects[self.object_key].reset_index(), on=self.object_key, how='left')
        sales_summary.rename(columns={'index': 'row_idx'}, inplace=True)

        sales_summary = sales_summary.merge(self.users[self.users_key].reset_index(), on=self.users_key, how='left').sort_values(
            [self.object_key, self.users_key])

        sales_summary.rename(columns={'index': 'col_idx'}, inplace=True)

        return sales_summary

    def create_X_matrix(self, counts):
        """

        :param objects:
        :param users:
        :param counts:
        :return:
        """
        print(f"Now for our objects/user matrix - populate with the sales we have observed")

        # Create a sparse matrix of shape 'articles' by 'customers'
        X_ = sparse.lil_matrix((self.objects.shape[0], self.users.shape[0]))

        # Update values of sparse matrix
        X_[counts.row_idx, counts.col_idx] = counts.volume

        print(f"Shape of X = {X_.shape}")

        return X_

    def create_model(self, n_components, init, random_state, verbose):

        """
        Create instance of NMF model and train it

        :param n_components:
        :param init:
        :param random_state:
        :param verbose:
        :return:
        """

        # Create instance of NMF (decomposition) model
        self.model = NMF(n_components=n_components, init=init, random_state=random_state, verbose=verbose)

        # Make note of time starting to train model
        start = datetime.now()

        # Fit transform
        self.W = sparse.lil_matrix(self.model.fit_transform(self.X))

        # Trained model time
        print(f"Fit model - took: {datetime.now() - start}")

        # Return the 'user' matrix - n components by Y users
        self.H = sparse.lil_matrix(self.model.components_)

    def approx_data(self):
        """ Using the weights - calculate approximation of original data """

        start = datetime.now()
        self.estimated_sales = self.model.inverse_transform(sparse.lil_matrix(self.W))
        print(f"Processed estimated sales based on W x H. Took {datetime.now() - start}")

    def estimate_sales(self, idx):
        """

        Returns vector of length matching number of objects for a single user

        :param idx: Integer or list of integers representing index locations to calculate
        :return:
        """
        start = datetime.now()
        est_sales = sparse.lil_matrix(self.W.dot(self.H[:,[idx]]))

        print(f"Processed estimated sales based on W x H. Took {datetime.now()-start}")
        return est_sales

    def save_model(self):
        pass

if __name__ == "__main__":

    # Create instance of model with datasets
    nmf = nmf_model('article_id', 'customer_id', nsample)

    # Create instance of nmf model and train it
    nmf.create_model(10, 'random', rseed, 10)

    ## Approximate date using full matrix
    #nmf.approx_data()

    # Pickle nmf model
    with open('nmf_model.pkl', 'wb') as f: pickle.dump(nmf, f)

