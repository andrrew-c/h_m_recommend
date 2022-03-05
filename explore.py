import pandas as pd
import altair as alt
import altair_viewer


from scipy import sparse



from data_explore import load_datasets, histogram

# Load up datasets
df_articles, df_customers, df_train = load_datasets()

# Price distribution
chart = histogram(df_train, 'price', col='sales_channel_id')
#chart = histogram(df_train, 'price')
altair_viewer.display(chart)

def get_counts(objects, object_key, users, users_key, sales):

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
    # Summarise sales (counts)
    sales_summary = sales.groupby([object_key, users_key]).size().reset_index()

    # Get index position for rows and columns
    sales_summary = sales_summary.merge(objects[object_key].reset_index(), on=object_key, how='left')
    sales_summary.rename(columns={'index': 'row_idx'}, inplace=True)

    sales_summary = sales_summary.merge(users[users_key].reset_index(), on=users_key, how='left')
    sales_summary.rename(columns={'index': 'col_idx'}, inplace=True)

    return sales_summary

def create_X_matrix(objects, object_key, users, users_key, sales):

    """

    Initialise a sparse matrix and populate the cells with the volume of transactions made per object/customer

    :param objects:
    :param object_key:
    :param users:
    :param users_key:
    :param sales:
    :return:
    """

    """ Objects: Rows of matrix - representing the count of times the article was purchases
        users: Columns of matrix - representing each customer

        NOTE - 2022-03-04 - The original datraframe was going ot have >1m column and 105k rows.
                May better to try using a sparse dataframe.
    """

    # Create a sparse matrix of shape 'articles' by 'customers'
    X_ = sparse.csr_matrix((objects.shape[0], users.shape[0]))



a = create_X_matrix(df_articles, 'article_id', df_customers, 'customer_id', df_train)




def plot_sales_vols(X):

    """ Line chart providing volumes of sales by channel and date """
    chart = alt.Chart(X.sample(5000)).mark_line().encode(
        x=alt.X('t_dat:T'),
        y=alt.Y('count()'),
        color='sales_channel_id:N'

    ).properties(
        width=1000,
        height=300
    )

    # Display the chart
    altair_viewer.display(chart)
s