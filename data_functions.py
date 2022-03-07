import pandas as pd
from scipy import sparse

import altair as alt

from constants import rseed
from project_secrets import data_path

def load_datasets(nsample = None):

    print("Loading datasets")
    # What does a sample submission look like?
    df_sample_sub = pd.read_csv(f"{data_path}/sample_submission.csv")

    # Data on the articles
    df_articles = pd.read_csv(f"{data_path}/articles.csv")
    print(f"Loaded df_articles with shape {df_articles.shape}")

    # Customers
    df_customers = pd.read_csv(f"{data_path}/customers.csv")
    print(f"Loaded df_customers with shape {df_customers.shape}")

    # Training data
    df_train = pd.read_csv(f"{data_path}/transactions_train.csv", nrows=nsample)
    print(f"Loaded training data where nsample = {nsample} - df_train shape: {df_train.shape}")

    return df_articles, df_customers, df_train


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


def histogram(df, var, col):

    """ Use Altair to create a histogram """
    alt.data_transformers.disable_max_rows()


    chart = alt.Chart(df).mark_bar(opacity=0.4).encode(
        alt.X(f'{var}:Q', bin=True),
        y='count()',
        color=col
    )

    # Turn this back on
    # alt.data_transformers.enable_max_rows()
