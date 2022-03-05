import pandas as pd
import altair as alt

from project_secrets import data_path

def load_datasets(nsample = None):
    # What does a sample submission look like?
    df_sample_sub = pd.read_csv(f"{data_path}/sample_submission.csv")

    # Data on the articles
    df_articles = pd.read_csv(f"{data_path}/articles.csv")

    # Customers
    df_customers = pd.read_csv(f"{data_path}/customers.csv")

    # Training data
    df_train = pd.read_csv(f"{data_path}/transactions_train.csv", nrows=nsample)

    return df_articles, df_customers, df_train

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
