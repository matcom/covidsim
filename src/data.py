import streamlit as st
import pandas as pd

from pathlib import Path


@st.cache
def load_contact_matrices():
    data = pd.read_csv(
        Path(__file__).parent.parent / "data" / "contact.csv",
        names=["country", "type", "location", "age_subject", "age_other", "value"]
    ).drop(columns=['country'])

    return data
