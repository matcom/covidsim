import streamlit as st
import pandas as pd

from pathlib import Path


@st.cache
def load_contact_matrices():
    data = pd.read_csv(
        Path(__file__).parent.parent / "data" / "contact.csv",
        names=["country", "type", "location", "age_subject", "age_other", "value"]
    )

    # quedarse con el rango inicial de la edad
    data["subject"] = data["age_subject"].transform(lambda s: int(s.split(" to ")[0].strip("+")))
    data["other"] = data["age_other"].transform(lambda s: int(s.split(" to ")[0].strip("+")))

    return data.drop(columns=['country', "age_subject", "age_other"])


@st.cache
def load_disease_transition():
    return pd.read_csv("./data/transitions.csv")
    

@st.cache
def load_states():
    return pd.read_csv("./data/states.csv").set_index("label").to_dict("index")
