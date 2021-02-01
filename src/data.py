import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

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


@st.cache
def load_real_data():
    return pd.read_csv("./data/data.csv")


@st.cache
def process_events(df: pd.DataFrame):
    events = []

    for i, row in df.iterrows():
        # Si es Viajero, tendra 'Fecha Arribo' distinta de nan
        arrival = pd.to_datetime(row['Fecha Arribo'], errors='coerce')
        detected = pd.to_datetime(row['FI'], errors='coerce')
        symptoms = pd.to_datetime(row['FIS'], errors='coerce')

        if pd.isna(detected):
            continue

        if not pd.isna(arrival):
            # es viajero
            days = (detected - arrival).days

            if days < 0:
                continue

            # Aquí es detectado
            events.append(
                dict(id=row["Cons"], from_state="Viajero", to_state="Detectado", days=days, age=row['Edad'], sex=row["Sexo"])
            )
        elif not pd.isna(symptoms):
            # es contagiado interno
            days = (detected - symptoms).days

            if days < 0:
                continue

            # Aquí es detectado
            events.append(
                dict(id=row["Cons"], from_state="Contagiado", to_state="Detectado", days=days, age=row['Edad'], sex=row["Sexo"])
            )

        # Aquí se decide si es asintomático o sintomático
        # Como los datos están sucios, a veces dice 'asint' o 'Asintomático', con o sin tilde
        if isinstance(row['FIS'], str) and row['FIS'].lower().startswith("asint"):
            to_state = "Detec.Asint"
        else:
            to_state = "Detec.Sint"
        
        events.append(
            dict(id=row["Cons"], from_state="Detectado", to_state=to_state, days=0, age=row['Edad'], sex=row["Sexo"])
        )

        # Aquí se decide el resultado final
        release = pd.to_datetime(row["Fecha Alta"], errors='coerce')
        recovered = row['Evolución'] == 'Alta'
        dead = row['Evolución'] == 'Fallecido'

        if not pd.isna(release):
            days = (release - detected).days

            if recovered:
                events.append(
                    dict(id=row["Cons"], from_state=to_state, to_state="Recuperado", days=days, age=row['Edad'], sex=row["Sexo"])
                )
            elif dead:
                events.append(
                    dict(id=row["Cons"], from_state=to_state, to_state="Fallecido", days=days, age=row['Edad'], sex=row["Sexo"])
                )

    return pd.DataFrame(events)


def estimate_transitions(events: pd.DataFrame):
    by_state = events.groupby("from_state")

    table = []

    for key, group in by_state:
        features = group[["age", "sex"]].to_dict('record')
        target = group["to_state"]

        try:
            ml = Pipeline(steps=[("vectorizer", DictVectorizer()), ('classifier', LogisticRegression())])
            ml.fit(features, target)
        except:
            continue

        for age in range(0,80,5):
            for sex in ["M", "F"]:
                X = dict(age=age, sex=sex)
                y = ml.predict_proba(X)[0]
                for prob, to_state in zip(y, ml.steps[-1][1].classes_):
                    table.append(dict(
                        age=age, sex=sex, from_state=key, to_state=to_state, change=prob
                    ))

    st.write(pd.DataFrame(table))
