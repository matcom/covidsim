import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import random
import networkx as nx
import collections
import pydot

from .data import load_cuba_data, load_interaction_estimates


@st.cache
def get_events(data):
    events = []

    for i, d in data.iterrows():
        person_id = d["Cons"]

        try:
            age = int(d["Edad"])
            sex = "FEMALE" if d["Sexo"] == "F" else "MALE"
        except ValueError:
            continue

        farr = d["Fecha Arribo"]

        state = "L" if pd.isna(farr) else "F"

        if d["Asintomatico"]:
            events.append(
                dict(
                    from_state=state,
                    to_state="A",
                    duration=0,
                    age=age,
                    sex=sex,
                    id=person_id,
                )
            )

            if d["Evolución"] == "Alta":
                events.append(
                    dict(
                        from_state="A",
                        to_state="R",
                        duration=(
                            pd.to_datetime(d["Fecha Alta"])
                            - pd.to_datetime(d["F. Conf"])
                        ).days,
                        age=age,
                        sex=sex,
                        id=person_id,
                    )
                )

            continue

        try:
            symptoms_start = pd.to_datetime(d["FIS"], format="%m/%d/%Y", errors="raise")
            admission_start = pd.to_datetime(d["FI"], format="%m/%d/%Y", errors="raise")
        except:
            continue

        events.append(
            dict(
                from_state=state,
                to_state="I",
                duration=0 if state == "L" else (symptoms_start - farr).days,
                age=age,
                sex=sex,
                id=person_id,
            )
        )

        events.append(
            dict(
                from_state="I",
                to_state="H",
                duration=(admission_start - symptoms_start).days,
                age=age,
                sex=sex,
                id=person_id,
            )
        )

        try:
            alta = pd.to_datetime(d["Fecha Alta"], format="%m/%d/%Y", errors="raise")
        except:
            continue

        if d["Evolución"] == "Fallecido":
            events.append(
                dict(
                    from_state="H",
                    to_state="D",
                    duration=(alta - admission_start).days,
                    age=age,
                    sex=sex,
                    id=person_id,
                )
            )
        elif d["Evolución"] == "Alta":
            events.append(
                dict(
                    from_state="H",
                    to_state="R",
                    duration=(alta - admission_start).days,
                    age=age,
                    sex=sex,
                    id=person_id,
                )
            )

    return pd.DataFrame(events)


@st.cache
def get_daily_values(data, asympt_length):
    day_states = []
    fend = data["Fecha Alta"].max() + pd.Timedelta(days=1)

    interactions = load_interaction_estimates("general")
    print(interactions)

    for i, row in data.iterrows():
        fe: pd.Timestamp = row["Fecha Arribo"]
        fs: pd.Timestamp = row["FIS"]
        fi: pd.Timestamp = row["FI"]
        fc: pd.Timestamp = row["F. Conf"]
        fa: pd.Timestamp = row["Fecha Alta"]

        try:
            age = max(5, min(80, (int(row['Edad']) // 5)  * 5))
            contacts = sum(interactions[age].values())
        except ValueError:
            continue

        if pd.isna(fa):
            fa = fend

        if pd.isna(fc):
            continue

        if not pd.isna(fe):
            day_states.append(dict(day=fe, id=row["Cons"], status="nuevo-extranjero"))

        day_states.append(dict(day=fc, id=row["Cons"], status="nuevo-confirmado"))

        if fa < fend:
            day_states.append(dict(day=fa, id=row["Cons"], status="nuevo-alta"))

        for day in range((fa - fc).days):
            day_states.append(
                dict(day=fc + pd.Timedelta(days=day), id=row["Cons"], status="activo")
            )

        if not pd.isna(fi):
            day_states.append(dict(day=fi, id=row["Cons"], status="nuevo-ingreso"))

            for day in range((fa - fi).days):
                day_states.append(
                    dict(
                        day=fi + pd.Timedelta(days=day),
                        id=row["Cons"],
                        status="ingresado",
                    )
                )

        if not pd.isna(fs):
            day_states.append(dict(day=fs, id=row["Cons"], status="nuevo-síntoma"))

            if pd.isna(fe):
                day_states.append(
                    dict(
                        day=fs - pd.Timedelta(days=random.randint(0, asympt_length)),
                        id=row["Cons"],
                        status="infectado",
                    )
                )

            for day in range((fc - fs).days):
                day_states.append(
                    dict(
                        day=fs + pd.Timedelta(days=day),
                        id=row["Cons"],
                        status="infeccioso",
                    )
                )
                day_states.append(
                    dict(
                        day=fs + pd.Timedelta(days=day),
                        id=row["Cons"],
                        status="contacto",
                        value=contacts
                    )
                )
        else:
            day_states.append(
                dict(
                    day=fc - pd.Timedelta(days=random.randint(0, asympt_length)),
                    id=row["Cons"],
                    status="infectado",
                )
            )

        # if contacts == 0:
        #     continue

        # if row["Asintomatico"]:
        #     for day in range(asympt_length):
        #         day_states.append(
        #             dict(
        #                 day=fc - pd.Timedelta(days=day),
        #                 id=row["Cons"],
        #                 status="contacto",
        #                 value=contacts / asympt_length,
        #             )
        #         )
        # else:
        #     if pd.isna(fi) or pd.isna(fs):
        #         continue

        #     total_days = asympt_length + (fi - fs).days

        #     for day in range(total_days):
        #         day_states.append(
        #             dict(
        #                 day=fi - pd.Timedelta(days=day),
        #                 id=row["Cons"],
        #                 status="contacto",
        #                 value=contacts / total_days,
        #             )
        #         )

    return pd.DataFrame(day_states).fillna(1)


def run():
    data = load_cuba_data()

    asympt_length = st.sidebar.number_input(
        "Días promedio de desarrollar síntomas", 0, 100, 5
    )

    if st.checkbox("Ver datos raw"):
        st.write(data)

    transitions(data)

    day_states = get_daily_values(data, asympt_length)

    if st.checkbox("Ver eventos"):
        st.write(day_states)

    st.write("### Infecciosos vs. infectados diarios")

    infected = day_states[day_states['status'].isin(['infectado', 'infeccioso', 'contacto'])]
    infected = infected.groupby(['day', 'status']).agg(count=('value', 'sum')).reset_index()
    infected = infected.pivot(index='day', columns='status', values='count').reset_index().fillna(0)
    infected = infected.loc[infected['infeccioso'] > 0]
    infected['rate'] = infected['infectado'] / infected['contacto']
    st.write(infected)

    infected = infected.loc[infected['rate'] < 1.0]

    st.altair_chart(
        alt.Chart(infected).mark_line().encode(x=alt.X("day", title='Date'), y=alt.Y("rate", title="Estimated infection rate")),
        use_container_width=True,
    )

    foreigners = day_states[day_states["status"] == "nuevo-extranjero"]

    st.write("### Rate de llegada de extranjeros diaria")

    st.altair_chart(
        alt.Chart(foreigners).mark_bar().encode(x=alt.X("day", title="Date"), y=alt.Y("count()", title="Infected foreigner arrivals")),
        use_container_width=True,
    )


def transitions(data: pd.DataFrame):
    st.write("## Estimación de transiciones para simulación")

    df = get_events(data)

    if st.checkbox("Ver todos los eventos"):
        st.write(df)

    st.write("### Transiciones generales")

    def compute_transitions(df):
        df = (
            df.groupby(["from_state", "to_state"])
            .agg(
                duration_mean=("duration", "mean"),
                duration_std=("duration", "std"),
                count=("id", "count"),
            )
            .fillna(0)
            .reset_index()
        )

        df["freq"] = df.apply(
            lambda r: r["count"]
            / df[df["from_state"] == r["from_state"]]["count"].sum(),
            axis="columns",
        )

        return df

    states = compute_transitions(df)
    st.write(states)

    if st.checkbox('Ver en LaTeX'):
        st.code(states.to_latex())

    graph = pydot.Graph(rankdir="LR")

    for _, row in states.iterrows():
        graph.add_edge(
            pydot.Edge(
                src=row["from_state"],
                dst=row["to_state"],
                label="%.2f (%.1f ~ %.1f)"
                % (row["freq"], row["duration_mean"], row["duration_std"]),
            )
        )

    st.graphviz_chart(str(graph))

    if st.checkbox("Ver transiciones en CSV"):
        csv = []

        for age in range(0, 85, 5):
            for sex in ["FEMALE", "MALE"]:
                df_filter = df[
                    (df["age"] >= age) & (df["age"] <= age * 5) & (df["sex"] == sex)
                ]

                if len(df_filter) > 0:
                    states = compute_transitions(df_filter)
                    states["age"] = age
                    states["sex"] = sex
                    csv.append(states.set_index(["age", "sex", "from_state"]).round(3))

        df = pd.concat(csv)
        st.code(df.to_csv())
