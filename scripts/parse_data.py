# %%
from datetime import datetime, timedelta
from pathlib import Path
import random
import pandas as pd

# %%
data = pd.read_csv("../data/base2020.csv", sep=";")

# %%
def report(state, date, last_date, last_state, age, sex):
    if last_state is not None:
        events.append(dict(
            from_state=last_state,
            to_state=state,
            age=age,
            sex=sex,
            days=(date - last_date).days if not pd.isna(last_date) else 0,
        ))

    return date, state

# %%
events = []

for i,r in data.iterrows():
    # NOTA: NO HAY SEXO EN LOS DATOS!!!!
    sex = random.choice(["M", "F"])
    age = r["Edad2020"]
    date = pd.NaT
    state = None

    symptoms_date = pd.to_datetime(r['FIS2020'], format="%m/%d/%Y", errors="coerce")
    hospital_date = pd.to_datetime(r['Fingreso2020'], format="%m/%d/%Y", errors="coerce")
    confirm_date = pd.to_datetime(r['F.Conf2020'], format="%m/%d/%Y", errors="coerce")
    uci_enter_date = pd.to_datetime(r['FechaingresoUCI3112'], format="%m/%d/%Y", errors="coerce")
    uci_exit_date = pd.to_datetime(r['FechaegresoUTI'], format="%m/%d/%Y", errors="coerce")
    release_date = pd.to_datetime(r['FechaAltaN'], format="%m/%d/%Y", errors="coerce")

    if pd.isna(confirm_date) or pd.isna(release_date):
        # Si estas fechas no se conocen, pues entonces no se tiene datos suficientes
        continue

    # Si es fuente de infección en el exterior, entonces entra como viajero
    # a la simulación
    if r["Fuente2020"] == "Exterior":
        arrival_date = pd.to_datetime(r['FechaArribo2020'], format="%m/%d/%Y", errors="coerce")
        date, state = report("Viajero", arrival_date, date, state, age, sex)

    # Aquí se contagia

    if pd.isna(symptoms_date):
        # Es asintomático, asumamos que se contagia entre 5 y 10 días antes de la fecha de confirmación
        contagion_date = confirm_date - timedelta(days=random.randint(5, 10))
        date, state = report("Contagiado", contagion_date, date, state, age, sex)
        date, state = report("Asintomático", contagion_date, date, state, age, sex)
    else:
        # Es sintomático, los síntomas surgen en la fecha indicada,
        # asumamos que se contagió de 1 a 5 días antes
        date, state = report("Contagiado", symptoms_date - timedelta(days=random.randint(1,5)), date, state, age, sex)
        date, state = report("Sintomático", symptoms_date, date, state, age, sex)

    if not pd.isna(hospital_date):
        date, state = report("Hospitalizado", hospital_date, date, state, age, sex)

    if not pd.isna(uci_enter_date):
        date, state = report("UCI", uci_enter_date, date, state, age, sex)

    if not pd.isna(uci_exit_date):
        date, state = report("Hospitalizado", uci_exit_date, date, state, age, sex)

    if r["Evolucion311220"] == "Fallecido":
        date, state = report("Fallecido", release_date, date, state, age, sex)
    else:
        date, state = report("Recuperado", release_date, date, state, age, sex)

print(len(events))

# %%
events = pd.DataFrame(events)
events.to_csv("../data/events.cvs", index=False)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor


by_state = events.groupby("from_state")

table = []

model = MultinomialNB()

for key, group in by_state:
    print(key, len(group))

    features = group[["age", "sex"]].to_dict('record')
    target = group["to_state"]

    ml = Pipeline(steps=[("vectorizer", DictVectorizer()), ('classifier', model)])
    ml.fit(features, target)
    print("Fit ML")

    gauss = Pipeline(steps=[("vectorizer", DictVectorizer(sparse=False)), ('regressor', GaussianProcessRegressor())])
    target = group["days"]
    gauss.fit(features, target)
    print("Fit Gauss")

    for age in range(0,100,5):
        for sex in ["M", "F"]:
            X = dict(age=age, sex=sex)
            y = ml.predict_proba(X)[0]
            days, stdev = gauss.predict(X, return_std=True)
            for prob, to_state in zip(y, ml.steps[-1][1].classes_):
                table.append(dict(
                    age=age, sex=sex, from_state=key, to_state=to_state, chance=prob, mean_days=days[0], std_days=stdev[0]
                ))
                print(dict(
                    age=age, sex=sex, from_state=key, to_state=to_state, chance=prob, mean_days=days[0], std_days=stdev[0]
                ))

table = pd.DataFrame(table)
table.to_csv("../data/transitions.csv", index=False)
