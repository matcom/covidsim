#%%
import json

with open("covid19-cuba.json") as fp:
    data = json.load(fp)

# %%
data: dict = data['casos']['dias']

# %%
diagnosticados = []

for d,v in data.items():
    diagnosticados.append(len(v.get('diagnosticados', [])))

diagnosticados_total = [diagnosticados[0]]

for v in diagnosticados[1:]:
    diagnosticados_total.append(diagnosticados_total[-1] + v)

# %%
recuperados = []

for d,v in data.items():
    recuperados.append(v.get('recuperados_numero', 0))

recuperados_total = [recuperados[0]]

for v in recuperados[1:]:
    recuperados_total.append(recuperados_total[-1] + v)

# %%
muertes = []

for d,v in data.items():
    muertes.append(v.get('muertes_numero', 0))

muertes_total = [muertes[0]]

for v in muertes[1:]:
    muertes_total.append(muertes_total[-1] + v)

# %%
casos_activos = [
    diagnosticados_total[i] - recuperados_total[i] - muertes_total[i] 
    for i in range(len(diagnosticados_total))
]

# %%
with open("cuba.json", "w") as fp:
    json.dump({
        "Contagiado": casos_activos,
        "Recuperado": recuperados_total,
        "Fallecido": muertes_total
    }, fp, indent=2)

# %%
