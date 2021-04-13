import streamlit as st
import pandas as pd
import json
from pathlib import Path
import datetime

st.set_page_config(page_title="COVID Simulator", page_icon="😷", layout='wide', initial_sidebar_state='auto')


from src import data
from src.interventions import INTERVENTIONS
from src.simulation import JsonCallback, MultiCallback, Simulation, SimulationParameters, Region, TransitionEstimator, StateMachine, StreamlitCallback, VaccinationParameters
from src.estimation import estimate_parameter


def main():
    main_container, side_container = st.beta_columns([3,1])

    available_params = [f.name for f in (Path(__file__).parent / "params").glob("*.json")]

    default_values = {}

    if available_params:
        params_name = st.sidebar.selectbox("💾 Parámetros", available_params)
        
        with open(f"/src/params/{params_name}") as fp:
            default_values = json.load(fp)        

    parameters = SimulationParameters(
        days=st.sidebar.number_input("📆 Días a simular", 1, 1000, default_values.get("days", 90)),
        total_population=st.sidebar.number_input("🙆 Población total", value=default_values.get("total_population", 1000)),
        foreigner_arrivals=st.sidebar.number_input("✈️ Llegada de extranjeros", 0, 10000, default_values.get("foreigner_arrivals", 10)),
        chance_of_infection=st.sidebar.number_input("🤢 Probabilidad de infección", 0.0, 1.0, default_values.get("change_of_infection", 0.01)),
        initial_infected=st.sidebar.number_input("🤒 Infectados iniciales", 0, 1000, default_values.get("initial_infected", 0)),
        working_population=st.sidebar.slider("🧑‍🏭 Población laboral", 0.0, 1.0, default_values.get("working_population", 0.25)),
    )

    with side_container:
        with st.beta_expander("💉 Vacunación", True):
            vaccination_data = default_values.get("vaccines", [])
            vaccination_list = []
            vaccination_total = st.number_input("Vacunas", 0, 10, len(vaccination_data))

            for i in range(vaccination_total):
                if i < len(vaccination_data):
                    vaccination_params = vaccination_data[i]
                else:
                    vaccination_params = {}

                vaccination = VaccinationParameters(
                    start_day=st.slider("📆 Inicio", 0, parameters.days, value=vaccination_params.get("start_day", 0), key=f"vaccination{i}_start"),    
                    name=st.text_input("🏷️ Nombre", value=vaccination_params.get("name", f"Vacuna {i+1}"), key=f"vaccination_{i}_name"),
                    shots=st.number_input("🧴 Dosis", value=vaccination_params.get("shots", 1), key=f"vaccination_{i}_shots"),
                    shots_every=st.number_input("⌛ Dosis", value=vaccination_params.get("shots_every", 10), key=f"vaccination_{i}_shots_every"),
                    immunity_growth=st.number_input("📈 Crecimiento immunidad", 0, value=vaccination_params.get("immunity_growth", 15), key=f"vaccination{i}_growth"),
                    immunity_last=st.number_input("📉 Duración immunidad", 0, value=vaccination_params.get("immunity_last", 180), key=f"vaccination{i}_last"),
                    vaccinated_per_day=st.number_input("💉 Vacunados diarios", 0, value=vaccination_params.get("vaccinated_per_day", 100), key=f"vaccination{i}_per_day"),
                    maximum_immunity=st.slider("💖 Máxima immunidad", 0.0, 1.0, vaccination_params.get("maximum_immunity", 0.9), key=f"vaccination{i}_immunity"),
                    age_bracket=(st.number_input("👶 Edad mínima", 0,90,vaccination_params.get("age_bracket", [20,70])[0], step=5, key=f"vaccination{i}_age_min"),st.number_input("👴 Edad máxima", 0,90,vaccination_params.get("age_bracket", [20,70])[1], step=5, key=f"vaccination{i}_age_max"))
                )
                vaccination_list.append(vaccination)

        with st.beta_expander("⚕️ Intervenciones"):
            interventions = []
            total_interventions = st.number_input("Total de intervenciones a aplicar en el período", 0, 100, 0)
            interventions_names = {} #{"-": None}
            for cls in INTERVENTIONS:
                interventions_names[cls.description()] = cls

            for i in range(total_interventions):
                cls = interventions_names[st.selectbox("Tipo de intervención", list(interventions_names), key=f"intervention_{i}_type")]
                
                if cls is None:
                    continue

                interventions.append(cls.build(i))   

        with st.beta_expander("🧑‍🤝‍🧑 Matrices de contacto"):
            contact = data.load_contact_matrices()
            st.write(contact)

        with st.beta_expander("⚙️ Máquina de Estados"):
            state_machine = StateMachine()
            st.write(state_machine.states)

        with st.beta_expander("🔀 Transiciones"):
            transitions = TransitionEstimator()
            st.write(transitions.data)

        with st.beta_expander("⚗️ Estimar transiciones"):
            model = st.selectbox("Modelo", ["MultinomialNB", "LogisticRegression"])

            if st.button("Estimar"):
                real_data = data.load_real_data()
                st.write(real_data)

                processed_data = data.process_events(real_data)
                st.write(processed_data)

                df = data.estimate_transitions(processed_data, model)
                st.write(df)

                df.to_csv(Path(__file__).parent / "data" / "transitions.csv", index=False)

                st.success("💾 Data was saved to `data/transitions.csv`. Clear cache and reload.")

        with st.beta_expander("🔮 Estimar hiper-parámetros"):
            history_data = st.selectbox("Historial", [f.name for f in (Path(__file__).parent / "curves").glob("*.json")])

            with open(f"/src/curves/{history_data}") as fp:
                history = json.load(fp)
                st.line_chart(history, height=200)

            kwargs = dict(
                x_min=st.number_input("x_min", value=0.0),
                x_max=st.number_input("x_max", value=1.0),
                start_day=st.number_input("start_day", value=0),
                end_day=st.number_input("end_day", value=90),
                steps=st.number_input("steps", value=10)
            )

            if st.button("🧙‍♂️ Estimar"):
                def simulation_factory(p):
                    region = Region(1000, state_machine, parameters.initial_infected)
                    return Simulation([region], contact, p, transitions, state_machine, interventions)

                estimate_parameter("chance_of_infection", history, parameters, simulation_factory, **kwargs)

    with st.sidebar.beta_expander("Salvar parámetros"):
        save_params_as = st.text_input("Salvar parámetros (nombre)")
        params = dict(parameters.__dict__)
        params["vaccines"] = [v.__dict__ for v in vaccination_list]
        
        if st.button("💾 Salvar") and save_params_as:
            with open(Path(__file__).parent / "params" / (save_params_as + ".json"), "w") as fp:
                json.dump(params, fp, indent=4)

            st.success(f"🥳 Parámetros salvados en `params/{save_params_as}.json`")


    with main_container:
        if st.button("🚀 Simular"):
            region = Region(parameters.total_population, state_machine, parameters.initial_infected)
            sim = Simulation([region], contact, parameters, vaccination_list, transitions, state_machine, interventions)

            with open(f"logs/simulation_{datetime.datetime.now()}.jsonl", "w") as fp:
                sim.run(MultiCallback([StreamlitCallback(), JsonCallback(fp)]))
        else:
            st.info("Presione el botón **🚀 Simular** para ejecutar la simulación con los parámetros actuales.")
            st.write(params)


if __name__ == "__main__":
    main()
