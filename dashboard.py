import streamlit as st
import pandas as pd
import json
from pathlib import Path


st.set_page_config(page_title="COVID Simulator", page_icon="😷", layout='wide', initial_sidebar_state='auto')


from src import data
from src.interventions import INTERVENTIONS
from src.simulation import Simulation, SimulationParameters, Region, TransitionEstimator, StateMachine, StreamlitCallback
from src.estimation import estimate_parameter


def main():
    main_container, side_container = st.beta_columns([2,1])

    available_params = [f.name for f in (Path(__file__).parent / "params").glob("*.json")]

    default_values = {}

    if available_params:
        params_name = st.sidebar.selectbox("💾 Parámetros", available_params)
        
        with open(f"/src/params/{params_name}") as fp:
            default_values = json.load(fp)        

    parameters = SimulationParameters(
        days=st.sidebar.number_input("📆 Días a simular", 1, 1000, default_values.get("days", 90)),
        foreigner_arrivals=st.sidebar.number_input("✈️ Llegada de extranjeros", 0, 10000, default_values.get("foreigner_arrivals", 10)),
        chance_of_infection=st.sidebar.number_input("🤢 Probabilidad de infección", 0.0, 1.0, default_values.get("change_of_infection", 0.01)),
        initial_infected=st.sidebar.number_input("🤒 Infectados iniciales", 0, 1000, default_values.get("initial_infected", 0)),
        working_population=st.sidebar.slider("🧑‍🏭 Población laboral", 0.0, 1.0, default_values.get("working_population", 0.25)),
    )

    with st.sidebar.beta_expander("Salvar parámetros"):
        save_params_as = st.text_input("Salvar parámetros (nombre)")
        
        if st.button("💾 Salvar") and save_params_as:
            with open(Path(__file__).parent / "params" / (save_params_as + ".json"), "w") as fp:
                json.dump(parameters.__dict__, fp, indent=4)

            st.success(f"🥳 Parámetros salvados en `params/{save_params_as}.json`")

    with side_container:
        with st.beta_expander("💉 Intervenciones"):
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
                    region = Region(1000, transitions, state_machine, parameters.initial_infected)
                    return Simulation([region], contact, p, transitions, state_machine, interventions)

                estimate_parameter("chance_of_infection", history, parameters, simulation_factory, **kwargs)

    with main_container:
        region = Region(1000, state_machine, parameters.initial_infected)
        sim = Simulation([region], contact, parameters, transitions, state_machine, interventions)

        if st.button("🚀 Simular"):
            sim.run(StreamlitCallback())
        else:
            st.info("Presione el botón `Simular` para ejecutar la simulación con los parámetros actuales.")
            st.write(parameters.__dict__)


if __name__ == "__main__":
    main()
