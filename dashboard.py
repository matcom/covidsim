import streamlit as st

st.set_page_config(page_title="COVID Simulator", page_icon="😷", layout='wide', initial_sidebar_state='auto')

from src import data
from src.simulation import Simulation, SimulationParameters, Region, TransitionEstimator, StateMachine


def main():
    main_container, side_container = st.beta_columns([2,1])

    with side_container:
        with st.spinner("Cargando matrices de contacto"):
            contact = data.load_contact_matrices()
    
        with st.beta_expander("Matrices de contacto"):
            st.write(contact)

        with st.beta_expander("Máquina de Estados"):
            state_machine = StateMachine()
            st.write(state_machine.states)

        with st.beta_expander("Transiciones"):
            transitions = TransitionEstimator()
            st.write(transitions.data)

        with st.beta_expander("Datos reales"):
            real_data = data.load_real_data()
            st.write(real_data)

            processed_data = data.process_events(real_data)
            st.write(processed_data)

        with st.beta_expander("Estimar transiciones"):
            if st.button("Estimar"):
                data.estimate_transitions(processed_data)


    parameters = SimulationParameters(
        days=st.sidebar.number_input("Días a simular", 1, 1000, 90),
        foreigner_arrivals=st.sidebar.number_input("Llegada de extranjeros", 0, 10000, 10),
        chance_of_infection=st.sidebar.number_input("Probabilidad de infección", 0.0, 1.0, 0.01),
        initial_infected=st.sidebar.number_input("Infectados iniciales", 0, 1000, 0),
    )

    with main_container:
        region = Region(1000, transitions, state_machine, parameters.initial_infected)

        sim = Simulation([region], contact, parameters, transitions, state_machine, main_container)

        if st.button("🚀 Simular"):
            sim.run()


if __name__ == "__main__":
    main()
