import streamlit as st

from src import data
from src.simulation import Simulation, SimulationParameters, Region, TRANSITIONS


st.set_page_config(page_title="COVID Simulator", page_icon="😷", layout='wide', initial_sidebar_state='auto')


def main():
    main_container, side_container = st.beta_columns([2,1])

    with side_container:
        with st.spinner("Cargando matrices de contacto"):
            contact = data.load_contact_matrices()
    
        with st.beta_expander("Matrices de contacto"):
            st.write(contact)

        with st.beta_expander("Transiciones"):
            st.write(TRANSITIONS.data)

    parameters = SimulationParameters(
        days=st.sidebar.number_input("Días a simular", 1, 1000, 90),
        foreigner_arrivals=st.sidebar.number_input("Llegada de extranjeros", 0.0, 10000.0, 100.0),
        chance_of_infection=st.sidebar.number_input("Probabilidad de infección", 0.0, 1.0, 0.1),
    )

    region = Region(1000, st.sidebar.number_input("Infectados iniciales", 1, 1000, 10))

    sim = Simulation([region], contact, parameters, main_container)

    if main_container.button("🚀 Simular"):
        sim.run()


if __name__ == "__main__":
    main()
