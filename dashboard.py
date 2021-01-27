import streamlit as st

from src import data
from src.simulation import Simulation


st.set_page_config(page_title="COVID Simulator", page_icon="😷", layout='wide', initial_sidebar_state='auto')


def main():
    main_container, side_container = st.beta_columns([2,1])

    with side_container:
        with st.spinner("Cargando matrices de contacto"):
            contact = data.load_contact_matrices()
    
        with st.beta_expander("Matrices de contacto"):
            st.write(contact)

    total_days = st.sidebar.number_input("Días a simular", 1, 1000, 90)

    sim = Simulation(total_days)
    sim.run()


if __name__ == "__main__":
    main()
