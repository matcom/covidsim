import collections
import random
from typing import Dict, List
from dataclasses import dataclass


import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


PARAMETERS = dict()


class InterventionsManager:
    def __init__(self):
        self._closed_borders = []
        self._testing = []
        self._school_open = []
        self._workforce = []
        self._social_distance = []
        self._use_masks = []
        self._reduce_aglomeration = []
        self._restriction_internal_movement = []
        self.day = 0

    def close_borders(self, start, end):
        """ Activa la medida de cerrar los aeropuertos
        """
        self._closed_borders.append((start, end))

    def is_airport_open(self):
        """ Informa si los aeropuertos están cerrados
        """
        for start, end in self._closed_borders:
            if self.day >= start and self.day <= end:
                return False

        return True

    def activate_testing(self, start, end, percent):
        """ Activa la medida de testiar un % de la población
        """
        self._testing.append((start, end, percent))

    def is_testing_active(self):
        """ Informa si la medida de testeo de personas está activa
        """

        for start, end, percent in self._testing:
            if self.day >= start and self.day <= end:
                return percent

        return 0.0

    def school_close(self, start, end):
        """ Activa la medida de cerrar las escuelas
        """
        self._school_open.append((start, end))

    def is_school_open(self):
        """ Informa si la medida de cerrar escuelasestá activa
        """

        for start, end in self._school_open:
            if self.day >= start and self.day <= end:
                return False

        return True

    def activate_workforce(self, start, end, percent):
        """ Activa la medida de activar el teletrabajo en un % de la población
        """
        self._workforce.append((start, end, percent))

    def is_workforce(self):
        """ Informa si la medida de activar el teletrabajo en un % de la población
        """

        for start, end, percent in self._workforce:
            if self.day >= start and self.day <= end:
                return percent

        return 1.0

    def activate_social_distance(self, start, end, percent):
        """ Activa la medida de activar el distanciamiento social
        """
        self._social_distance.append((start, end, percent))

    def is_social_distance(self):
        """ Informa si la medida de activar el distanciamiento social
        """

        for start, end, percent in self._social_distance:
            if self.day >= start and self.day <= end:
                return 1 - percent

        return 1.0

    def activate_use_masks(self, start, end):
        """ Activa la medida de utilizar mascarillas
        """
        self._use_masks.append((start, end))

    def is_use_masks(self):
        """ Informa si la medida de utilizar mascarillas
        """

        for start, end in self._use_masks:
            if self.day >= start and self.day <= end:
                return True

        return False

    def activate_reduce_aglomeration(self, start, end, percent):
        """ Activa la medida de no participar en aglomeraciones públicas
        """
        self._reduce_aglomeration.append((start, end, percent))

    def is_reduce_aglomeration(self):
        """ Informa si la medida de no participar en aglomeraciones públicas
        """

        for start, end, percent in self._reduce_aglomeration:
            if self.day >= start and self.day <= end:
                return percent

        return 1.0

    def activate_restriction_internal_movement(self, start, end, percent):
        """ Activa la medida de restricciones en el transporte interno
        """
        self._restriction_internal_movement.append((start, end, percent))

    def is_restriction_internal_movement(self):
        """ Informa si la medida de restricciones en el transporte interno
        """

        for start, end, percent in self._restriction_internal_movement:
            if self.day >= start and self.day <= end:
                return percent

        return 1.0


Interventions = InterventionsManager()


@st.cache
def load_disease_transition():
    return pd.read_csv("./data/transitions.csv")
    

class TransitionEstimator:
    def __init__(self):
        self.data = load_disease_transition()

    def transition(self, from_state, age, sex):
        age = (age // 5) * 5
        df = self.data[(self.data["Age"] == age) & (self.data["StateFrom"] == from_state) & (self.data["Sex"] == sex)]

        if len(df) == 0:
            raise ValueError(f"No transitions for {from_state}, age={age}, sex={sex}.")

        return pd.DataFrame(df).sort_values("Count")


class StatePerson:
    """Estados en los que puede estar una persona.
    """

    S = "S"
    L = "L"
    I = "I"
    A = "A"
    U = "U"
    R = "R"
    H = "H"
    D = "D"
    F = "F"


class Person:
    total = 0

    def __init__(self, region: "Region", age:int, sex:str, transitions: TransitionEstimator):
        """Crea una nueva persona que por defecto está en el estado de suseptible al virus.
        """
        Person.total += 1
        self.state = StatePerson.S
        self.next_state = None
        self.steps_remaining = None
        self.is_infectious = None
        self.transitions = transitions
        # llamar método de estado inicial
        self.set_state(StatePerson.S)

        # la persona conoce la region a la que pertenece
        self.region = region
        self.age = age
        self.sex = sex
        self.health_conditions = None

    def next_step(self):
        """Ejecuta un step de tiempo para una persona.
        """
        if self.steps_remaining == 0:
            # actualizar state
            self.state = self.next_state

            if self.state == StatePerson.L:
                self.p_latent()
            elif self.state == StatePerson.I:
                self.p_infect()
            elif self.state == StatePerson.A:
                self.p_asintomatic()
            elif self.state == StatePerson.H:
                self.p_hospitalized()
            elif self.state == StatePerson.U:
                self.p_uci()
            elif self.state == StatePerson.F:
                self.p_foreigner()
            else:
                return False
            # en los estados restantes no hay transiciones
        else:
            # decrementar los steps que faltan para cambiar de estado
            self.steps_remaining = self.steps_remaining - 1

        return True

    def __repr__(self):
        return f"Person(age={self.age}, sex={self.sex}, state={self.state}, steps_remaining={self.steps_remaining})"

    # Funciones de que pasa en cada estado para cada persona
    # debe devolver el tiempo que la persona va estar en ese estado y
    # a que estado le toca pasar en el futuro.

    def set_state(self, state):
        self.next_state = state
        self.steps_remaining = 0
        self.next_step()

    def _evaluate_transition(self):
        """Computa a qué estado pasar dado el estado actual y los valores de la tabla.
        """
        df = self.transitions.transition(self.state, self.age, self.sex)
        # calcular el estado de transición y el tiempo
        to_state = random.choices(df["StateTo"].values, weights=df["Count"].values, k=1)[
            0
        ]
        state_data = df.set_index("StateTo").to_dict("index")[to_state]
        time = random.normalvariate(state_data["MeanDays"], state_data["StdDays"])

        return to_state, int(time)

    def p_latent(self):
        self.is_infectious = False
        change_to_symptoms = 0.5

        if random.uniform(0, 1) < change_to_symptoms:
            self.next_state = StatePerson.A
            self.steps_remaining = 0
            return

        # convertirse en sintomático en (2,14) dias
        self.next_state = StatePerson.I
        self.steps_remaining = random.randint(2, 14)

    def p_foreigner(self):
        self.is_infectious = True
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_infect(self):
        self.is_infectious = True
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_asintomatic(self):
        self.is_infectious = True
        self.next_state = StatePerson.R
        # tiempo en que un asintomático se cura
        self.steps_remaining = random.randint(2, 14)

    def p_recovered(self):
        self.is_infectious = False
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_hospitalized(self):
        self.is_infectious = False
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_uci(self):
        self.is_infectious = False
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_death(self):
        self.is_infectious = False
        self.next_state, self.steps_remaining = self._evaluate_transition()


class Region:
    def __init__(self, population, transitions: TransitionEstimator, initial_infected=1):
        self._recovered = 0
        self._population = population
        self._death = 0
        self._simulations = 0
        self.transitions = transitions
        self._individuals = []

        for i in range(initial_infected):
            p = Person(self, random.randint(20, 80), random.choice(["MALE", "FEMALE"]), transitions)
            p.set_state(StatePerson.I)
            self._individuals.append(p)

    def __len__(self):
        return len(self._individuals)

    def __iter__(self):
        for i in list(self._individuals):
            if i.state != StatePerson.S:
                yield i

    def spawn(self, age) -> Person:
        p = Person(self, age, random.choice(["MALE", "FEMALE"]), self.transitions)
        self._individuals.append(p)
        return p


@dataclass
class SimulationParameters:
    days:int
    foreigner_arrivals:float
    chance_of_infection:float
    initial_infected:int


class Simulation:
    def __init__(self, regions:List[Region], contact, parameters:SimulationParameters, transitions: TransitionEstimator, container) -> None:
        self.regions = regions
        self.contact = contact
        self.parameters = parameters
        self.transitions = transitions
        self.container = container

    # método de tranmisión espacial, teniendo en cuenta la localidad
    def run(self):
        """
        Método de transmisión espacial, teniendo en cuenta la localidad.

        Args:
            - regions: regiones consideradas en la simulación.
            - social: es el conjunto de grafos que describen las redes en cada región.
            - status: contiene característucas y el estado de salud en cada región, osea las medidas.
            - distance: almacena la distancia entre cada par de regiones.
            - parameters: parameters del modelo epidemiológico para cada individuo.

        Returns:
            - output: es el estado actualizado de cada persona.
        """

        simulation_time = self.parameters.days

        # estadísticas de la simulación
        with self.container:
            progress = st.progress(0)
            progress_person = st.progress(0)
            day = st.empty()
            sick_count = st.empty()
            all_count = st.empty()

            chart = st.altair_chart(
                alt.Chart(pd.DataFrame(columns=["personas", "dia", "estado"]))
                .mark_line()
                .encode(
                    y=alt.Y("personas:Q", title="Individuals"),
                    x=alt.X("dia:Q", title="Days of simulation"),
                    color=alt.Color("estado:N", title="State"),
                ),
                use_container_width=True,
            )

        # por cada paso de la simulación
        for i in range(simulation_time):

            total_individuals = 0
            by_state = collections.defaultdict(lambda: 0)

            # por cada región
            for region in self.regions:
                # llegadas del estranjero
                self._simulate_arrivals(region)
                # por cada persona
                individuals = list(region)
                for j, ind in enumerate(individuals):
                    # actualizar estado de la persona
                    ind.next_step()
                    if ind.is_infectious:
                        self._simulate_spread(ind)

                    progress_person.progress((j+1) / len(individuals))
                    total_individuals += 1
                    by_state[ind.state] += 1

                self._apply_interventions(region)
                # movimientos
                for n_region in self.regions:
                    if n_region != region:
                        # calcular personas que se mueven de una region a otras
                        self._simulate_transportation(region, n_region)

            progress.progress((i + 1) / simulation_time)
            sick_count.markdown(f"#### Individuos simulados: {total_individuals}")
            all_count.code(dict(by_state))
            day.markdown(f"#### Día: {i+1}")

            chart.add_rows(
                [dict(dia=i + 1, personas=v, estado=k) for k, v in by_state.items()]
            )


    def _simulate_arrivals(self, region: Region):
        if Interventions.is_airport_open():
            people = np.random.poisson(self.parameters.foreigner_arrivals)

            for _ in range(people):
                p = region.spawn(random.randint(20, 80))
                p.set_state(StatePerson.F)


    def _apply_interventions(region, status):
        """Modifica el estado de las medidas y como influyen estas en la población.

        Los cambios se almacenan en status
        """
        # si se está testeando activamente
        p = Interventions.is_testing_active()

        if p > 0:
            for ind in region:
                if ind.state in [StatePerson.L, StatePerson.A] and random.uniform(0, 1) < p:
                    ind.set_state(StatePerson.H)


    def _simulate_transportation(n_region, region, distance):
        """Las personas que se mueven de n_region a region.
        """
        pass


    def _simulate_spread(self, ind):
        """Calcula que personas serán infectadas por 'ind' en el paso actual de la simulación.
        """
        connections = self._eval_connections(ind)

        for other in connections:
            if other.state != StatePerson.S:
                continue

            if self._eval_infections(ind):
                other.set_state(StatePerson.L)


    def _eval_connections(
        # social: Dict[str, Dict[int, Dict[int, float]]], person: "Person"
        self,
        person: Person
    ) -> List["Person"]:
        """Devuelve las conexiones que tuvo una persona en un step de la simulación.
        """

        the_age = person.age

        if the_age % 5 != 0:
            the_age = the_age // 5 * 5

        p_social = Interventions.is_social_distance()

        if random.uniform(0, 1) < Interventions.is_reduce_aglomeration():
            p_social *= (
                2  # Cada cierta cantidad de días, un día te toca ver el doble de personas
            )

        # quedarse con la submatrix de contacto de la edad correspondiente
        social = self.contact[(self.contact['type']=='overall') & (self.contact['subject']==the_age)]

        # contactos en la calle
        other_ages: pd.DataFrame = social[social['location']=='all'][["other", "value"]]

        for age, lam in other_ages.itertuples(index=False):
            people = np.random.poisson(lam * p_social)

            for i in range(people):
                yield person.region.spawn(age)

        # contactos en la escuela
        if Interventions.is_school_open():
            other_ages = social[social['location']=='school'][["other", "value"]]

            for age, lam in other_ages.itertuples(index=False):
                people = np.random.poisson(lam * p_social)

                for i in range(people):
                    yield person.region.spawn(age)

        # contactos en el trabajo
        if the_age < 18:
            return

        p_work = Interventions.is_workforce()

        if random.uniform(0, 1) < p_work:
            other_ages = social[social['location']=='work'][["other", "value"]]

            for age, lam in other_ages.itertuples(index=False):
                people = np.random.poisson(lam * p_work * p_social)

                for i in range(people):
                    yield person.region.spawn(age)


    def _eval_infections(self, person:Person) -> bool:
        """Determina si una persona cualquiera se infesta o no, dado que se cruza con "person". 

        En general depende del estado en el que se encuentra person y las probabilidades de ese estado
        """

        p = self.parameters.chance_of_infection

        if Interventions.is_use_masks():
            # usar mascarillas reduce en un 50% la probabilidad de infección
            p *= 0.5

        return random.uniform(0, 1) < p
