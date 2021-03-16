import abc
import collections
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .data import *



def _dummy_cache(f):
    return f
cache = _dummy_cache


class TransitionEstimator:
    def __init__(self, data=None):
        self.data = load_disease_transition() if data is None else data

    @cache
    def transition(self, from_state, age, sex):
        age = (age // 5) * 5
        df = self.data[(self.data["age"] == age) & (self.data["from_state"] == from_state) & (self.data["sex"] == sex)]

        if len(df) == 0:
            raise ValueError(f"No transitions for {from_state}, age={age}, sex={sex}.")

        return pd.DataFrame(df).sort_values("chance")


@dataclass
class State:
    label: str
    starting: bool
    susceptible: bool
    testeable: bool
    infectious: bool
    temporal: bool
    final: bool


class StateMachine:
    def __init__(self) -> None:
        self.states = load_states()
        self._start = None

        for state in iter(self):
            if state.starting:
                self._start = state
                break

        assert self._start is not None, "Initial state is None"    

    @property
    def start(self) -> State:
        return self._start

    def __iter__(self) -> Iterable[State]:
        for k in self.states:
            yield self[k]

    def __getitem__(self, index) -> State:
        return State(label=index, **{k:v == "yes" for k,v in self.states[index].items()})


class Person:
    total = 0

    def __init__(self, region: "Region", age:int, sex:str,  states: StateMachine):
        """Crea una nueva persona que por defecto está en el estado de susceptible al virus.
        """
        Person.total += 1
        self.state_machine = states
        self.state = self.state_machine.start
        self.next_state = None
        self.steps_remaining = None
        self.infected = 0

        # la persona conoce la region a la que pertenece
        self.region = region
        self.age = age
        self.sex = sex
        self.health_conditions = None

        # llamar método de estado inicial
        self.set_state(states.start)

        assert self.state is not None, "individual state is None"

    @property
    def is_infectious(self):
        return self.state.infectious

    def next_step(self, transition_data):
        """Ejecuta un step de tiempo para una persona.
        """
        if self.steps_remaining == 0:
            # actualizar state
            self.state = self.next_state
            self.next_state, self.steps_remaining = self._evaluate_transition(transition_data)
        else:
            # decrementar los steps que faltan para cambiar de estado
            self.steps_remaining = self.steps_remaining - 1

        return True

    def __repr__(self):
        return f"Person(age={self.age}, sex={self.sex}, state={self.state}, steps_remaining={self.steps_remaining})"

    def set_state(self, state):
        if state is None:
            raise ValueError("state cannot be None")

        self.state = state
        self.next_state = state
        self.steps_remaining = 0

    def _evaluate_transition(self, transition_data):
        """Computa a qué estado pasar dado el estado actual y los valores de la tabla.
        """
        if not self.state.temporal:
            return self.state, 1

        df = TransitionEstimator(transition_data).transition(self.state.label, self.age, self.sex)
        # calcular el estado de transición y el tiempo
        to_state = random.choices(df["to_state"].values, weights=df["chance"].values, k=1)[0]
        state_data = df.set_index("to_state").to_dict("index")[to_state]
        time = random.normalvariate(state_data["mean_days"], state_data["std_days"])

        return self.state_machine[to_state], int(time)


class Region:
    def __init__(self, population, states: StateMachine, initial_infected:int):
        self._recovered = 0
        self._population = population
        self._death = 0
        self._simulations = 0
        self.states = states
        self._individuals = []

        for i in range(initial_infected):
            p = Person(self, random.randint(20, 80), random.choice(["M", "F"]), states)
            p.set_state(states["Contagiado"])
            self._individuals.append(p)

    def __len__(self):
        return len(self._individuals)

    def __iter__(self) -> Iterable[Person]:
        for i in list(self._individuals):
            if i.state != self.states.start:
                yield i

    def spawn(self, age) -> Person:
        p = Person(self, age, random.choice(["M", "F"]), self.states)
        self._individuals.append(p)
        return p

    def prune(self, states):
        self._individuals = [p for p in self._individuals if p.state.label not in states]


@dataclass
class SimulationParameters:
    days:int
    foreigner_arrivals:float
    chance_of_infection:float
    initial_infected:int
    working_population:float

    def clone(self) -> "SimulationParameters":
        return SimulationParameters(**self.__dict__)


class SimulationCallback:
    def on_day_begin(self, day:int, total_days:int):
        pass

    def on_day_end(self, day:int, total_days:int):
        pass

    def on_person(self, person: Person, total_people:int):
        pass

    def on_infection(self, from_person: Person, to_person: Person):
        pass


class StreamlitCallback(SimulationCallback):
    def __init__(self) -> None:
        # estadísticas de la simulación
        self.progress = st.progress(0)
        self.progress_person = st.progress(0)
        self.day = st.empty()
        self.speed = st.empty()
        self.sick_count = st.empty()
        self.all_count = st.empty()
        self.total_individuals = 0
        self.start_iter_time = time.time()
        self.dead_set = set()
        self.infection_rate_counted = set()

        self.chart = st.altair_chart(
            alt.Chart(pd.DataFrame(columns=["personas", "dia", "estado"]))
            .mark_bar()
            .encode(
                y=alt.Y("personas:Q", title="Individuos"),
                x=alt.X("dia:N", title="Días"),
                color=alt.Color("estado:N", title="Estado"),
            ),
            use_container_width=True,
        )

        self.deaths_chart = st.altair_chart(
            alt.Chart(pd.DataFrame(columns=["gender", "age"]))
            .mark_bar().
            encode(
                x=alt.X("age:Q"),# domain=list(range(5,100,5))),
                y="count()",
                color="gender",
            ),
            use_container_width=True,
        )

        self.infected_per_day = st.altair_chart(
            alt.Chart(pd.DataFrame(columns=["infected", "day"]))
            .mark_line().
            encode(
                x="day:N",
                y="infected:Q",
            ),
            use_container_width=True,
        )

        self.infection_rate = st.altair_chart(
            alt.Chart(pd.DataFrame(columns=['day', 'rate']))
            .mark_line()
            .encode(
                x="day:N",
                y="mean(rate):Q"
            )
        )

    def on_day_begin(self, day: int, total_days:int):
        self.by_state = collections.defaultdict(lambda: 0)
        self.current_day_total = 0
        self.current_day = day
        self.infections = collections.defaultdict(lambda: 0)

    def on_person(self, person: Person, total_people:int):
        self.total_individuals += 1
        self.current_day_total += 1
        self.progress_person.progress(self.current_day_total / total_people)
        self.by_state[person.state.label] += 1
        speed = self.total_individuals / (time.time() - self.start_iter_time)
        self.speed.markdown(f"#### Velocidad: *{speed:0.2f}* ind/s")
        self.sick_count.markdown(f"#### Individuos simulados: {self.total_individuals}")

        if person.state.label == "Fallecido" and person not in self.dead_set:
            self.deaths_chart.add_rows([
                dict(age=(person.age//5)*5, gender=person.sex)
            ])
            self.dead_set.add(person)

        if person.state.final and person not in self.infection_rate_counted:
            self.infection_rate_counted.add(person)
            self.infection_rate.add_rows([
                dict(day=self.current_day, rate=person.infected)
            ])

    def on_infection(self, from_person: Person, to_person: Person):
        self.infections[from_person] += 1

    def on_day_end(self, day:int, total_days:int):
        self.progress.progress((day + 1) / total_days)
        self.all_count.code(dict(self.by_state))
        self.day.markdown(f"#### Día: {day+1}")

        self.chart.add_rows(
            [dict(dia=day + 1, personas=v, estado=k) for k, v in self.by_state.items()]
        )

        self.infected_per_day.add_rows([
            dict(day=day, infected=sum(self.infections.values()))
        ])

class Simulation:
    def __init__(
        self, 
        regions:List[Region], 
        contact, 
        parameters: SimulationParameters, 
        transitions: TransitionEstimator, 
        state_machine: StateMachine,
        interventions,
    ) -> None:
        self.regions = regions
        self.contact = contact
        self.parameters = parameters
        self.transitions = transitions
        self.state_machine = state_machine
        self.interventions = interventions

    def run(self, callback:SimulationCallback=None):
        if callback is None:
            callback = SimulationCallback()

        # por cada paso de la simulación
        for day in range(self.parameters.days):
            callback.on_day_begin(day, self.parameters.days)

            # por cada región
            for region in self.regions:
                # aplicar intervenciones generales
                parameters, contact = self._apply_interventions(region, self.parameters, self.contact, day)
                
                # llegadas del extranjero
                self._simulate_arrivals(region, parameters)
                
                # por cada persona
                individuals = list(region)
                for j, ind in enumerate(individuals):
                    # aplicar intervenciones a nivel de individuo
                    data = self._apply_individual_interventions(ind, day, self.transitions.data)

                    # actualizar estado de la persona
                    ind.next_step(data)

                    if ind.is_infectious:
                        self._simulate_spread(ind, parameters, contact, callback)

                    callback.on_person(ind, len(individuals))

                region.prune(['Persona'])

                # movimientos
                for n_region in self.regions:
                    if n_region != region:
                        # calcular personas que se mueven de una region a otras
                        self._simulate_transportation(region, n_region)

            callback.on_day_end(day, self.parameters.days)


    def _simulate_arrivals(self, region: Region, parameters: SimulationParameters):
        people = np.random.poisson(parameters.foreigner_arrivals)

        for _ in range(people):
            p = region.spawn(random.randint(20, 80))
            p.set_state(region.states["Viajero"])


    def _apply_interventions(self, region: Region, parameters:SimulationParameters, contact, day:int):
        """Modifica el estado de las medidas y como influyen estas en la población.
        """
        for intervention in self.interventions:
            if not intervention.is_active(day):
                continue

            parameters, contact = intervention.apply(parameters, contact)

        # si se está testeando activamente
        return parameters, contact

    def _apply_individual_interventions(self, person:Person, day, data):
        """Modifica los parámetros personales de una persona
        """
        for intervention in self.interventions:
            if not intervention.is_active(day):
                continue

            if not intervention.applies_to(person):
                continue

            data = intervention.apply_individual(data)

        return data


    def _simulate_transportation(self, here:Region, there:Region, distance):
        """Las personas que se mueven de n_region a region.
        """
        pass


    def _simulate_spread(self, ind, parameters, contact, callback):
        """Calcula que personas serán infectadas por 'ind' en el paso actual de la simulación.
        """
        connections = self._eval_connections(ind, parameters, contact)

        for other in connections:
            if other.state != self.state_machine["Persona"]:
                continue

            if self._eval_infections(ind, parameters):
                callback.on_infection(ind, other)
                other.set_state(self.state_machine["Contagiado"])

    def _eval_connections(
        # social: Dict[str, Dict[int, Dict[int, float]]], person: "Person"
        self,
        person: Person,
        parameters: SimulationParameters,
        contact,
    ) -> List["Person"]:
        """Devuelve las conexiones que tuvo una persona en un step de la simulación.
        """

        the_age = person.age

        if the_age % 5 != 0:
            the_age = the_age // 5 * 5

        # quedarse con la submatrix de contacto de la edad correspondiente
        social = contact[(contact['type']=='overall') & (contact['subject']==the_age)]

        # contactos en la calle
        other_ages: pd.DataFrame = social[social['location']=='all'][["other", "value"]]

        for age, lam in other_ages.itertuples(index=False):
            people = np.random.poisson(lam)

            for i in range(people):
                yield person.region.spawn(age)

        # contactos en la escuela
        other_ages = social[social['location']=='school'][["other", "value"]]

        for age, lam in other_ages.itertuples(index=False):
            people = np.random.poisson(lam)

            for i in range(people):
                yield person.region.spawn(age)

        # contactos en el trabajo
        if the_age < 18:
            return

        p_work = parameters.working_population

        if random.uniform(0, 1) < p_work:
            other_ages = social[social['location']=='work'][["other", "value"]]

            for age, lam in other_ages.itertuples(index=False):
                people = np.random.poisson(lam)

                for i in range(people):
                    yield person.region.spawn(age)


    def _eval_infections(self, person:Person, parameters:SimulationParameters) -> bool:
        """Determina si una persona cualquiera se infesta o no, dado que se cruza con "person". 

        En general depende del estado en el que se encuentra person y las probabilidades de ese estado
        """

        if random.uniform(0, 1) < parameters.chance_of_infection:
            person.infected += 1
            return True

        return False
