import abc
import collections
from io import FileIO
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import time
import json
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import argparse

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
        self.id = Person.total
        Person.total += 1
        self.state_machine = states
        self.state = self.state_machine.start
        self.next_state = None
        self.steps_remaining = None
        self.infected = 0
        self.vaccinated = False
        self.vaccinated_day = None
        self.vaccine: VaccinationParameters = None

        # la persona conoce la region a la que pertenece
        self.region = region
        self.age = age
        self.sex = sex
        self.health_conditions = None

        # llamar método de estado inicial
        self.set_state(states.start)

        assert self.state is not None, "individual state is None"

    def to_json(self):
        return self.id

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

            if self.is_infectious:
                self.region._infectious.add(self)

            return True
        else:
            # decrementar los steps que faltan para cambiar de estado
            self.steps_remaining = self.steps_remaining - 1

        return False

    def __repr__(self):
        return f"Person(age={self.age}, sex={self.sex}, state={self.state}"

    def set_state(self, state):
        if state is None:
            raise ValueError("state cannot be None")

        self.state = state
        self.next_state = state
        self.steps_remaining = 0

        if self.is_infectious:
            self.region._infectious.add(self)
        else:
            if self in self.region._infectious:
                self.region._infectious.remove(self)

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
    def __init__(self, population:int, states: StateMachine, initial_infected:int=0, initial_recovered:int=0):
        self._population = population
        self.states = states
        self._individuals: List[Person] = []
        self._infectious = set()
        self._by_age = collections.defaultdict(list)

        ages = list(range(0, 100, 5))

        if isinstance(population, int):
            weights = [population // len(ages)] * len(ages)
        elif isinstance(population, (list, tuple)):
            weights = population

        for _ in range(sum(weights)):
            self.add(random.choices(ages, weights=weights, k=1)[0], random.choice(("M", "F")), states.start)

        for p in random.sample(self._individuals, initial_infected):
            p.set_state(states["Contagiado"])

        for p in random.sample(self._individuals, initial_recovered):
            p.set_state(states["Recuperado"])

    def add(self, age:int, sex:str, state:str):
        p = Person(self, age, sex, self.states)
        p.set_state(state)

        self._individuals.append(p)
        self._by_age[p.age].append(p)

        if p.is_infectious:
            self._infectious.add(p)

        return p

    def sample(self, age:int, k:int=1):
        pop = self._by_age[age]
        return random.sample(pop, min(k, len(pop)))

    def __len__(self):
        return len(self._individuals)

    def __iter__(self) -> Iterable[Person]:
        for i in list(self._infectious):
            yield i


@dataclass
class SimulationParameters:
    days:int
    foreigner_arrivals:float
    chance_of_infection:float
    initial_infected:int
    initial_recovered: int
    total_population:int
    working_population:float

    def clone(self) -> "SimulationParameters":
        return SimulationParameters(**self.__dict__)


class SimulationCallback:
    def __call__(self, event: str, **kwds):
        self._on_event(event, **kwds)

    def _on_event(self, event: str, **kwds):
        event = f"on_{event}"

        if not hasattr(self, event):
            return

        getattr(self, event)(**kwds)


class JsonCallback(SimulationCallback):
    def __init__(self, fp: FileIO) -> None:
        self.fp = fp

    def to_json(self, v):
        if hasattr(v, "to_json"):
            return v.to_json()

        return v

    def _on_event(self, event: str, **kwds):
        data = dict(
            event=event,
            **{ k:self.to_json(v) for k,v in kwds.items() }
        )

        self.fp.write(json.dumps(data))
        self.fp.write("\n")


class MultiCallback(SimulationCallback):
    def __init__(self, callbacks: List[SimulationCallback]) -> None:
        self.callbacks = callbacks

    def _on_event(self, event: str, **kwds):
        for callback in self.callbacks:
            callback._on_event(event, **kwds)


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
            ),
            use_container_width=True,
        )

        self.vaccinated_chart = st.altair_chart(
            alt.Chart(pd.DataFrame(columns=["vacuna", "dia"]), title="Vacunados")
            .mark_bar()
            .encode(
                y=alt.Y("count():Q", title="Vacunados"),
                x=alt.X("dia:N", title="Días"),
                color=alt.Color("vacuna:N", title="Estado"),
            ),
            use_container_width=True,
        )

    def on_vaccine(self, vaccine: str, day:int, person:Person):
        self.vaccinated_chart.add_rows([
            dict(vacuna=vaccine, dia=self.current_day)
        ])

    def on_day_begin(self, day: int, total_days:int):
        self.by_state = collections.defaultdict(lambda: 0)
        self.current_day_total = 0
        self.current_day = day
        self.infections = collections.defaultdict(lambda: 0)

    def on_person(self, person: Person, total_people:int, **kwargs):
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


@dataclass
class VaccinationParameters:
    name: str
    start_day: int
    vaccinated_per_day: int
    maximum_immunity: float
    symptom_reduction: float
    effect_growth: int
    shots: int
    shots_every: int
    effect_duration: int
    strategy: str
    age_bracket: Tuple[int]

    def to_json(self):
        return self.__dict__

    def evaluate_immunity(self, p: Person, chance_of_infection: float, day:int):
        immunity_per_day = self.maximum_immunity / self.effect_growth
        immunity = (day - p.vaccinated_day) * immunity_per_day

        if day > p.vaccinated_day + self.effect_duration:
            return chance_of_infection

        return chance_of_infection * (1 - immunity)

    def evaluate_symptoms_reduction(self, p:Person, day: int):
        symptom_per_date = self.symptom_reduction / self.effect_growth
        symptom_reduction = (day - p.vaccinated_day) * symptom_per_date
        return (1 - symptom_reduction)


class Simulation:
    def __init__(
        self,
        regions:List[Region],
        contact,
        parameters: SimulationParameters,
        vaccination: List[VaccinationParameters],
        transitions: TransitionEstimator,
        state_machine: StateMachine,
        interventions: List["Intervention"],
    ) -> None:
        self.regions = regions
        self.contact = contact
        self.parameters = parameters
        self.transitions = transitions
        self.state_machine = state_machine
        self.interventions = interventions
        self.vaccination = vaccination

    def run(self, callback:SimulationCallback=None):
        if callback is None:
            callback = SimulationCallback()

        for vaccine in self.vaccination:
            vaccine._vaccination_pool = []

            for _ in range(self.parameters.days):
                if isinstance(vaccine.vaccinated_per_day, int):
                    vaccinated = vaccine.vaccinated_per_day
                else:
                    vaccinated = random.randint(*vaccine.vaccinated_per_day)

                vaccine._vaccination_pool.append(vaccinated)

        # por cada paso de la simulación
        for day in range(self.parameters.days):
            callback("day_begin", day=day, total_days=self.parameters.days)

            # por cada región
            for region in self.regions:
                # aplicar intervenciones generales
                parameters, contact = self._apply_interventions(region, self.parameters, self.contact, day)

                # llegadas del extranjero
                self._simulate_arrivals(region, parameters)

                # por cada persona
                individuals = list(region)
                for ind in individuals:
                    # aplicar intervenciones a nivel de individuo
                    data = self._apply_individual_interventions(ind, day, self.transitions.data)

                    # actualizar estado de la persona
                    if ind.next_step(data):
                        callback("person", person=ind, age=ind.age, sex=ind.sex, state=ind.state.label, total_people=len(individuals))

                    if ind.is_infectious:
                        self._simulate_spread(ind, parameters, contact, callback, day)

                # vacunación
                self._simulate_vaccination(day, region, callback)

                # movimientos
                for n_region in self.regions:
                    if n_region != region:
                        # calcular personas que se mueven de una region a otras
                        self._simulate_transportation(region, n_region)

            callback("day_end", day=day, total_days=self.parameters.days)


    def _simulate_vaccination(self, day, region: Region, callback):
        for vaccine in self.vaccination:
            if day >= vaccine.start_day:
                age_min, age_max = vaccine.age_bracket
                pool = [p for p in region._individuals if age_min <= p.age <= age_max and not p.vaccinated and not p.is_infectious]

                if len(pool) > vaccine._vaccination_pool[day]:
                    if vaccine.strategy == "random":
                        pool = random.sample(pool, vaccine._vaccination_pool[day])
                    elif vaccine.strategy == "bottom-up":
                        pool = sorted(pool, key=lambda p: p.age)[:vaccine._vaccination_pool[day]]
                    elif vaccine.strategy == "top-down":
                        pool = sorted(pool, key=lambda p: p.age)[-vaccine._vaccination_pool[day]:]

                for p in pool:
                    if p.state.label != "Persona":
                        continue

                    p.vaccinated = True
                    p.vaccinated_day = day
                    p.vaccine = vaccine

                    callback("vaccine", vaccine=vaccine.name, day=day, person=p)

                    if vaccine.shots == 1:
                        continue

                    for i in range(1, vaccine.shots):
                        next_vaccination = day + i * vaccine.shots_every

                        if next_vaccination < len(vaccine._vaccination_pool):
                            vaccine._vaccination_pool[next_vaccination] -= 1


    def _simulate_arrivals(self, region: Region, parameters: SimulationParameters):
        people = np.random.poisson(parameters.foreigner_arrivals)

        for _ in range(people):
            region.add(random.randint(20, 80), random.choice(['M', 'F']), region.states["Viajero"])


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

        # Apply vaccination intervention
        # TODO: Can this be made more generally?
        if person.vaccinated:
            data = data.copy()
            data.loc[data["to_state"]=="Sintomático","chance"] *= person.vaccine.evaluate_symptoms_reduction(person, day)

        return data


    def _simulate_transportation(self, here:Region, there:Region, distance):
        """Las personas que se mueven de n_region a region.
        """
        pass


    def _simulate_spread(self, ind: Person, parameters: SimulationParameters, contact, callback, day):
        """Calcula que personas serán infectadas por 'ind' en el paso actual de la simulación.
        """
        connections = self._eval_connections(ind, parameters, contact)

        for other in connections:
            if other.state != self.state_machine["Persona"]:
                continue

            if self._eval_infections(ind, other, parameters, day):
                callback("infection", from_person=ind, to_person=other, day=day)
                other.set_state(self.state_machine["Contagiado"])

    def _eval_connections(
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

            for p in person.region.sample(age):
                yield p

        # contactos en la escuela
        other_ages = social[social['location']=='school'][["other", "value"]]

        for age, lam in other_ages.itertuples(index=False):
            people = np.random.poisson(lam)

            for p in person.region.sample(age, people):
                yield p

        # contactos en el trabajo
        if the_age < 18 or the_age > 65:
            return

        p_work = parameters.working_population

        if random.uniform(0, 1) < p_work:
            other_ages = social[social['location']=='work'][["other", "value"]]

            for age, lam in other_ages.itertuples(index=False):
                people = np.random.poisson(lam)

                for p in person.region.sample(age):
                    yield p


    def _eval_infections(self, person:Person, other:Person, parameters:SimulationParameters, day) -> bool:
        """Determina si una persona cualquiera se infesta o no, dado que se cruza con "person".

        En general depende del estado en el que se encuentra person y las probabilidades de ese estado
        """

        # Parámetro de infección original
        p = parameters.chance_of_infection

        # Parámetro de infección nuevo
        if other.vaccinated:
            p = other.vaccine.evaluate_immunity(other, p, day)

        if random.uniform(0, 1) < p:
            person.infected += 1
            return True

        return False
