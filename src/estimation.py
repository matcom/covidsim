import collections
import math
import streamlit as st
from sklearn.linear_model import ElasticNet
import numpy as np
from typing import Callable, List, Dict

from src.simulation import SimulationParameters, Simulation, StreamlitCallback, SimulationCallback, Person


class EstimationCallback(SimulationCallback):
    def __init__(self) -> None:
        self.info = st.empty()
        self.progress = st.progress(0)
        self.history = collections.defaultdict(list)
        self.day_history = {}

    def on_day_begin(self, day: int, total_days: int):
        self.progress.progress((day+1) / total_days)
        self.info.markdown("---")
        self.info.markdown(f"**Día: {day+1}**: `{dict(self.day_history)}`")
        self.day_history = collections.defaultdict(lambda: 0)

    def on_person(self, person: Person, total_people: int):
        self.day_history[person.state.label] += 1

    def on_day_end(self, day: int, total_days: int):
        for k,v in self.day_history.items():
            self.history[k].append(v)


def estimate_parameter(
    param: str, 
    history:Dict[str,List[int]],
    parameters: SimulationParameters,
    simulation_factory: Callable[[SimulationParameters], Simulation],
    x_min:float=0, 
    x_max:float=1, 
    start_day:int=0, 
    end_day:int=90, 
    steps: int = 10,
):
    """
    Estima el valor óptimo de un parámetro específico `param`, en un rango de valores
    `[x_min, x_max]`.

    Recibe un diccionario de listas que representa la cantidad *real* de personas en cada estado por cada día.

    `start_day` y `end_day` definen los días a los que corresponden las mediciones en `history`.

    `parameters` contiene el resto de los parámetros de la simulación.
    """

    X = []
    y = []

    for x in np.linspace(x_min, x_max, steps):
        # Crear los parámetros para esta simulación
        p = parameters.clone()
        p.days = end_day
        setattr(p, param, x)

        st.write(f"**{param} = `{x}`**")
        simulation = simulation_factory(p)

        # Ejecutar la simulación y quedarse con las curvas
        callback = EstimationCallback()
        simulation.run(callback)

        # Calcular el error
        error = _compute_curve_error(callback.history, history, start_day, end_day)
        st.write(f"**Error = `{error}`**")
        
        X.append([1, x, x*x])
        y.append(error)

    regressor = ElasticNet(positive=True)
    regressor.fit(X, y)

    X = np.linspace(x_min, x_max, 100 * steps).reshape(-1,1)
    X = np.hstack((np.ones_like(X), X, X*X))    
    y = regressor.predict(X)
    st.write(X, y)


def _compute_curve_error(simulated_history: Dict[str,List[int]], history: Dict[str, List[int]], start_day:int, end_day:int):
    error = 0

    for key in simulated_history:
        if key not in history:
            continue

        result_values = simulated_history[key][start_day:end_day]
        history_values = history[key][start_day:end_day]

        for xi,yi in zip(result_values, history_values):
            error += (xi-yi)**2 / (xi**2 + yi**2)

    for key in history:
        if not key in simulated_history:
            error += (end_day - start_day)

    return math.sqrt(error) / (end_day - start_day)
