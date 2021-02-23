import numpy as np
from typing import List, Dict

from src.simulation import SimulationParameters, Simulation


def estimate_parameter(
    param: str, 
    history:List[Dict[str,int]], 
    parameters: SimulationParameters,
    simulation: Simulation,
    x_min:float=0, 
    x_max:float=1, 
    start_day:int=0, 
    end_day:int=90, 
    steps: int = 10,
):
    """
    Estima el valor óptimo de un parámetro específico `param`, en un rango de valores
    `[x_min, x_max]`.

    Recibe un lista que representa la cantidad *real* de personas en cada estado por cada día.

    `start_day` y `end_day` definen los días a los que corresponden las mediciones en `history`.

    `parameters` contiene el resto de los parámetros de la simulación.
    """

    for x in np.arange(x_min, x_max):
        # Crear los parámetros para esta simulación
        p = parameters.clone()
        setattr(p, param, x)
        simulation.parameters = p

        # Ejecutar la simulación y quedarse con las curvas
        result = simulation.run()

        # Calcular el error
        error = _compute_curve_error(result, history)
    