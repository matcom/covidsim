from typing import Callable
from dataclasses import dataclass

import collections


@dataclass
class SimulationParameters:
    days:int


class Region:
    pass


class Simulation:
    def __init__(self, parameters: SimulationParameters, regions, interventions, contact, callback:Callable[["Simulation"], None]=None) -> None:
        self.parameters = parameters
        self.regions = regions
        self.interventions = interventions
        self.contact = contact
        self.callback = callback or self._no_callback

    # método de tranmisión espacial, teniendo en cuenta la localidad
    def run(self):
        # por cada paso de la simulación
        for day in range(self.parameters.days):
            total_individuals = 0
            by_state = collections.defaultdict(lambda: 0)

            # por cada región
            for region in self.regions:
                params = self._simulate_interventions(region, day, self.parameters)

                # llegadas del estranjero
                self._simulate_arrivals(region, day)
                # por cada persona
                for ind in region:
                    # actualizar estado de la persona
                    ind.next_step()
                    if ind.is_infectious:
                        self._simulate_spread(ind, day)

                    total_individuals += 1
                    by_state[ind.state] += 1

                # movimientos
                for n_region in self.regions:
                    if n_region != region:
                        # calcular personas que se mueven de esta region a otras
                        self._simulate_transportation(region, n_region, day)

    def _simulate_arrivals(self, region:Region, day:int):
        pass

    def _no_callback(self):
        pass