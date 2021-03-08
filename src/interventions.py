import abc
import streamlit as st
import inspect
import pandas as pd


from .simulation import SimulationParameters


class Intervention(abc.ABC):
    def __init__(self, start_day: int, end_day: int) -> None:
        self.start_day = start_day
        self.end_day = end_day

    def is_active(self, current_day:int):
        return self.start_day <= current_day <= self.end_day

    @abc.abstractmethod
    def apply(self, parameters: SimulationParameters, contact: pd.DataFrame):
        pass

    @abc.abstractclassmethod
    def description(cls):
        pass

    @classmethod
    def build(cls, key):
        args = inspect.signature(cls.__init__)
        values = {}

        for arg, param in args.parameters.items():
            if param.annotation == int:
                values[arg] = st.number_input(f"{cls.__name__}.{arg}", value=0, key=f"intervention_{key}_{arg}")
            if param.annotation == float:
                values[arg] = st.slider(f"{cls.__name__}.{arg}", 0.0, 1.0, key=f"intervention_{key}_{arg}")

        return cls(**values)


class CloseAirports(Intervention):
    def apply(self, parameters: SimulationParameters, contact):
        new_params = parameters.clone()
        new_params.foreigner_arrivals = 0

        return new_params, contact
    
    @classmethod
    def description(cls):
        return "✈️ Cerrar aeropuertos"


class CloseSchools(Intervention):
    def apply(self, parameters: SimulationParameters, contact: pd.DataFrame):
        contact_new = contact.copy()
        contact_new.loc[contact_new["location"] == "school","value"] = 0.0

        return parameters, contact_new

    @classmethod
    def description(cls):
        return "🏫 Cerrar escuelas"


class StayAtHome(Intervention):
    def __init__(self, start_day: int, end_day: int, percentage:float) -> None:
        super().__init__(start_day, end_day)
        self.percentage = percentage

    def apply(self, parameters: SimulationParameters, contact: pd.DataFrame):
        contact_new = contact.copy()
        contact_new.loc[:,"value"] *= (1 - self.percentage)

        return parameters, contact_new

    @classmethod
    def description(cls):
        return "🏠 Aislarse en casa"


class WearMask(Intervention):
    def __init__(self, start_day: int, end_day: int, effect:float) -> None:
        super().__init__(start_day, end_day)
        self.effect = effect

    def apply(self, parameters: SimulationParameters, contact: pd.DataFrame):
        new_params = parameters.clone()
        new_params.chance_of_infection *= self.effect

        return new_params, contact

    @classmethod
    def description(cls):
        return "😷 Usar máscarillas"


INTERVENTIONS = [CloseAirports, CloseSchools, StayAtHome, WearMask]
