import argparse
from covidsim.data import load_contact_matrices
import json
from pathlib import Path
# import tqdm

from .simulation import Simulation, JsonCallback, Region, SimulationCallback, SimulationParameters, StateMachine, TransitionEstimator, VaccinationParameters
from .interventions import INTERVENTIONS


class TqdmCallback(SimulationCallback):
    def on_day_begin(self, day, total_days):
        print(f"\rDay: {day}/{total_days}", end="")


def main():
    parser = argparse.ArgumentParser("python -m covidsim")
    parser.add_argument("config",type=Path, help="Location a .json file with config options.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of different runs to simulate.")
    parser.add_argument("--logdir", type=Path, default=Path("."), help="Directory to put log files.")

    args = parser.parse_args()

    with open(args.config) as fp:
        config = json.load(fp)

    simulation = build(config)

    for i in range(args.iterations):
        print(f"Iteration {i+1}")
        with open(args.logdir / f"simulation_{i+1}.jsonl", "w") as fp:
            simulation.run(JsonCallback(fp))#, TqdmCallback())


def build(config: dict) -> Simulation:
    parameters = SimulationParameters(
        days=config["days"],
        total_population=config["total_population"],
        foreigner_arrivals=config["foreigner_arrivals"],
        chance_of_infection=config["chance_of_infection"],
        initial_infected=config["initial_infected"],
        working_population=config["working_population"],
    )

    vaccination_data = config.get("vaccines", [])
    vaccination_list = []

    for params in vaccination_data:
        vaccination = VaccinationParameters(
            start_day=params["start_day"],
            name=params["name"],
            strategy=params["strategy"],
            age_bracket=params["age_bracket"],
            shots=params["shots"],
            shots_every=params["shots_every"],
            maximum_immunity=params["maximum_immunity"],
            symptom_reduction=params["symptom_reduction"],
            effect_growth=params["effect_growth"],
            effect_duration=params["effect_duration"],
            vaccinated_per_day=params["vaccinated_per_day"],
        )
        vaccination_list.append(vaccination)

    intervention_data = config.get("interventions", [])
    interventions = []
    interventions_names = {} 
    
    for cls in INTERVENTIONS:
        interventions_names[cls.description()] = cls

    for params in intervention_data:
        cls = interventions_names[params.pop("type")]
        interventions.append(cls(**params))

    state_machine = StateMachine()
    transitions = TransitionEstimator()
    contact = load_contact_matrices()

    region = Region(parameters.total_population, state_machine, parameters.initial_infected)
    return Simulation([region], contact, parameters, vaccination_list, transitions, state_machine, interventions)


if __name__ == "__main__":
    main()
