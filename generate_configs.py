from covidsim.simulation import SimulationParameters, VaccinationParameters
from covidsim.interventions import WearMask

import json
import itertools


PARAMETERS = dict(
    soberana_inmunidad = [0.5, 0.6, 0.7, 0.8, 0.9],
    soberana_reduccion = [0.9, 0.8, 0.7],
    abdala_inmunidad = [0.5, 0.6, 0.7, 0.8, 0.9],
    abdala_reduccion = [0.9, 0.8, 0.7],
    strategy=["bottom-up", "top-down"],
    mask_effect=[0.3, 0.2, 0.05]
)


def generate_config(
    soberana_inmunidad: float,
    soberana_reduccion: float,
    abdala_inmunidad: float,
    abdala_reduccion: float,
    strategy: str,
    mask_effect: float,
):
    params = SimulationParameters(
        days=180,
        foreigner_arrivals=10,
        chance_of_infection=0.2,
        initial_infected=0,
        total_population=1_500_000,
        working_population=0.2
    )

    vaccines = [
        VaccinationParameters(
            name="Soberana",
            start_day=0,
            age_bracket=[20,80],
            vaccinated_per_day=1000,
            maximum_immunity=soberana_inmunidad,
            symptom_reduction=soberana_reduccion,
            effect_duration=180,
            shots=3,
            shots_every=28,
            effect_growth=50,
            strategy=strategy,
        ),
        VaccinationParameters(
            name="Abdala",
            start_day=0,
            age_bracket=[20,80],
            vaccinated_per_day=1000,
            maximum_immunity=abdala_inmunidad,
            symptom_reduction=abdala_reduccion,
            effect_duration=180,
            shots=3,
            shots_every=14,
            effect_growth=50,
            strategy=strategy
        )
    ]

    interventions = [
        WearMask(0, 180, mask_effect)
    ]

    return dict(
        params.__dict__, 
        vaccines=[
            v.__dict__ for v in vaccines
        ],
        interventions=[
            dict(i.__dict__, type=i.description()) for i in interventions
        ]
    )


if __name__ == "__main__":
    keys = PARAMETERS.keys()

    for i, permutation in enumerate(itertools.product(*PARAMETERS.values())):
        kwargs = { k:v for k,v in zip(keys, permutation) }
        config = generate_config(**kwargs)

        with open(f"params/experiment/simulation_{i+1}.json", "w") as fp:
            json.dump(config, fp, indent=4)

    print(f"Done. Generated {i+1} configs.")
