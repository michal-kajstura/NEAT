from typing import Optional

from gym import Env
from neat import StatisticsReporter, StdOutReporter, Config
from neat.nn import FeedForwardNetwork

from neat_improved.action_handler import handle_action
from neat_improved.evaluator import MultipleRunGymEvaluator
from neat_improved.runner import NEATRunner


def render_result(
    environment: Env,
    network: FeedForwardNetwork,
    steps: int = 500,
):

    observation = environment.reset()
    for _ in range(steps):
        outputs = network.activate(observation)
        action = handle_action(outputs, environment)
        observation, _, done, _ = environment.step(action)
        environment.render()

        if done:
            break

    environment.close()


def run(
    environment: Env,
    config: Config,
    num_generations: int,
    num_workers: Optional[int] = None,
):
    evaluator = MultipleRunGymEvaluator(
        environment=environment,
        render=False,
    )

    runner = NEATRunner(
        config=config,
        evaluator=evaluator,
        reporters=[
            StatisticsReporter(),
            StdOutReporter(
                show_species_detail=False,
            ),
        ],
        num_workers=num_workers,
    )

    best_genome = runner.run(num_generations)
    return best_genome
