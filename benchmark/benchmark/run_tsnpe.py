import logging
import math
from typing import Optional, Tuple

import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import posterior_nn

from sbi.utils.support_posterior import PosteriorSupport

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 10,
    neural_net: str = "nsf",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 10000,
    num_atoms: int = 10,
    automatic_transforms_enabled: bool = False,
    z_score_x: bool = True,
    z_score_theta: bool = True,
    max_num_epochs: int = 100_000,
    num_samples_to_estimate_support: int = 10_000,
    allowed_false_negatives: float = 0.0,
    use_constrained_prior: bool = False,
    constrained_prior_quanitle: float = 0.0,
    proposal_sampling: str = "rejection",
    sir_oversample: int = 1024,
    atomic_loss: bool = False,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs (S)NPE from `sbi`
    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Simulation budget
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_rounds: Number of rounds
        neural_net: Neural network to use, one of maf / mdn / made / nsf
        hidden_features: Number of hidden features in network
        simulation_batch_size: Batch size for simulator
        training_batch_size: Batch size for training network
        num_atoms: Number of atoms, -1 means same as `training_batch_size`
        automatic_transforms_enabled: Whether to enable automatic transforms
        z_score_x: Whether to z-score x
        z_score_theta: Whether to z-score theta
        max_num_epochs: Maximum number of epochs
    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)

    # For TSNPE: we want to get the the fraction of gt posterior samples that are
    # accepted. For this, we have to load the reference posterior samples.
    # Load task
    reference_posterior_samples = task.get_reference_posterior_samples(num_observation)[
        :10000, :
    ]

    if num_rounds == 1:
        log.info(f"Running NPE")
        num_simulations_per_round = num_simulations
    else:
        log.info(f"Running SNPE")
        num_simulations_per_round = math.floor(num_simulations / num_rounds)

    if simulation_batch_size > num_simulations_per_round:
        simulation_batch_size = num_simulations_per_round
        log.warn("Reduced simulation_batch_size to num_simulation_per_round")

    if training_batch_size > num_simulations_per_round:
        training_batch_size = num_simulations_per_round
        log.warn("Reduced training_batch_size to num_simulation_per_round")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    simulator = task.get_simulator(max_calls=num_simulations)

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]

    if automatic_transforms_enabled:
        prior = wrap_prior_dist(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)

    density_estimator_fun = posterior_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )

    inference_method = inference.SNPE_C(prior, density_estimator=density_estimator_fun)
    posteriors = []
    proposal = prior
    acceptance_rates = []
    gt_acceptances = []
    iw_variances = []

    for round_num in range(num_rounds):
        theta, x = inference.simulate_for_sbi(
            simulator,
            proposal,
            num_simulations=num_simulations_per_round,
            simulation_batch_size=simulation_batch_size,
        )
        # Compute acceptance rate
        if isinstance(proposal, PosteriorSupport):
            _, acceptance_rate, log_iw = proposal.sample(
                (100,), return_acceptance_rate=True, return_iw=True
            )
            log_iw = log_iw[:, :sir_oversample]
            norm_log_iw = (log_iw.T - torch.logsumexp(log_iw, 1)).T
            iw = torch.exp(norm_log_iw)
            summed_iw = torch.sum(iw, dim=1)
            assert torch.all(summed_iw > 0.99) and torch.all(summed_iw < 1.01)
            ess = 1 / torch.sum(iw**2, dim=1)
            ess = torch.mean(ess).item()
            acceptance_rate = acceptance_rate.item()
        else:
            acceptance_rate = 1.0
            ess = 0.0
        iw_variances.append(ess)
        acceptance_rates.append(acceptance_rate)

        # Compute fraction of accepted gt samples
        if isinstance(proposal, PosteriorSupport):
            accepted_or_not = proposal.predict(reference_posterior_samples)
            gt_acceptance = (torch.sum(accepted_or_not) / 10000).item()
        else:
            gt_acceptance = 1.0
        gt_acceptances.append(gt_acceptance)

        # This will not be used anyways, so we can just pass the posterior to make
        # sure that SNPE use
        if round_num > 0:
            proposal_train = posterior if atomic_loss else None
            force_first_round = False if atomic_loss else True
        else:
            proposal_train = None
            force_first_round = True

        _ = inference_method.append_simulations(
            theta, x, proposal=proposal_train
        ).train(
            num_atoms=num_atoms,
            training_batch_size=training_batch_size,
            retrain_from_scratch=False,
            discard_prior_samples=False,
            use_combined_loss=False,
            show_train_summary=True,
            max_num_epochs=max_num_epochs,
            force_first_round_loss=force_first_round,
        )
        posterior = inference_method.build_posterior()
        posterior = posterior.set_default_x(observation)
        posterior_support = PosteriorSupport(
            prior,
            posterior,
            num_samples_to_estimate_support=num_samples_to_estimate_support,
            allowed_false_negatives=allowed_false_negatives,
            use_constrained_prior=use_constrained_prior,
            constrained_prior_quanitle=constrained_prior_quanitle,
            sampling_method=proposal_sampling,
            sir_oversample=sir_oversample,
        )
        proposal = posterior_support
        posteriors.append(posterior)

    posterior = wrap_posterior(posteriors[-1], transforms)

    assert simulator.num_simulations == num_simulations

    samples = posterior.sample((num_samples,)).detach()

    if num_observation is not None:
        true_parameters = task.get_true_parameters(num_observation=num_observation)
        log_prob_true_parameters = posterior.log_prob(true_parameters)
        return (
            samples,
            simulator.num_simulations,
            log_prob_true_parameters,
            torch.as_tensor(acceptance_rates),
            torch.as_tensor(gt_acceptances),
            torch.as_tensor(iw_variances),
        )
    else:
        return (
            samples,
            simulator.num_simulations,
            None,
            torch.as_tensor(acceptance_rates),
            torch.as_tensor(gt_acceptances),
            torch.as_tensor(iw_variances),
        )
