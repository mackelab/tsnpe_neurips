from typing import Any, Optional
import torch
from torch import zeros, ones, eye, tensor, as_tensor, float32, Tensor
from tqdm.auto import tqdm

from sbi.types import Shape
from sbi.utils import BoxUniform


class PosteriorSupport:
    def __init__(
        self,
        prior: Any,
        posterior: Any,
        num_samples_to_estimate_support: int = 10_000,
        allowed_false_negatives: float = 0.0,
        use_constrained_prior: bool = False,
        constrained_prior_quanitle: bool = 0.0,
        sampling_method: str = "rejection",
        sir_oversample: int = 1024,
    ) -> None:
        r"""
        Initialize the simulation informed prior.
        Args:
            prior: Prior distribution, will be used as proposal distribution whose
                samples will be evaluated by the classifier.
            classifier: Classifier that is evaluated to check if a parameter set
                $\theta$ is valid or not.
            validation_theta: The parameters in the latest validation set that the
                classifier was trained on. Used to calibrate the classifier threshold.
            validation_label: The labels in the latest validation set that the
                classifier was trained on. Used to calibrate the classifier threshold.
        """
        self._prior = prior
        self._posterior = posterior
        self._posterior_thr = None
        self.tune_threshold(allowed_false_negatives, num_samples_to_estimate_support)
        self.use_constrained_prior = use_constrained_prior
        if not self.use_constrained_prior:
            self._proposal = self._prior
        else:
            assert isinstance(self._prior, BoxUniform), "Prior must be BoxUniform"
            self._proposal = self.constrain_prior(
                num_samples_to_estimate_support, constrained_prior_quanitle
            )

        self.sampling_method = sampling_method
        self.sir_oversample = sir_oversample

    def constrain_prior(
        self, num_samples_to_estimate_support: int = 10_000, quant: float = 0.0
    ):
        samples = self._posterior.sample((num_samples_to_estimate_support,))
        lower_quantile = torch.quantile(samples, tensor(quant), dim=0)
        upper_quanitle = torch.quantile(samples, tensor(1 - quant), dim=0)

        return BoxUniform(lower_quantile, upper_quanitle)

    def tune_threshold(
        self,
        allowed_false_negatives: float = 0.0,
        num_samples_to_estimate_support: int = 10_000,
    ) -> None:
        samples = self._posterior.sample((num_samples_to_estimate_support,))
        log_probs = self._posterior.log_prob(samples)
        sorted_log_probs, _ = torch.sort(log_probs)
        self.thr = sorted_log_probs[
            int(allowed_false_negatives * num_samples_to_estimate_support)
        ]

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        show_progress_bars: bool = False,
        sampling_batch_size: int = 10_000,
        return_acceptance_rate: bool = False,
        return_iw: bool = False,
    ) -> Tensor:
        """
        Return samples from the `RestrictedPrior`.
        Samples are obtained by sampling from the prior, evaluating them under the
        trained classifier (`RestrictionEstimator`) and using only those that were
        accepted.
        Args:
            sample_shape: Shape of the returned samples.
            show_progress_bars: Whether or not to show a progressbar during sampling.
            max_sampling_batch_size: Batch size for drawing samples from the posterior.
            Takes effect only in the second iteration of the loop below, i.e., in case
            of leakage or `num_samples>max_sampling_batch_size`. Larger batch size
            speeds up sampling.
        Returns:
            Samples from the `RestrictedPrior`.
        """

        if self.sampling_method == "rejection":
            num_samples = torch.Size(sample_shape).numel()
            num_sampled_total, num_remaining = tensor(0), num_samples
            accepted, acceptance_rate = [], float("Nan")

            # Progress bar can be skipped.
            pbar = tqdm(
                disable=not show_progress_bars,
                total=num_samples,
                desc=f"Drawing {num_samples} posterior samples",
            )

            # To cover cases with few samples without leakage:
            while num_remaining > 0:
                # Sample and reject.
                candidates = self._proposal.sample((sampling_batch_size,)).reshape(
                    sampling_batch_size, -1
                )
                are_accepted_by_classifier = self.predict(candidates)
                samples = candidates[are_accepted_by_classifier.bool()]
                accepted.append(samples)

                # Update.
                num_sampled_total += tensor(sampling_batch_size)
                num_remaining -= samples.shape[0]
                pbar.update(samples.shape[0])

                # To avoid endless sampling when leakage is high, we raise a warning if the
                # acceptance rate is too low after the first 1_000 samples.
                acceptance_rate = (num_samples - num_remaining) / num_sampled_total
                print("Remaining:", num_remaining, end="\r")

            pbar.close()
            print(
                f"The classifier rejected {(1.0 - acceptance_rate) * 100:.4f}% of all "
                f"samples. You will get a speed-up of "
                f"{(1.0 / acceptance_rate - 1.0) * 100:.1f}%.",
            )

            # When in case of leakage a batch size was used there could be too many samples.
            samples = torch.cat(accepted)[:num_samples]
            assert (
                samples.shape[0] == num_samples
            ), "Number of accepted samples must match required samples."

            if return_acceptance_rate:
                if self.use_constrained_prior:
                    proposal_lower = self._proposal.support.base_constraint.lower_bound
                    proposal_upper = self._proposal.support.base_constraint.upper_bound
                    proposal_range = proposal_upper - proposal_lower
                    prior_lower = self._prior.support.base_constraint.lower_bound
                    prior_upper = self._prior.support.base_constraint.upper_bound
                    prior_range = prior_upper - prior_lower
                    prior_acceptance_rate = torch.prod(proposal_range / prior_range)
                    log_prior_acceptance = torch.log10(prior_acceptance_rate)
                    log_proposal_acceptance = torch.log10(acceptance_rate)
                    log_acceptance = log_prior_acceptance + log_proposal_acceptance
                else:
                    log_acceptance = torch.log10(acceptance_rate)
                if return_iw:
                    return samples, log_acceptance, torch.randn((num_samples, self.sir_oversample))
                else:
                    return samples, log_acceptance
            else:
                return samples

        elif self.sampling_method == "sir":
            num_samples = torch.Size(sample_shape).numel()
            num_sampled_total, num_remaining = tensor(0), num_samples
            accepted, acceptance_rate = [], float("Nan")

            # To cover cases with few samples without leakage:
            # Progress bar can be skipped.
            pbar = tqdm(
                disable=True,
                total=num_samples,
                desc=f"Drawing {num_samples} posterior samples",
            )

            all_samples = []
            all_importance_weights = []
            while num_remaining > 0:
                batch_size = int(
                    torch.ceil(
                        torch.tensor(sampling_batch_size / self.sir_oversample)
                    ).item()
                )

                posterior_samples = self._posterior.sample(
                    (batch_size * self.sir_oversample,))
                posterior_log_probs = self._posterior.log_prob(posterior_samples)
                prior_log_probs = self._prior.log_prob(posterior_samples)
                truncated_prior_log_probs = prior_log_probs
                truncated_prior_log_probs[posterior_log_probs < self.thr] = -float(
                    "inf"
                )
                ratio = truncated_prior_log_probs - posterior_log_probs
                reshaped_ratio = torch.reshape(ratio, (batch_size, self.sir_oversample))
                all_importance_weights.append(reshaped_ratio)
                cat_dist = torch.distributions.Categorical(logits=reshaped_ratio)
                categorical_samples = cat_dist.sample((1,))[0, :]
                reshaped_posterior_samples = torch.reshape(
                    posterior_samples, (batch_size, self.sir_oversample, -1)
                )
                selected_posterior_samples = reshaped_posterior_samples[
                    torch.arange(batch_size), categorical_samples
                ]
                num_remaining -= batch_size
                all_samples.append(selected_posterior_samples)
                pbar.update(batch_size)
            pbar.close()
            all_samples = torch.cat(all_samples)
            all_samples = all_samples[:num_samples]
            all_importance_weights = torch.cat(all_importance_weights)
            assert all_samples.shape[0] == num_samples
            if return_acceptance_rate:
                self.sampling_method = "rejection"
                _, acceptance_rate = self.sample(
                    sample_shape, return_acceptance_rate=True
                )
                self.sampling_method = "sir"
                if return_iw:
                    return all_samples, acceptance_rate, all_importance_weights
                else:
                    return all_samples, acceptance_rate
            else:
                return all_samples
        else:
            raise ValueError

    def predict(self, theta: Tensor) -> Tensor:
        r"""
        Run classifier to predict whether the parameter set is `invalid` or `valid`.
        Args:
            theta: Parameters whose label to predict.
        Returns:
            Integers that indicate whether the parameter set is predicted to be
            `invalid` (=0) or `valid` (=1).
        """

        theta_log_probs = self._posterior.log_prob(theta)
        predictions = theta_log_probs > self.thr
        return predictions.int()
