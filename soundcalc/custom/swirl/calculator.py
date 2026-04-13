from __future__ import annotations

import math
from dataclasses import dataclass

from soundcalc.common.fields import FieldParams
from soundcalc.common.utils import apply_grinding
from soundcalc.pcs.whir import WHIR
from soundcalc.proxgaps.johnson_bound import JohnsonBoundRegime
from soundcalc.proxgaps.proxgaps_regime import ProximityGapsRegime


SWIRL_SECURITY_BITS_TARGET = 100
SWIRL_WHIR_K = 4
SWIRL_WHIR_MAX_LOG_FINAL_POLY_LEN = 10
SWIRL_QUERY_PHASE_POW_BITS = 20
SWIRL_MAX_CONSTRAINT_DEGREE = 4


@dataclass(frozen=True)
class SWIRLWhirRoundConfig:
    num_queries: int


@dataclass(frozen=True)
class SWIRLWhirProximityMode:
    kind: str
    m: int | None = None

    def build_regime(self, field: FieldParams) -> ProximityGapsRegime:
        if self.kind == "unique":
            return SWIRLUniqueDecodingRegime(field)
        if self.kind == "list":
            if self.m is None:
                raise ValueError("list-decoding mode requires multiplicity m")
            return SWIRLListDecodingRegime(field, self.m)
        raise ValueError(f"Unknown SWIRL proximity mode: {self.kind}")

    def whir_query_security_bits(self, num_queries: int, log_inv_rate: int) -> float:
        rho = 2.0 ** (-log_inv_rate)
        if self.kind == "unique":
            max_agreement = (1.0 + rho) / 2.0
        else:
            if self.m is None:
                raise ValueError("list-decoding mode requires multiplicity m")
            max_agreement = math.sqrt(rho * (1.0 + 1.0 / self.m)) + 1e-6

        max_agreement = max(max_agreement, math.ldexp(1.0, -1022))
        return -(num_queries * math.log2(max_agreement))


@dataclass(frozen=True)
class SWIRLWhirConfig:
    k: int
    rounds: list[SWIRLWhirRoundConfig]
    mu_pow_bits: int
    query_phase_pow_bits: int
    folding_pow_bits: int
    proximity: SWIRLWhirProximityMode


@dataclass(frozen=True)
class SWIRLLogUpSecurityParameters:
    """
    SWIRL's interaction LogUp bound expressed through the shared error-to-bits path.
    """

    max_interaction_count: int
    log_max_message_length: int
    pow_bits: int

    def max_message_length(self) -> int:
        return 1 << self.log_max_message_length

    def get_soundness_error(self, challenge_field_size: int, list_size: float = 1.0) -> float:
        return (
            2.0
            * self.max_interaction_count
            * self.max_message_length()
            / (challenge_field_size * list_size)
        )

    def get_soundness_bits(self, challenge_field_size: int, list_size: float = 1.0) -> float:
        grounded_error = apply_grinding(
            self.get_soundness_error(challenge_field_size, list_size),
            self.pow_bits,
        )
        return -math.log2(grounded_error)


@dataclass(frozen=True)
class SWIRLSystemParams:
    l_skip: int
    n_stack: int
    w_stack: int
    log_blowup: int
    whir: SWIRLWhirConfig
    logup: SWIRLLogUpSecurityParameters
    max_constraint_degree: int

    def log_stacked_height(self) -> int:
        return self.l_skip + self.n_stack


@dataclass(frozen=True)
class SWIRLWhirDetails:
    mu_batching_bits: float
    pre_loop_proximity_gaps_bits: float
    shared_whir_bits: float
    shared_fold_bits: float | None
    shared_ood_bits: float | None
    shared_shift_bits: float | None
    shared_final_bits: float | None


@dataclass(frozen=True)
class SWIRLSoundnessResult:
    logup_bits: float
    gkr_sumcheck_bits: float
    gkr_batching_bits: float
    zerocheck_sumcheck_bits: float
    constraint_batching_bits: float
    stacked_reduction_bits: float
    whir_bits: float
    whir_details: SWIRLWhirDetails
    total_bits: float


class SWIRLUniqueDecodingRegime(ProximityGapsRegime):
    """
    SWIRL's unique-decoding WHIR bound uses `n / |F|` for the proximity-gap term.

    This differs from soundcalc's generic UDR MCA bound, so it stays isolated here.
    """

    def identifier(self) -> str:
        return "SWIRL-UDR"

    def get_proximity_parameter(self, rate: float, dimension: int) -> float:
        return (1.0 - rate) / 2.0

    def get_max_list_size(self, rate: float, dimension: int) -> int:
        return 1

    def get_error_powers(self, rate: float, dimension: int, batch_size: int) -> float:
        if batch_size <= 1:
            return 0.0
        return self.get_error_linear(rate, dimension) * (batch_size - 1)

    def get_error_linear(self, rate: float, dimension: int) -> float:
        code_length = dimension / rate
        return code_length / self.field.F

    def get_error_multilinear(self, rate: float, dimension: int, batch_size: int) -> float:
        if batch_size <= 1:
            return 0.0
        return self.get_error_linear(rate, dimension) * math.ceil(math.log2(batch_size))


class SWIRLListDecodingRegime(JohnsonBoundRegime):
    """
    SWIRL uses the default BCHKS25 closed-form `a`-bound with an explicit multiplicity `m`.

    The shared Johnson-bound implementation already contains the default `a`-bound algebra.
    We only override the parts where SWIRL fixes `m` directly and uses `D_Y` as the list-size
    proxy for subsequent soundness terms.
    """

    def __init__(self, field: FieldParams, m: int):
        super().__init__(field)
        self.explicit_m = max(m, 1)

    def identifier(self) -> str:
        return f"SWIRL-LDR(m={self.explicit_m})"

    def get_proximity_parameter(self, rate: float, dimension: int) -> float:
        sqrt_rate = math.sqrt(rate)
        return 1.0 - sqrt_rate - (sqrt_rate / (2.0 * self.explicit_m))

    def get_m(self, rate: float, dimension: int) -> int:
        return self.explicit_m

    def get_max_list_size(self, rate: float, dimension: int) -> float:
        sqrt_rate = math.sqrt(rate)
        return (self.explicit_m + 0.5) / sqrt_rate


def _challenge_field_bits(field: FieldParams) -> float:
    return field.field_extension_degree * math.log2(field.p)


def _finite_min(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return math.inf
    return min(finite_values)


def _optional_bits(value: float) -> float | None:
    if math.isfinite(value):
        return value
    return None


def _pre_main_loop_proximity_gaps_bits(
    params: SWIRLSystemParams,
    whir: WHIR,
    regime: ProximityGapsRegime,
) -> float:
    min_bits = math.inf
    for sub_round in range(params.whir.k):
        rate, dimension = whir._get_code_for_iteration_and_round(0, sub_round + 1)
        proximity_error = regime.get_error_powers(rate, dimension, 2)
        proximity_bits = -math.log2(proximity_error) + params.whir.folding_pow_bits
        min_bits = min(min_bits, proximity_bits)
    return min_bits


def _get_shared_whir_details(whir: WHIR, regime: ProximityGapsRegime) -> SWIRLWhirDetails:
    levels = whir.get_pcs_security_levels(regime)

    shared_fold_bits = math.inf
    shared_ood_bits = math.inf
    shared_shift_bits = math.inf
    shared_final_bits = math.inf

    for label, bits in levels.items():
        bits = float(bits)
        if label == "batching":
            continue
        if label.startswith("fold(i=0,"):
            continue
        if label.startswith("fold(i="):
            shared_fold_bits = min(shared_fold_bits, bits)
        elif label.startswith("OOD("):
            shared_ood_bits = min(shared_ood_bits, bits)
        elif label.startswith("Shift("):
            shared_shift_bits = min(shared_shift_bits, bits)
        elif label == "fin":
            shared_final_bits = bits

    shared_whir_bits = _finite_min([
        shared_fold_bits,
        shared_ood_bits,
        shared_shift_bits,
        shared_final_bits,
    ])

    return SWIRLWhirDetails(
        mu_batching_bits=math.inf,
        pre_loop_proximity_gaps_bits=math.inf,
        shared_whir_bits=shared_whir_bits,
        shared_fold_bits=_optional_bits(shared_fold_bits),
        shared_ood_bits=_optional_bits(shared_ood_bits),
        shared_shift_bits=_optional_bits(shared_shift_bits),
        shared_final_bits=_optional_bits(shared_final_bits),
    )


def build_swirl_system_params(
    *,
    l_skip: int,
    n_stack: int,
    w_stack: int,
    log_blowup: int,
    folding_pow_bits: int,
    mu_pow_bits: int,
    proximity: SWIRLWhirProximityMode,
    logup: SWIRLLogUpSecurityParameters,
    security_bits_target: int = SWIRL_SECURITY_BITS_TARGET,
) -> SWIRLSystemParams:
    protocol_security_level = security_bits_target - SWIRL_QUERY_PHASE_POW_BITS
    log_stacked_height = l_skip + n_stack
    num_rounds = math.ceil(
        max(log_stacked_height - SWIRL_WHIR_MAX_LOG_FINAL_POLY_LEN, 0) / SWIRL_WHIR_K
    )

    rounds: list[SWIRLWhirRoundConfig] = []
    log_inv_rate = log_blowup
    for _round in range(num_rounds):
        per_query_bits = proximity.whir_query_security_bits(1, log_inv_rate)
        num_queries = math.ceil(protocol_security_level / per_query_bits)
        rounds.append(SWIRLWhirRoundConfig(num_queries=num_queries))
        log_inv_rate += SWIRL_WHIR_K - 1

    return SWIRLSystemParams(
        l_skip=l_skip,
        n_stack=n_stack,
        w_stack=w_stack,
        log_blowup=log_blowup,
        whir=SWIRLWhirConfig(
            k=SWIRL_WHIR_K,
            rounds=rounds,
            mu_pow_bits=mu_pow_bits,
            query_phase_pow_bits=SWIRL_QUERY_PHASE_POW_BITS,
            folding_pow_bits=folding_pow_bits,
            proximity=proximity,
        ),
        logup=logup,
        max_constraint_degree=SWIRL_MAX_CONSTRAINT_DEGREE,
    )


def calculate_swirl_soundness(
    *,
    params: SWIRLSystemParams,
    field: FieldParams,
    whir: WHIR,
    max_num_constraints_per_air: int,
    num_airs: int,
    max_log_trace_height: int,
    num_trace_columns: int,
    max_interactions_per_air: int,
) -> SWIRLSoundnessResult:
    del max_interactions_per_air

    challenge_field_bits = _challenge_field_bits(field)
    regime = params.whir.proximity.build_regime(field)

    mu_batching_bits = -math.log2(whir._get_batching_error(regime))
    pre_loop_proximity_gaps_bits = _pre_main_loop_proximity_gaps_bits(params, whir, regime)
    initial_list_size = whir._get_list_size_for_iteration_and_round(0, 0, regime)
    log2_list_size = math.log2(initial_list_size)

    logup_bits = params.logup.get_soundness_bits(
        field.F,
        initial_list_size,
    )

    gkr_sumcheck_bits = challenge_field_bits - math.log2(3.0)
    gkr_batching_bits = challenge_field_bits

    univariate_degree = (params.max_constraint_degree + 1) * ((1 << params.l_skip) - 1)
    multilinear_degree = params.max_constraint_degree + 1
    zerocheck_sumcheck_bits = challenge_field_bits - math.log2(max(univariate_degree, multilinear_degree))

    n_max = max_log_trace_height - params.l_skip
    poly_degree_sum = ((1 << params.l_skip) - 1) + n_max
    poly_identity_bits = challenge_field_bits - math.log2(poly_degree_sum)
    zerocheck_bits = log2_list_size + min(zerocheck_sumcheck_bits, poly_identity_bits)

    lambda_batching_bits = challenge_field_bits - math.log2(max_num_constraints_per_air)
    mu_constraint_bits = challenge_field_bits - math.log2(3.0 * num_airs)
    constraint_batching_bits = log2_list_size + min(lambda_batching_bits, mu_constraint_bits)

    stacked_batching_bits = challenge_field_bits - math.log2(2.0 * num_trace_columns)
    stacked_univariate_bits = challenge_field_bits - math.log2(2.0 * ((1 << params.l_skip) - 1))
    stacked_multilinear_bits = challenge_field_bits - 1.0
    stacked_reduction_bits = log2_list_size + min(
        stacked_batching_bits,
        stacked_univariate_bits,
        stacked_multilinear_bits,
    )

    shared_whir_details = _get_shared_whir_details(whir, regime)
    whir_details = SWIRLWhirDetails(
        mu_batching_bits=mu_batching_bits,
        pre_loop_proximity_gaps_bits=pre_loop_proximity_gaps_bits,
        shared_whir_bits=shared_whir_details.shared_whir_bits,
        shared_fold_bits=shared_whir_details.shared_fold_bits,
        shared_ood_bits=shared_whir_details.shared_ood_bits,
        shared_shift_bits=shared_whir_details.shared_shift_bits,
        shared_final_bits=shared_whir_details.shared_final_bits,
    )

    whir_bits = _finite_min([
        mu_batching_bits,
        pre_loop_proximity_gaps_bits,
        shared_whir_details.shared_whir_bits,
    ])

    total_bits = min(
        logup_bits,
        gkr_sumcheck_bits,
        gkr_batching_bits,
        zerocheck_bits,
        constraint_batching_bits,
        stacked_reduction_bits,
        whir_bits,
    )

    return SWIRLSoundnessResult(
        logup_bits=logup_bits,
        gkr_sumcheck_bits=gkr_sumcheck_bits,
        gkr_batching_bits=gkr_batching_bits,
        zerocheck_sumcheck_bits=zerocheck_bits,
        constraint_batching_bits=constraint_batching_bits,
        stacked_reduction_bits=stacked_reduction_bits,
        whir_bits=whir_bits,
        whir_details=whir_details,
        total_bits=total_bits,
    )
