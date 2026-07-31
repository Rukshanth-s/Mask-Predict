# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""Linear-chain CRF primitives, checked against brute-force path enumeration.

The dynamic programs below are the part of the CRF head that is easy to get
subtly wrong, so every one of them is compared against an oracle that simply
enumerates all ``k ** T`` paths and sums/maximises over them directly.
"""

import itertools

import pytest
import torch

from fairseq.modules.dynamic_crf_output_layer import (
    backward_algorithm, forward_algorithm, log_marginals, viterbi_decode,
)


def enumerate_paths(emissions, transitions):
    """Score every path through a single sequence.

    Args:
        emissions: ``T x k`` beam label scores.
        transitions: ``(T-1) x k x k``, ``transitions[i, a, c]`` scoring slot
            ``a`` at position ``i`` followed by slot ``c`` at position ``i+1``.

    Returns:
        ``(paths, scores)`` with ``paths`` a list of ``T``-tuples of slot indices.
    """
    length, beam = emissions.shape
    paths = list(itertools.product(range(beam), repeat=length))
    scores = []
    for path in paths:
        score = sum(emissions[i, path[i]] for i in range(length))
        for i in range(length - 1):
            score = score + transitions[i, path[i], path[i + 1]]
        scores.append(score)
    return paths, torch.stack(scores)


def brute_force(emissions, transitions):
    """Oracle: exact log-partition, Viterbi solution and posterior marginals."""
    length, beam = emissions.shape
    paths, scores = enumerate_paths(emissions, transitions)

    log_partition = torch.logsumexp(scores, dim=0)
    best = int(scores.argmax())

    marginals = torch.full((length, beam), float('-inf'), dtype=emissions.dtype)
    for i in range(length):
        for c in range(beam):
            selected = [scores[j] for j, path in enumerate(paths) if path[i] == c]
            marginals[i, c] = torch.logsumexp(torch.stack(selected), dim=0) - log_partition

    return {
        'log_partition': log_partition,
        'best_score': scores[best],
        'best_path': torch.tensor(paths[best]),
        'log_marginals': marginals,
    }


@pytest.fixture
def chain():
    """A small random chain: 2 sequences, length 4, beam 3 (81 paths each)."""
    torch.manual_seed(7)
    emissions = torch.randn(2, 4, 3, dtype=torch.float64)
    transitions = torch.randn(2, 3, 3, 3, dtype=torch.float64)
    return emissions, transitions


# --- forward algorithm ------------------------------------------------------

def test_log_partition_matches_brute_force(chain):
    emissions, transitions = chain
    _, log_partition = forward_algorithm(emissions, transitions)
    for b in range(emissions.size(0)):
        expected = brute_force(emissions[b], transitions[b])['log_partition']
        torch.testing.assert_close(log_partition[b], expected)


def test_alpha_shape_and_final_column_gives_the_partition(chain):
    emissions, transitions = chain
    alpha, log_partition = forward_algorithm(emissions, transitions)
    assert alpha.shape == emissions.shape
    torch.testing.assert_close(torch.logsumexp(alpha[:, -1], dim=-1), log_partition)


def test_alpha_first_column_is_the_first_emission(chain):
    emissions, transitions = chain
    alpha, _ = forward_algorithm(emissions, transitions)
    torch.testing.assert_close(alpha[:, 0], emissions[:, 0])


def test_single_position_has_no_transitions():
    emissions = torch.randn(2, 1, 3, dtype=torch.float64)
    transitions = torch.zeros(2, 0, 3, 3, dtype=torch.float64)
    _, log_partition = forward_algorithm(emissions, transitions)
    torch.testing.assert_close(log_partition, torch.logsumexp(emissions[:, 0], dim=-1))


def test_zero_transitions_make_positions_independent():
    """With no transition scores the CRF degenerates to independent softmaxes,
    so the partition factorises into a product over positions."""
    torch.manual_seed(1)
    emissions = torch.randn(2, 4, 3, dtype=torch.float64)
    transitions = torch.zeros(2, 3, 3, 3, dtype=torch.float64)
    _, log_partition = forward_algorithm(emissions, transitions)
    expected = torch.logsumexp(emissions, dim=-1).sum(dim=-1)
    torch.testing.assert_close(log_partition, expected)


def test_forward_algorithm_is_differentiable(chain):
    emissions, transitions = chain
    emissions = emissions.clone().requires_grad_(True)
    transitions = transitions.clone().requires_grad_(True)
    _, log_partition = forward_algorithm(emissions, transitions)
    log_partition.sum().backward()
    assert emissions.grad is not None and emissions.grad.abs().sum() > 0
    assert transitions.grad is not None and transitions.grad.abs().sum() > 0


# --- viterbi ----------------------------------------------------------------

def test_viterbi_score_and_path_match_brute_force(chain):
    emissions, transitions = chain
    path, score = viterbi_decode(emissions, transitions)
    assert path.shape == emissions.shape[:2]
    assert path.dtype == torch.long
    for b in range(emissions.size(0)):
        expected = brute_force(emissions[b], transitions[b])
        torch.testing.assert_close(score[b], expected['best_score'])
        assert torch.equal(path[b], expected['best_path'])


def test_viterbi_score_is_the_score_of_the_returned_path(chain):
    emissions, transitions = chain
    path, score = viterbi_decode(emissions, transitions)
    for b in range(emissions.size(0)):
        manual = sum(emissions[b, i, path[b, i]] for i in range(emissions.size(1)))
        for i in range(emissions.size(1) - 1):
            manual = manual + transitions[b, i, path[b, i], path[b, i + 1]]
        torch.testing.assert_close(score[b], manual)


def test_viterbi_never_scores_below_the_log_partition(chain):
    """A single path's score can never exceed the sum over all paths."""
    emissions, transitions = chain
    _, score = viterbi_decode(emissions, transitions)
    _, log_partition = forward_algorithm(emissions, transitions)
    assert bool((score <= log_partition + 1e-9).all())


def test_viterbi_with_zero_transitions_is_per_position_argmax():
    torch.manual_seed(3)
    emissions = torch.randn(2, 4, 3, dtype=torch.float64)
    transitions = torch.zeros(2, 3, 3, 3, dtype=torch.float64)
    path, _ = viterbi_decode(emissions, transitions)
    assert torch.equal(path, emissions.argmax(dim=-1))


# --- backward algorithm and marginals ---------------------------------------

def test_marginals_match_brute_force(chain):
    emissions, transitions = chain
    marginals = log_marginals(emissions, transitions)
    assert marginals.shape == emissions.shape
    for b in range(emissions.size(0)):
        expected = brute_force(emissions[b], transitions[b])['log_marginals']
        torch.testing.assert_close(marginals[b], expected)


def test_marginals_are_normalised_at_every_position(chain):
    emissions, transitions = chain
    marginals = log_marginals(emissions, transitions)
    total = torch.logsumexp(marginals, dim=-1)
    torch.testing.assert_close(total, torch.zeros_like(total))


def test_beta_final_column_is_zero(chain):
    emissions, transitions = chain
    beta = backward_algorithm(emissions, transitions)
    assert beta.shape == emissions.shape
    torch.testing.assert_close(beta[:, -1], torch.zeros_like(beta[:, -1]))


def test_forward_and_backward_agree_on_the_partition_at_every_position(chain):
    emissions, transitions = chain
    alpha, log_partition = forward_algorithm(emissions, transitions)
    beta = backward_algorithm(emissions, transitions)
    for i in range(emissions.size(1)):
        torch.testing.assert_close(
            torch.logsumexp(alpha[:, i] + beta[:, i], dim=-1), log_partition)


def test_marginals_with_zero_transitions_are_the_emission_softmax():
    torch.manual_seed(5)
    emissions = torch.randn(2, 4, 3, dtype=torch.float64)
    transitions = torch.zeros(2, 3, 3, 3, dtype=torch.float64)
    marginals = log_marginals(emissions, transitions)
    torch.testing.assert_close(marginals, torch.log_softmax(emissions, dim=-1))


# --- padding ----------------------------------------------------------------

def test_masked_partition_equals_the_shorter_unpadded_sequence():
    torch.manual_seed(11)
    emissions = torch.randn(1, 5, 3, dtype=torch.float64)
    transitions = torch.randn(1, 4, 3, 3, dtype=torch.float64)
    mask = torch.tensor([[True, True, True, False, False]])

    _, padded = forward_algorithm(emissions, transitions, mask)
    _, unpadded = forward_algorithm(emissions[:, :3], transitions[:, :2])
    torch.testing.assert_close(padded, unpadded)


def test_masked_viterbi_matches_the_shorter_unpadded_sequence():
    torch.manual_seed(13)
    emissions = torch.randn(1, 5, 3, dtype=torch.float64)
    transitions = torch.randn(1, 4, 3, 3, dtype=torch.float64)
    mask = torch.tensor([[True, True, True, False, False]])

    path, score = viterbi_decode(emissions, transitions, mask)
    short_path, short_score = viterbi_decode(emissions[:, :3], transitions[:, :2])
    torch.testing.assert_close(score, short_score)
    assert torch.equal(path[:, :3], short_path)


def test_masked_marginals_match_the_shorter_unpadded_sequence():
    torch.manual_seed(17)
    emissions = torch.randn(1, 5, 3, dtype=torch.float64)
    transitions = torch.randn(1, 4, 3, 3, dtype=torch.float64)
    mask = torch.tensor([[True, True, True, False, False]])

    marginals = log_marginals(emissions, transitions, mask)
    short = log_marginals(emissions[:, :3], transitions[:, :2])
    torch.testing.assert_close(marginals[:, :3], short)


def test_padding_of_one_row_does_not_affect_another():
    """Rows in a batch must be independent regardless of their lengths."""
    torch.manual_seed(19)
    emissions = torch.randn(3, 5, 3, dtype=torch.float64)
    transitions = torch.randn(3, 4, 3, 3, dtype=torch.float64)
    mask = torch.tensor([
        [True, True, True, True, True],
        [True, True, True, False, False],
        [True, True, False, False, False],
    ])

    _, batched = forward_algorithm(emissions, transitions, mask)
    lengths = [5, 3, 2]
    for b, length in enumerate(lengths):
        _, alone = forward_algorithm(
            emissions[b:b + 1, :length], transitions[b:b + 1, :length - 1])
        torch.testing.assert_close(batched[b:b + 1], alone)


def test_all_positions_unmasked_is_the_same_as_no_mask(chain):
    emissions, transitions = chain
    mask = torch.ones(emissions.shape[:2], dtype=torch.bool)
    _, with_mask = forward_algorithm(emissions, transitions, mask)
    _, without = forward_algorithm(emissions, transitions)
    torch.testing.assert_close(with_mask, without)
