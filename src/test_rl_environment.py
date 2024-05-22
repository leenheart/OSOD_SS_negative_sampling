import pytest
import torch
import numpy as np
from rl_environment import RL_Environment



@pytest.fixture(scope="session")
def rl_env_empty():
    return RL_Environment(None, 0)

# ------------- init --------------

def test_rl_env_init_empty(rl_env_empty):

    assert rl_env_empty.nb_actions == 0
    assert rl_env_empty.nb_predictions_intervals == 10
    assert len(rl_env_empty.predictions_intervals) == 9

# ------------- get indexes from interval --------------

def test_rl_env_get_indexes_from_interval_empty(rl_env_empty):

    values = np.array([])
    interval = rl_env_empty.predictions_intervals

    indexes = RL_Environment.get_indexes_from_interval(values, interval)

    np.testing.assert_array_equal(indexes, np.array([]))

def test_rl_env_get_indexes_from_interval_one(rl_env_empty):

    values = [0.55]
    interval = rl_env_empty.predictions_intervals

    indexes = RL_Environment.get_indexes_from_interval(values, interval)

    np.testing.assert_array_equal(indexes, [5])

def test_rl_env_get_indexes_from_interval_multiple(rl_env_empty):

    values = [0.5, 0, 0.99, 0.32, 0.26]
    interval = rl_env_empty.predictions_intervals

    indexes = RL_Environment.get_indexes_from_interval(values, interval)

    np.testing.assert_array_equal(indexes, [5, 0, 9, 3, 2])

def test_rl_env_get_indexes_from_interval_multiple2(rl_env_empty):

    values = [0.5, 0, 0.99, 0.32, 0.26, 2.5, 6000, -20]
    interval = rl_env_empty.color_contrast_intervals

    indexes = RL_Environment.get_indexes_from_interval(values, interval)

    np.testing.assert_array_equal(indexes, [2, 0, 3, 1, 1, 6, 11, 0])

# ---------------- get states +--------------------------------

def test_rl_env_get_states_empty(rl_env_empty):

    predictions = np.array([])

    states = rl_env_empty.get_states_batch(predictions)

    np.testing.assert_array_equal(states, np.array([]))


def test_rl_env_get_states_empty2(rl_env_empty):

    predictions = [{"scores": np.array([]), "boxes":[], "custom_scores": {"color_contrast": np.array([])}, "boxes_area": np.array([])}]

    states = rl_env_empty.get_states_batch(predictions)

    np.testing.assert_array_equal(states, [[]])

def test_rl_env_get_states_empty3(rl_env_empty):

    predictions = [{"scores": torch.tensor([0]), "boxes": torch.tensor([[0, 0, 0, 0]]), "custom_scores": {"color_contrast": np.array([0])}, "boxes_area": np.array([0])}]

    states = rl_env_empty.get_states_batch(predictions)

    np.testing.assert_array_equal(states, [[0]])

def test_rl_env_get_states_one(rl_env_empty):

    predictions = [{"scores": torch.tensor([0.55]), "boxes": torch.tensor([[0, 0, 0, 0]]), "custom_scores": {"color_contrast": np.array([0.8])}, "boxes_area": np.array([100*100])}]

    states = rl_env_empty.get_states_batch(predictions)

    np.testing.assert_array_equal(states, [[5 + 2 * 10 + 3 * 10 * 3]])

def test_rl_env_get_states_multiple(rl_env_empty):

    predictions = [{"scores": torch.tensor([0.55, 0.999]), "boxes": torch.tensor([[0, 0, 0, 0]]), "custom_scores": {"color_contrast": np.array([0.8, 15661])}, "boxes_area": np.array([100*100, 158*623])}]

    states = rl_env_empty.get_states_batch(predictions)

    np.testing.assert_array_equal(states, [[5 + 2 * 10 + 3 * 10 * 3, 9 + 2 * 10 + 11 * 10 * 3]])



# ---------------- get rewards +--------------------------------

def test_rl_env_get_rewards_empty(rl_env_empty):

    predictions = np.array([])

    rewards = rl_env_empty.get_rewards(predictions)

    np.testing.assert_array_equal(rewards, np.array([]))


def test_rl_env_get_rewards_empty2(rl_env_empty):

    predictions = [{"scores": np.array([]), "boxes":[], "custom_scores": {"color_contrast": np.array([]), "edge_density": np.array([])}, "boxes_area": np.array([]), "tags": np.array([], dtype=bool), "IoU": np.array([])}]

    rewards = rl_env_empty.get_rewards(predictions)

    np.testing.assert_array_equal(rewards, [[]])

def test_rl_env_get_rewards_empty3(rl_env_empty):

    predictions = [{"scores": torch.tensor([0]), "boxes": torch.tensor([[0, 0, 0, 0]]), "custom_scores": {"color_contrast": np.array([0]), "edge_density": np.array([0])}, "boxes_area": np.array([0]), 'tags': np.array([False]), "IoU": np.array([0])}]

    rewards = rl_env_empty.get_rewards(predictions)

    np.testing.assert_array_equal(rewards, [[[ rl_env_empty.bad_known_reward, 0, 0]]])

def test_rl_env_get_rewards_one_known_true(rl_env_empty):

    predictions = [{"scores": torch.tensor([0.55]), "boxes": torch.tensor([[0, 0, 0, 0]]), "custom_scores": {"color_contrast": np.array([0.8]), "edge_density": np.array([15])}, "boxes_area": np.array([100*100]), 'tags': np.array([True]), "IoU": np.array([0.56])}]

    rewards = rl_env_empty.get_rewards(predictions)

    np.testing.assert_array_equal(rewards, [[[0.56, -15 + rl_env_empty.bad_known_reward, -15 + rl_env_empty.bad_known_reward]]])


def test_rl_env_get_rewards_one_known_false(rl_env_empty):

    predictions = [{"scores": torch.tensor([0.55]), "boxes": torch.tensor([[0, 0, 0, 0]]), "custom_scores": {"color_contrast": np.array([0.8]), "edge_density": np.array([15])}, "boxes_area": np.array([100*100]), 'tags': np.array([False]), "IoU": np.array([0.56])}]

    rewards = rl_env_empty.get_rewards(predictions)

    np.testing.assert_array_equal(rewards, [[[rl_env_empty.bad_known_reward, 15, 15]]])

