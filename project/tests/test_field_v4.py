import random
import unittest
from unittest.mock import patch

import numpy as np

from src import AgentV4, FieldV4
from src.factory import get_agent_class, get_field_class


class FieldV4Tests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())

    def make_field(self, **kwargs):
        return FieldV4(initial_agents=0, initial_food_prob=0, step_food_prob=0, **kwargs)

    def make_agent(self, x=0, y=0, energy=0, action=(0, 0)):
        agent = AgentV4(x, y, mutation_rate=0)
        agent.energy = energy
        agent.weights[:] = 0
        # With a neighbor present, choose the specified action.
        agent.weights[:, 1] = action
        return agent

    def test_exports_and_policy_interface(self):
        self.assertIs(get_agent_class('AgentV4'), AgentV4)
        self.assertIs(get_field_class('FieldV4'), FieldV4)
        agent = self.make_agent(action=(0, 1))
        np.testing.assert_array_equal(agent.decide_action(0, 1, 0, 0), [0, 1])
        self.assertIsInstance(FieldV4(initial_agents=1).agents[0], AgentV4)

    def test_snapshot_is_copied_read_only_and_calculation_is_order_independent(self):
        field = self.make_field()
        field.agents = [self.make_agent(0, 0, 10, (0, 1)), self.make_agent(1, 0, 20)]
        field.grid_food[0, 0] = 5
        state = field.snapshot()
        random.seed(42)
        inputs = field.prepare_step_inputs(state)
        forward = [field.compute_agent_step(state, item) for item in inputs]
        reverse = [field.compute_agent_step(state, item) for item in reversed(inputs)]
        self.assertEqual(forward, list(reversed(reverse)))
        self.assertEqual([a.energy for a in field.agents], [10, 20])
        self.assertEqual([a.age for a in field.agents], [0, 0])
        self.assertEqual(field.grid_food[0, 0], 5)
        field.agents[0].x = 8
        field.agents[0].energy = 100
        field.agents[0].weights[:] = 1
        field.grid_food[:] = 0
        self.assertEqual(forward, [field.compute_agent_step(state, item) for item in inputs])
        for array in (state.positions, state.energies, state.weights, state.food):
            with self.assertRaises(ValueError):
                array.flat[0] = 0

    def test_observation_uses_starting_energy(self):
        field = self.make_field(death_threshold=-10, life_left=1, energy_loss_per_step=1)
        field.agents = [self.make_agent(0, 0, -8.5), self.make_agent(1, 0, -9.5)]
        with patch.object(AgentV4, 'action_from_weights', return_value=(0, 0)) as decide:
            field.step()
        self.assertEqual(decide.call_args_list[0].args[1:], (0, 1, 0, 1))
        self.assertEqual(decide.call_args_list[1].args[1:], (0, 1, 1, 0))

    def test_shared_food_rewards_every_agent_without_consumption(self):
        field = self.make_field(energy_loss_per_step=0, reward_food=2)
        field.agents = [self.make_agent(0, 0), self.make_agent(1, 0)]
        field.grid_food[0, 0] = 5
        field.step()
        self.assertEqual([a.energy for a in field.agents], [2, 2])
        self.assertEqual([(a.x, a.y) for a in field.agents], [(0, 0), (0, 0)])
        self.assertEqual(field.grid_food[0, 0], 4)

    def test_food_absence_and_new_spawn_do_not_reward(self):
        field = self.make_field(field_size=1, energy_loss_per_step=0.25)
        agent = self.make_agent(energy=10)
        field.agents = [agent]
        field.step_food_prob = 1
        field.step()
        self.assertEqual(agent.energy, 9.75)
        self.assertEqual(field.grid_food[0, 0], 5)
        field.step_food_prob = 0
        field.step()
        self.assertEqual(agent.energy, 10.5)

    def test_food_is_available_for_exactly_five_steps(self):
        field = self.make_field(field_size=1, energy_loss_per_step=0)
        agent = self.make_agent()
        field.agents = [agent]
        field.spawn_objects(prob=1)
        for remaining in (4, 3, 2, 1, 0):
            field.step()
            self.assertEqual(field.grid_food[0, 0], remaining)
        self.assertEqual(agent.energy, 5)
        field.step()
        self.assertEqual(agent.energy, 5)

    def test_respawn_refreshes_lifetime(self):
        field = self.make_field(field_size=1)
        field.grid_food[0, 0] = 1
        field.step_food_prob = 1
        field.step()
        self.assertEqual(field.grid_food[0, 0], 5)
        field.grid_food[0, 0] = 3
        field.step()
        self.assertEqual(field.grid_food[0, 0], 5)

    def test_lifetime_requires_positive_integer_and_old_argument_is_removed(self):
        for value in (0, -1, 1.5, 5.0, True, np.bool_(True), '5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_field(food_lifetime_steps=value)
        self.assertEqual(self.make_field(food_lifetime_steps=2).food_lifetime_steps, 2)
        with self.assertRaises(TypeError):
            self.make_field(food_durability=5)

    def test_metabolic_death_can_be_rescued_in_either_list_order(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                field = self.make_field(energy_loss_per_step=1, transfer_amount=3)
                donor = self.make_agent(0, 0, 0, (0, 1))
                receiver = self.make_agent(1, 0, -50)
                field.agents = [receiver, donor] if reverse else [donor, receiver]
                field.step()
                self.assertEqual(len(field.agents), 2)
                self.assertEqual(donor.energy, -4)
                self.assertEqual(receiver.energy, -48)
                self.assertEqual((donor.x, donor.y), (1, 0))
                self.assertEqual(receiver.age, 1)
                self.assertEqual(field.acting_agent_count, 2)
                self.assertEqual(field.altruism_count, 1)

    def test_donor_at_threshold_can_be_rescued_after_giving(self):
        field = self.make_field(energy_loss_per_step=1, transfer_amount=3)
        first = self.make_agent(0, 0, -47, (0, 1))
        second = self.make_agent(1, 0, 0, (0, 1))
        field.agents = [first, second]
        field.step()
        self.assertEqual(len(field.agents), 2)
        self.assertEqual(first.energy, -47)  # Gives 2 down to -50, then receives 3.
        self.assertEqual(second.energy, -2)
        self.assertEqual(first.energy + second.energy, -49)
        self.assertEqual(field.positive_altruism_count, 2)

    def test_received_energy_is_not_retransmitted_and_no_negative_transfer(self):
        field = self.make_field(energy_loss_per_step=1, transfer_amount=3)
        exhausted = self.make_agent(0, 0, -50, (0, 1))
        donor = self.make_agent(1, 0, 0, (0, 1))
        field.agents = [exhausted, donor]
        field.step()
        self.assertEqual(exhausted.energy, -48)
        self.assertEqual(donor.energy, -4)
        self.assertEqual(field.altruism_count, 1)

    def test_food_can_rescue_agent_and_death_at_exact_threshold_is_deferred(self):
        field = self.make_field(field_size=1, energy_loss_per_step=1, reward_food=2)
        agent = self.make_agent(energy=-50)
        field.agents = [agent]
        field.grid_food[0, 0] = 5
        field.step()
        self.assertEqual(field.agents, [agent])
        self.assertEqual(agent.energy, -49)
        field.grid_food[:] = 0
        field.step()
        self.assertEqual(agent.age, 2)
        self.assertEqual(agent.energy, -50)
        self.assertEqual(field.agents, [])
        self.assertEqual(field.acting_agent_count, 1)
        field.step()
        self.assertEqual(field.acting_agent_count, 0)
        self.assertEqual(field.altruism_count, 0)

    def test_approach_uses_snapshot_positions(self):
        field = self.make_field()
        field.agents = [self.make_agent(0, 0, 10, (0, 1)), self.make_agent(1, 0, 10, (0, 1))]
        field.step()
        self.assertEqual([(a.x, a.y) for a in field.agents], [(1, 0), (0, 0)])

    def test_food_avoidance_wraps_around_torus(self):
        field = self.make_field(energy_loss_per_step=0)
        agent = self.make_agent()
        agent.weights[0, 0] = 1
        field.agents = [agent]
        field.grid_food[1, 0] = 5
        field.step()
        self.assertEqual((agent.x, agent.y), (9, 0))
        self.assertEqual(agent.energy, 0)

    def test_reproduction_and_counter_reset(self):
        field = self.make_field(field_size=1, energy_loss_per_step=0, mutation_rate=0)
        parent = self.make_agent(energy=100)
        field.agents = [parent]
        field.step()
        self.assertEqual(len(field.agents), 2)
        self.assertIs(field.agents[0], parent)
        self.assertEqual([a.energy for a in field.agents], [0, 0])
        self.assertEqual([a.age for a in field.agents], [1, 0])
        np.testing.assert_array_equal(field.agents[1].weights, parent.weights)
        field.agents[0].weights[:, 1] = (0, 1)
        field.step()
        self.assertEqual(field.altruism_count, 1)
        field.agents[0].weights[:] = 0
        field.step()
        self.assertEqual(field.altruism_count, 0)
        self.assertEqual(field.positive_altruism_count, 0)

    def test_seeded_multistep_energy_balance_and_reproducibility(self):
        def run():
            random.seed(73)
            np.random.seed(73)
            field = FieldV4(
                field_size=4, initial_agents=12, initial_food_prob=0.3,
                step_food_prob=0.2, food_lifetime_steps=3,
                energy_loss_per_step=0.2, reward_food=0.7, transfer_amount=2,
                death_threshold=-1000000, clone_threshold=1000000,
            )
            for _ in range(100):
                before_food = field.grid_food.copy()
                before_energy = sum(a.energy for a in field.agents)
                field.step()
                fed = sum(before_food[a.x, a.y] > 0 for a in field.agents)
                self.assertAlmostEqual(
                    sum(a.energy for a in field.agents), before_energy - 12 * 0.2 + fed * 0.7,
                )
                self.assertEqual(len(field.agents), 12)
                self.assertTrue(np.all((field.grid_food >= 0) & (field.grid_food <= 3)))
            return (
                [(a.x, a.y, a.energy, a.age) for a in field.agents],
                field.grid_food.tolist(), field.altruism_count,
            )

        self.assertEqual(run(), run())


if __name__ == '__main__':
    unittest.main()
