import unittest
from unittest.mock import Mock, patch

from src.agent import Agent
from src.field import Field


class FieldStepTests(unittest.TestCase):
    def make_field(self):
        return Field(initial_agents=0, initial_food_prob=0, step_food_prob=0)

    def make_agent(self, x, y, energy, move=(1, 1), give=False, amount=0):
        agent = Agent(x, y)
        agent.energy = energy
        agent.decide_action = Mock(return_value=(move, give, amount))
        return agent

    def test_energy_observation_precedes_metabolism_and_transfers(self):
        field = self.make_field()
        donor = self.make_agent(0, 0, 10, give=True, amount=3)
        receiver = self.make_agent(1, 0, 20)
        field.agents = [donor, receiver]

        field.step()

        donor_args = donor.decide_action.call_args.args
        receiver_args = receiver.decide_action.call_args.args
        self.assertEqual((donor_args[3], donor_args[5]), (10, 20))
        self.assertEqual((receiver_args[3], receiver_args[5]), (20, 10))
        self.assertAlmostEqual(donor.energy, 6.9)
        self.assertAlmostEqual(receiver.energy, 22.9)

    def test_avoidance_uses_target_position_before_movement(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                field = self.make_field()
                first = self.make_agent(0, 0, 0, move=(1, 0))
                second = self.make_agent(1, 0, 0, move=(1, 0))
                field.agents = [second, first] if reverse else [first, second]

                field.step()

                self.assertEqual((first.x, first.y), (9, 0))
                self.assertEqual((second.x, second.y), (2, 0))

    def test_metabolic_deaths_cannot_act_or_be_rescued(self):
        for energy in (-49.95, -49.9):
            for reverse in (False, True):
                with self.subTest(energy=energy, reverse=reverse):
                    field = self.make_field()
                    dying = self.make_agent(0, 0, energy, give=True, amount=10)
                    donor = self.make_agent(1, 0, 0, give=True, amount=10)
                    field.agents = [donor, dying] if reverse else [dying, donor]
                    field.grid_food[0, 0] = 5

                    field.step()

                    self.assertEqual(field.agents, [donor])
                    dying.decide_action.assert_not_called()
                    self.assertEqual((dying.x, dying.y), (0, 0))
                    self.assertEqual(field.grid_food[0, 0], 5)
                    self.assertEqual(field.altruism_count, 0)
                    self.assertAlmostEqual(donor.energy, -0.1)
                    self.assertEqual(donor.decide_action.call_args.args[4:6], (0, 0))

    def test_all_agents_can_die_and_empty_field_can_step(self):
        field = self.make_field()
        agent = self.make_agent(0, 0, -50)
        field.agents = [agent]

        field.step()
        field.step()

        self.assertEqual(field.agents, [])
        agent.decide_action.assert_not_called()

    def test_transfer_death_prevents_movement_eating_and_rescue(self):
        field = self.make_field()
        dying = self.make_agent(0, 0, -49, move=(0, 1), give=True, amount=10)
        receiver = self.make_agent(1, 0, 0, give=True, amount=10)
        field.agents = [dying, receiver]
        field.grid_food[0, 0] = 5
        field.grid_food[9, 0] = 5

        field.step()

        self.assertEqual(field.agents, [receiver])
        self.assertEqual(dying.energy, -50)
        self.assertEqual((dying.x, dying.y), (0, 0))
        self.assertEqual(field.grid_food[0, 0], 5)
        self.assertEqual(field.grid_food[9, 0], 5)
        self.assertAlmostEqual(receiver.energy, 0.8)
        self.assertEqual(receiver.age, 1)
        self.assertEqual(field.altruism_count, 1)

    def test_dead_target_is_not_used_for_avoidance(self):
        field = self.make_field()
        dying = self.make_agent(0, 0, -49, give=True, amount=10)
        survivor = self.make_agent(1, 0, 0, move=(1, 0))
        field.agents = [dying, survivor]
        # With no living target, avoidance falls back to a random move.
        with patch('src.field.random.choice', side_effect=[survivor, dying, 0, 0]):
            field.step()

        self.assertEqual(field.agents, [survivor])
        self.assertEqual((survivor.x, survivor.y), (1, 0))

    def test_zero_transfer_is_not_counted(self):
        field = self.make_field()
        field.agents = [
            self.make_agent(0, 0, 0, give=True, amount=0),
            self.make_agent(1, 0, 0),
        ]

        field.step()

        self.assertEqual(len(field.agents), 2)
        self.assertEqual(field.altruism_count, 0)

    def test_survivor_can_still_reproduce(self):
        field = self.make_field()
        parent = self.make_agent(0, 0, 101)
        field.agents = [parent]

        field.step()

        self.assertEqual(len(field.agents), 2)
        self.assertIs(field.agents[0], parent)
        self.assertEqual([agent.energy for agent in field.agents], [0, 0])


if __name__ == '__main__':
    unittest.main()
