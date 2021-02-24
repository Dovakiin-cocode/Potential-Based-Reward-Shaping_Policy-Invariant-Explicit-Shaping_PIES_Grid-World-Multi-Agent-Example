import numpy as np
import pandas as pd
import random


class Agent(object):
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1, beta=0.1):
        self.num_states, self.num_actions = num_states, num_actions
        self.q_table, self.phi_table = pd.DataFrame(np.zeros((self.num_states, self.num_actions))), pd.DataFrame(
            np.ones((self.num_states, self.num_actions)))
        self.alpha, self.gamma, self.epsilon, self.beta, self.C, self.delta = alpha, gamma, epsilon, beta, 100, 1

    def update_q_value(self, current_state, selected_action, next_action, next_state, reward):
        current_Q_s_a = self.q_table.iloc[current_state].at[selected_action]
        next_Q_s_a = self.q_table.iloc[next_state].at[next_action]
        current_Q_s_a = current_Q_s_a + self.alpha * (reward + self.gamma * next_Q_s_a - current_Q_s_a)
        self.q_table.iloc[current_state].at[selected_action] = current_Q_s_a

    def update_q_value_Q(self, current_state, selected_action, next_action, next_state, reward):
        old_Q = self.q_table.iloc[current_state].at[selected_action]
        max_Q = self.q_table.iloc[next_state].max()
        new_Q = old_Q + self.alpha * (reward + self.gamma * max_Q - old_Q)
        self.q_table.iloc[current_state].at[selected_action] = new_Q

    def update_q_value_PIES(self, current_state, selected_action, next_action, next_state, reward):
        current_Q_s_a = self.q_table.iloc[current_state].at[selected_action]
        next_Q_s_a = self.q_table.iloc[next_state].at[next_action]
        current_Q_s_a = current_Q_s_a + self.alpha * (reward + self.gamma * next_Q_s_a - current_Q_s_a)
        self.q_table.iloc[current_state].at[selected_action] = current_Q_s_a

    def update_phi_value(self, current_state, selected_action, next_action, next_state, phi_reward):
        current_phi = self.phi_table.iloc[current_state].at[selected_action]
        next_phi = self.phi_table.iloc[next_state].at[next_action]
        current_phi = current_phi + self.beta * (phi_reward + self.gamma * next_phi - current_phi)
        self.phi_table.iloc[current_state].at[selected_action] = current_phi

    def select_action(self, state):
        if np.random.uniform() > self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            s = self.q_table.iloc[state]
            action = s[s == s.max()].index
            action = np.random.choice(action)
        return action

    def select_action_PIES(self, state):
        if self.delta - 1 / self.C > 0:
            self.delta = self.delta - 1 / self.C
        else:
            self.delta = 0
        if np.random.uniform() > self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            combined_df = self.q_table + self.phi_table * self.delta
            s = combined_df.iloc[state]
            action = s[s == s.max()].index
            action = np.random.choice(action)
        return action

    def print_Q(self):
        print(self.q_table)
