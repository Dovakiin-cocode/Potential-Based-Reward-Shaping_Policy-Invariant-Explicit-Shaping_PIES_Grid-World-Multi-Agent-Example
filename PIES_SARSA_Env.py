from PBSR_PIES_SARSA.PIES_SARSA_Agent import Agent
import datetime

class Env(object):
    def __init__(self, PBSR=False):
        self.PBSR = PBSR
        self.flag_reach = False
        self.alpha, self.gamma, self.epsilon, self.beta = 0.05, 0.8, 0.8, 0.5
        self.num_actions, self.action_label = 4, ['west', 'east', 'north', 'south']
        self.max_timesteps, self.num_of_episodes = 3000, 400
        self.x_dimension, self.y_dimension = 20, 20
        self.num_states = self.x_dimension * self.y_dimension
        self.goal_reward = 1.0
        self.start_states, self.current_agent_state, self.goal_state = 19, 19, 380
        self.move_to_goal = 0
        self.moves_to_goal = []
        self.west_border, self.east_border, self.north_border, self.south_border = self.calculate_border()
        self.agent = Agent(self.num_states, self.num_actions, self.alpha, self.gamma, self.epsilon, self.beta)

    def calculate_border(self):
        x, y = self.x_dimension, self.y_dimension
        west_border = [i for i in range(0, y)]
        east_border = [i for i in range((x - 1) * y, y * x)]
        north_border = [i for i in range(y - 1, y * x, y)]
        south_border = [i for i in range(0, y * x, y)]
        return west_border, east_border, north_border, south_border

    def initialize(self):
        self.move_to_goal = 0
        self.flag_reach = False
        self.current_agent_state = self.start_states
        self.agent.delta = 1

    def do_experiment(self, title):
        print(title, " experiment begins")
        starttime = datetime.datetime.now()
        # long running
        # do something other
        self.agent = Agent(self.num_states, self.num_actions, self.alpha, self.gamma, self.epsilon, self.beta)
        for i in range(0, self.num_of_episodes):
            # print("round ", i + 1, " starts...")
            self.do_episode(title)
            i += 1
        endtime = datetime.datetime.now()
        print("This round of experiment takes ", (endtime - starttime).seconds, "seconds")

    def do_episode(self,title):
        self.initialize()
        for i in range(0, self.max_timesteps):
            if self.flag_reach:
                # print("reached the goal")
                break
            else:
                self.do_timestep(title)
        # print("Q table ", self.agent.q_table)
        # print("Phi table ", self.agent.phi_table)
        self.moves_to_goal.append(self.move_to_goal)

    def do_timestep(self,title):
        self.move_to_goal += 1
        current_state = self.current_agent_state
        if title == "Q_Learning" or title == "SARSA":
            selected_current_action = self.agent.select_action(current_state)
            next_state = self.get_next_state(current_state, selected_current_action)
            selected_next_action = self.agent.select_action(next_state)
            reward = self.calculate_reward(current_state)
            if title == "SARSA":
                self.agent.update_q_value(current_state, selected_current_action, selected_next_action, next_state, reward)
            elif title == "Q_Learning":
                self.agent.update_q_value_Q(current_state, selected_current_action, selected_next_action, next_state, reward)
        # -----------PIES---------------
        elif title=="SARSA_PIES":
            selected_current_action_PIES = self.agent.select_action_PIES(current_state)
            next_state = self.get_next_state(current_state, selected_current_action_PIES)
            selected_next_action_PIES = self.agent.select_action_PIES(current_state)
            reward_Phi, reward_Q = self.calculate_reward_PIES(current_state, next_state)
            self.agent.update_phi_value(current_state, selected_current_action_PIES, selected_next_action_PIES,
                                        next_state,
                                        reward_Phi)
            self.agent.update_q_value(current_state, selected_current_action_PIES, selected_next_action_PIES, next_state,
                                      reward_Q)

        # print("current at ", current_state, " take the ", selected_current_action_PIES," move to ", next_state," next action is", selected_next_action_PIES )
        self.current_agent_state = next_state

    def get_next_state(self, current_state, action):
        next_state = -1
        if action == 0:  # west
            if current_state in self.west_border:
                next_state = current_state
            else:
                next_state = current_state - self.y_dimension
        if action == 1:  # east
            if current_state in self.east_border:
                next_state = current_state
            else:
                next_state = current_state + self.y_dimension
        if action == 2:  # north
            if current_state in self.north_border:
                next_state = current_state
            else:
                next_state = current_state + 1
        if action == 3:  # south
            if current_state in self.south_border:
                next_state = current_state
            else:
                next_state = current_state - 1
        return next_state

    def calculate_reward(self, current_state_num):
        reward = 0.0
        if current_state_num == self.goal_state:
            self.flag_reach = True
            reward = self.goal_reward
        return reward

    def calculate_reward_PIES(self, current_state, next_state):
        reward_Phi = 0.0
        reward_Q = 0.0
        if next_state - current_state == self.y_dimension or current_state - next_state == 1:
            reward_Phi = 1.0
        if next_state == self.goal_state:
            self.flag_reach = True
            reward_Q = self.goal_reward

        return reward_Phi, reward_Q
