import random

class Actor:

    def __init__(self, critic, decay, discount, learning_rate, epsilon, reward_func):
        """
        Input:
        ------
        critic (Critic): The Agent's critic
        decay (float): The eligibility trace-decay
        discount (float): discount factor
        learning_rate (float): learning rate
        epsilon (float): epsilon. The Actor makes a random choice with probability epsilon

        Variables:
        ----------
        state_action_values: Dictionary containing values for each state-action pair
                                format: {([state,action]): value}}
        eligibility: Dictionary containing the eligibility value for each state-action pair
                                format: {([state,action]): value}}
        visited_state_actions: List of all state-action pairs that have been visited in the current episode
                                format: [[state, action]]
        """
        self.critic = critic
        self.decay = decay
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount = discount
        self.get_reward = reward_func

        self.state_action_values = {}
        self.eligibility = {}
        self.visited_state_actions = []

    def get_action(self, state, legal_actions, child_states, is_greedy):
        """
        Returns the best action for the progress of the game.
        This is the greedy move with probability 1-epsilon, and a random move with probability epsilon

        Input:
            state: The state of the surroundings, (in Peg Solitaire, this is the PSBoard)
            legal_actions: List of the legal actions in the game at the current state
                            (in Peg Solitaire, each action is of type tuple(PSCell, int direction)
            child_states: List of states. child_states[i] is the state that occurs when legal_action[i]
                            is performed
            is_greedy: True/False flag to force the greedy choice
        Output:
            action
        """
        if is_greedy:
            self.epsilon = 0

        best_action = None
        best_value = None

        # Iterating through the legal actions to find the [state,action]-pair with the highest value
        for action in legal_actions:
            key = tuple([state, action])
            if key in self.state_action_values:

                # a' <-  Pi(s') the action dictated by the current policy for state s'
                value = self.get_value(dictionary=self.state_action_values, state=state, action=action)
                if best_action is None or value > best_value:
                    best_value = value
                    best_action = action


        # Ensuring exploring and ensuring a selection when action is not yet mapped in state_action_values
        # Overwriting best_action with a random action with probability epsilon
        if best_action is None or self.epsilon > random.uniform(0, 1):
            best_action = random.choice(legal_actions)

        # Defining the next state based on the best action from current state
        next_state = child_states[legal_actions.index(best_action)]

        # Getting the reward of the next state from the game
        reward = self.get_reward(next_state)

        # Temporal Differencing Error
        # The Discounted value of the TD-error calculated by the Critic
        td_error = self.discount * self.critic.get_td_error(state_0=state, state_1=next_state, reward=reward)

        # Adding the state-action to the list of visited [state,action]-pairs
        self.visited_state_actions.append([state, best_action])

        # Updating the state-action values and eligibilities in the Actor, based on the TD-Error
        self.update_actor(td_error)

        return best_action

    def update_actor(self, td_error):
        """
        Updating the SA-values and eligibilities in the Actor based on the TD-Error

        Input:
            td_error: Float. The Discounted value of the TD-error

        No Output
        """
        # Iterating through visited SA-pairs, starting with the most recent

        # e(s,a) <- 1
        for i in range(len(self.visited_state_actions) - 1, -1, -1):

            # state, actions from the i'th state-action pair
            state = self.visited_state_actions[i][0]
            action = self.visited_state_actions[i][1]

            # e(s,a) <- 1
            if i == len(self.visited_state_actions) - 1:
                eligibility_value = 1
            # e(s,a)
            else:
                eligibility_value = self.get_value(self.eligibility, state, action)

            # Pi(s,a) <- Pi(s,a)+ alpha*delta*e(s,a)
            state_action_value = self.get_value(self.state_action_values, state, action) \
                                 + self.learning_rate * td_error * eligibility_value

            self.update_value(dictionary=self.state_action_values, state=state,
                              action=action, value=state_action_value)

            # e(s,a) <- gamma*lambda*e(s,a)
            self.update_value(dictionary=self.eligibility, state=state, action=action,
                              value=self.discount * self.decay * eligibility_value)

    def get_value(self, dictionary, state, action):
        """
        Returning the value of the desired dictionary given key (state, action)

        Input:
            dictionary: Dictionary. Either state_action_values or eligibilities
            state: state
            action: action

        Output:
            value: float
        """
        key = tuple([state, action])
        if key in dictionary:
            return dictionary[key]
        else:
            return 0

    def update_value(self, dictionary, state, action, value):
        """
        Updating the value of key (state, action) in the desired dictionary

        Input:
            dictionary: Dictionary. Either state_action_values or eligibilities
            state: state
            action: action
            value:  float
        No Output
        """
        key = tuple([state, action])
        dictionary[key] = value

    def reset(self, epsilon):
        """
        Reseting eligibilities at the end of each episode
        Input:
            epsilon (float): epsilon for the next episode
        """
        self.eligibility = {}
        self.visited_state_actions = []
        self.epsilon = epsilon
