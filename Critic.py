class Critic:
    """
    Critic of type Table-Critic
    """

    def __init__(self, decay, discount, learning_rate):
        """
        Input:
        ------
        decay (float): The eligibility trace-decay
        discount (float): discount factor
        learning_rate (float): learning rate

        Variables:
        ----------
        value_of_states: Dictionary containing values for each state, format: {state: value}}
        eligibility: Dictionary containing the eligibility value for each state, format: {state: value}}
        visited_states: List of all states that have been visited in the current episode, format: [state]
        """

        self.decay = decay
        self.discount = discount
        self.learning_rate = learning_rate

        self.value_of_states = {}
        self.eligibility = {}
        self.visited_states = []
    

    def get_td_error(self, state_0, state_1, reward):
        """
        Returns the Temporal Differencing Error.

        Input:
            state_0 (state): The current state of the surroundings, (in Peg Solitaire, this is the PSBoard)
            state_0 (state): The next state of the surroundings, (in Peg Solitaire, this is the PSBoard)
            reward (float): The actual reward of the next state
        Output:
            td_error (float): Temporal Differencing Error
        """
        #delta <- r + gamma*V(s') - V(s)
        td_error = reward + self.discount * self.get_value_of_state(state_1) - self.get_value_of_state(state_0)
        
        #Adding the state to the list of visited states 
        self.visited_states.append(state_0)

        #Updating the state-values and eligibilities in the Critic, based on the TD-Error
        self.update_critic(td_error)
        
        return td_error

    def update_critic(self, td_error):
        """
        Updating the  and eligibilities in the Critic, based on the TD-Error

        Input:
            td_error: Float. The Temporal Differencing Error
            
        No Output
        """
        # e(s) <- 1
        self.eligibility[self.visited_states[-1]] = 1

        for i in range(len(self.visited_states)-1, -1, -1):
    
            state = self.visited_states[i]

            e = self.eligibility[state]
            
            #V(s) <- V(s) + alpha*delta*e(s)
            self.value_of_states[state] = self.get_value_of_state(state) + \
                                        self.learning_rate * td_error * e

            #e(s) <- gamma*lambda*e(s)
            self.eligibility[state] = self.decay * self.discount * e


    def get_value_of_state (self, state):
        """
        Returning the value of the state, V(s)

        Input:
            state: state
            
        Output:
            value: float
        """
        #all end states get value
        if len(state.get_all_legal_moves()) == 0:
            return 0

        # return self.value_of_states.get(state, 0)
        if state in self.value_of_states:
            return self.value_of_states[state]
        else:
            return 0

  
    def end_of_episode(self):
        """
        Reseting eligibilities and visited_states at the end of each episode
        """
        self.eligibility = {}
        self.visited_states = []
    