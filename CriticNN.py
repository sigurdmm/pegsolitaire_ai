from tensorflow import keras
import numpy as np
import tensorflow as tf
import tensorflow.keras.models as KMOD
import tensorflow.keras.layers as KLAY
import tensorflow as tf
import tensorflow.keras.callbacks as KCALL

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



class CriticNN:
    """
    Critic of type Neural Network
    """

    def __init__(self, decay, discount):
        """
        Input:
        ------
        decay (float): The eligibility trace-decay
        discount (float): discount factor

        Variables:
        ----------
        model: tensorflow.keras.models.Sequential() neural network
        value_of_states: Dictionary containing values for each state, format: {state: value}}
        eligibilities: List containing the eligibility value for each state, format: [value]
        visited_states: List of all states that have been visited in the current episode, format: [state]
        td_errors[]: List of all TD-errors calculated during the current episode
        """
        self.decay = decay
        self.discount = discount
        
        self.model = Sequential()
        self.eligibilities = []
        self.visited_states = []
        self.td_errors = []
        

    def initialize_NN(self, state, nn_shape, activation_func='relu', optimizer='Adam'):
        """
        Initializing the Neural Network Model

        Input:
            state: the board of the game, used to define the size of the input layer
            nn_shape: List containing the integer size of the hidden layers in the NN
            activation_function: string identifier of built-in Keras activation function
            optimizer: string identifier of built-in Keras 
        """       
        
        ohe_state = state.one_hot_encode()

        #Bulding Neural Network
        #nn_shape on form [xi, ...., xi] where x is the dimension of the i'th layer
        self.model.add(Dense(nn_shape[0], input_dim=(len(ohe_state)), activation=activation_func))
        for i in range (1, len(nn_shape)):
            self.model.add(Dense(nn_shape[i], activation=activation_func))
        self.model.compile(optimizer=optimizer, loss='mse')

        #Matching the size of the eligibilities-list with the trainable weigths
        #so that we can map each weight to its eligibility  
        self.eligibilities = [0 for _ in self.model.trainable_weights]


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
        td_error = reward + self.discount * self.model(self.encode_state(state_1))[0,0].numpy()- self.model(self.encode_state(state_0))[0,0].numpy()
        # td_error = reward + self.discount * self.get_value_of_state(state_1) - self.get_value_of_state(state_0)

        #Adding the TD-error to the list of TD-errors
        self.td_errors.append(td_error)

        #Adding the tensor-object of the state to the list of visited states
        self.visited_states.append(self.encode_state(state_0))

        return td_error
    
    #Converting state to tensor
    def encode_state(self, state):
        tensor = tf.convert_to_tensor(state.one_hot_encode())
        tensor = tf.expand_dims(tensor, 0)
        return tensor


    def end_of_episode(self):
        """
        Updates the Neural Network based on the visited states and their td_errors in the entire episode
        Reseting eligibilities, td_errors and visited_states at the end
        """

        #creating tuples containing each visited state and the corresponding td_error
        state_error = zip(self.visited_states, self.td_errors)

        for element in state_error:
            state = element[0]
            td_error = element[1]
            

            with tf.GradientTape() as g:
                g.watch(self.model.trainable_weights)
                score = self.model(state)
            
            #gradients = parital of V(s) with respect to w_i
            #          = d(V(s))_d(w_i)
            gradients = g.gradient(score, self.model.trainable_weights)

            #e_i <- e_i + d(V(s))_d(w_i)         
            self.update_eligibilities(gradients)

            #w_i <- w_i + alpha*delta*e_i
            self.optimize_model(td_error)

            #e_i <- lambda*e_i
            self.decay_eligibilities()

        #Reseting eligibilities, visited_states and td_errors at the end of each episode
        self.eligibilities = []
        self.visited_states = []
        self.td_errors = []
        

    #e_i <- e_i + d(V(s))_d(w_i)
    def update_eligibilities(self, gradients):
        for i in range (len(self.eligibilities)):
            self.eligibilities[i] += gradients[i]
    
    #e_i <- lambda*e_i
    def decay_eligibilities(self):
        for i in range (len(self.eligibilities)):
            self.eligibilities[i] *= self.decay

    #optimizing model by updating   w_i <- w_i + alpha*delta*e_i
    #alpha is defined by the pre-determined optimizer and incorporated in 'model.optimizer'
    def optimize_model(self, td_error):
        e_times_error = [i * td_error for i in self.eligibilities]
        self.model.optimizer.apply_gradients(zip(e_times_error, self.model.trainable_weights))
