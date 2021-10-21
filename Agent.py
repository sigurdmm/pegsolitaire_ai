from Actor import Actor
from Critic import Critic
from CriticNN import CriticNN


class CriticType:
    """
    Defines the type of Critic in the Agent
    TABLE: Table based
    NN: Neural Network based
    """
    TABLE = 1
    NN = 2


class Agent:
    def __init__(self, settings, reward_func):
        """
        settings (Settings): Settings object containing all parameters used in the Agent and its Actor and Critic:
                 critic_type: TABLE or NN
                 initial_epsilon: epsilon value that the Actor starts out with

                 dynamic_epsilon: the epsilon value that is fed to the Actor at the current episode.
                 epsilon_decay_param: At what point the epsilon reach zero. Linear decrease
                 l_rate_actor: Learning rate for the Actor
                 l_rate_critic: Learning rate for the Critic
                 discount_actor: Discount rate for Actor
                 discount_critic: Discount rate for Critic
                 decay_actor: Eligibility decay rate Actor
                 decay_critic: Eligibility decay rate Critic
                 nn_shape: Shape of the Neural Network
                 activation_func: Activation function for the Neural Network
                 optimizer: Optimizer for the Neural Network

        reward_func: Passing from the environment the reward function that determines reward based on the state
        """

        self.critic_type = settings.critic_type
        self.initial_epsilon = settings.epsilon

        self.dynamic_epsilon = self.initial_epsilon
        self.epsilon_decay_param = settings.epsilon_decay_param
        self.l_rate_actor = settings.l_rate_actor
        self.l_rate_critic = settings.l_rate_critic
        self.discount_actor = settings.discount_actor
        self.discount_critic = settings.discount_critic
        self.decay_actor = settings.decay_actor
        self.decay_critic = settings.decay_critic
        self.nn_shape = settings.nn_shape
        self.activation_func = settings.activation_func
        self.optimizer = settings.optimizer

        if self.critic_type is CriticType.TABLE:
            self.critic = Critic(self.decay_critic, self.discount_critic, self.l_rate_critic)
        else:
            self.critic = CriticNN(self.decay_critic, self.discount_critic)

        self.actor = Actor(critic=self.critic, decay=self.decay_actor, discount=self.discount_actor,
                           learning_rate=self.l_rate_actor, epsilon=self.dynamic_epsilon, reward_func=reward_func)

    def initialize_game(self, board):
        if self.critic_type is CriticType.NN:
            self.critic.initialize_NN(board, self.nn_shape, self.activation_func, self.optimizer)

    def end_of_episode(self, episodes):
        self.reset_actor(episodes)
        self.critic.end_of_episode()

    def reset_actor(self, episodes):
        self.dynamic_epsilon -= (self.initial_epsilon / (episodes * self.epsilon_decay_param))
        self.dynamic_epsilon = max(0, self.dynamic_epsilon)

        self.actor.reset(epsilon=self.dynamic_epsilon)

    def get_action(self, state, legal_actions, child_states, is_greedy):
        return self.actor.get_action(state, legal_actions, child_states, is_greedy)
