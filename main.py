from Agent import Agent, CriticType
from PegSolitaire import PSBoard
from HexGrid import Shape
import copy
import imageio
import time
import matplotlib.pyplot as plt

# Unpopulates all the cells in a list of cells. 
# Used to unpopulate the cells which should be empty in the start state
def empty_cells(cells):
    for cell in cells:
        cell.is_populated = False
    return

# Move peg on board. Direction is an integer representing the direction: [n, e, se, s, w, nw] = [0,1,2,3,4,5]
# returns board in the state it's in after the peg has been moved
def move_peg(board, peg, direction):
    if (peg, direction) not in board.get_all_legal_moves():
        raise ValueError("Not a legal move")
    else:
        board_copy = copy.deepcopy(board)
        peg_copy = board_copy.get_cell(peg.get_nametag())
        peg_copy.unpopulate()
        peg_copy.neighbors[direction].unpopulate()
        peg_copy.neighbors[direction].neighbors[direction].populate()
        board_copy.update_remaining_pegs()
        return board_copy

# Asks the agent for action given board state, all legal moves and whether the agent should be greedy or not.
# Greedy action is equivalent to running the actori with epsilon=0. Returns the peg cell and direction the agent says it should be moved
def decide_move(board, legal_moves, agent, is_greedy):
    child_boards = []
    # build an array of child states to feed the agent with
    for move in legal_moves:
        board_copy = copy.deepcopy(board)
        child_boards.append(move_peg(board=board_copy, peg=move[0], direction=move[1]))

    move = agent.get_action(state=board, legal_actions=legal_moves, child_states=child_boards, is_greedy=is_greedy)

    cell = move[0]
    direction = move[1]
    return [cell, direction]

# reward/reinforcement function
def get_reward(board):
    if board.get_remaining_pegs() == 1:
        # game win
        return 1000
    elif len(board.get_all_legal_moves()) == 0:
        # game loss
        return -100
    else:
        #intermediate move
        return 1

# run a game given a board and a agent. True/False flags for whether the 
# game should be visualized and whether it should be greedy
def play_game(board, agent, is_greedy, visualize):
    board_history = [board]

    while len(board.get_all_legal_moves()) > 0:
        [peg, direction] = decide_move(board, board.get_all_legal_moves(), agent, is_greedy)
        board = move_peg(board, peg, direction)
        board_history.append(board)

    # visualize the board by generating a gif of the game: out/solution.gif
    if visualize:
        visualize_game(board_history)
    return board


def visualize_game(board_history):
    images = []

    for i in range(len(board_history)):
        filename = f'out/img{i}.png'
        board_history[i].visualize(filename)
        images.append(imageio.imread(filename))
    imageio.mimsave('out/solution.gif', images, duration=Settings.frame_delay)


def train(board, agent):
    agent.initialize_game(board)

    ep = []
    results = []
    start = time.time()
    episodes = Settings.episodes
    for n in range(episodes):
        board_copy = copy.deepcopy(board)
        board_copy = play_game(board_copy, agent, False, False)
        nr_pegs = board_copy.get_remaining_pegs()
        print("Game ", n + 1, " : ", nr_pegs, " in ", time.time() - start, "s")
        ep.append(n + 1)
        results.append(nr_pegs)
        agent.end_of_episode(episodes)

        start = time.time()

    print("Nr of victories: ", results.count(1))

    plt.bar(ep, results)
    plt.xlabel('Episode')
    plt.ylabel('Nr of remaining pegs')
    plt.savefig('out/plot.png')


def get_agent():
    return Agent(Settings(), reward_func=get_reward)

def get_game_board():
    b = PSBoard(board_size=Settings.board_size, board_shape=Settings.board_shape)
    for cell in Settings.empty_cells:
        empty_cells([b.board[cell[0], cell[1]]])
    return b

class Settings:
    """
    Input parameters for the simulation of the game
    """

    #Critic type either .TABLE or .NN
    critic_type=CriticType.NN

    epsilon=0.99
    epsilon_decay_param=0.9

    l_rate_actor=0.1     # 0.01 TABLE, 0.1 NN
    l_rate_critic=0.1    # 0.01 TABLE, 0.1 NN

    discount_actor=0.9
    decay_actor=0.9 #The eligibility trace-decay

    discount_critic=0.9
    decay_critic=0.9 #The eligibility trace-decay

    #Parameters for the Neural Net used by the .NN Critic
    nn_shape=[15, 1]
    activation_func='relu'
    optimizer='SGD'

    episodes=2000
    frame_delay = 0.5

    # Board format:
    # (Shape either .TRIANGLE or .DIAMOND)
    empty_cells=[(2,2)]
    board_shape = Shape.DIAMOND
    board_size = 4

def main():
    agent = get_agent()
    board = get_game_board()
    train(board=board, agent=agent)
    play_game(board, agent, is_greedy=True, visualize=True)


if __name__ == '__main__':
    main()