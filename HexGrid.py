import numpy as np
from HexVisualizer import HexVizualizer


class Shape:
    DIAMOND = 1
    TRIANGLE = 2


class Cell:
    neighbors = []
    is_populated = True

    def __init__(self, nametag):
        self.nametag = nametag
        self.is_populated = True

    def __str__(self):
        return str(self.nametag)

    def populate(self):
        self.is_populated = True
        return

    def unpopulate(self):
        self.is_populated = False
        return

    def get_neighbors(self):
        return self.neighbors

    def get_nametag(self):
        return self.nametag


class HexGrid:
    def __init__(self, board_size, board_shape):
        self.board_size = board_size
        self.board_shape = board_shape
        populated_board = self.populate_board(Cell)
        self.board = self.assign_neighbors(populated_board)

    def populate_board(self, cell_type):
        board = np.empty((self.board_size, self.board_size), dtype='O')
        nametag_counter = 1

        for i in range(self.board_size):
            if self.board_shape == Shape.DIAMOND:
                for j in range(self.board_size):
                    board[i, j] = cell_type(nametag_counter)
                    nametag_counter += 1
            elif self.board_shape == Shape.TRIANGLE:
                for j in range(i + 1):
                    board[i, j] = cell_type(nametag_counter)
                    nametag_counter += 1
        return board


    def assign_neighbors(self, board):
        for i in range(len(board)):
            for j in range(len(board[i])):

                # continue to next loop iteration if cell is None
                # Only used when Board shape is TRIANGLE, as the cells in the matrix not used by the triangle is None
                if board[i][j] is None:
                    continue

                # north, east, southeast, south, southwest, northwest
                n = e = se = s = w = nw = None


                if i > 0:
                    n = board[i - 1][j]

                if j < len(board[i]) - 1:
                    e = board[i][j + 1]
                    if i < len(board[i]) - 1:
                        se = board[i + 1][j + 1]

                if i < len(board[i]) - 1:
                    s = board[i + 1][j]

                if j > 0:
                    w = board[i][j - 1]
                    if i > 0:
                        nw = board[i - 1][j - 1]

                board[i][j].neighbors = [n, e, se, s, w, nw]
        return board

    def visualize(self, output_path):
        rotation = 0
        #Diamond boards are rendered as a square and rotated 45deg
        if self.board_shape == Shape.DIAMOND:
            rotation = 45
        HexVizualizer(self.board, rotation, output_path)

    # Override print method to print rendered image and return "It works"
    def __str__(self):
        self.visualize('output.png')
        return "It works"


# Returns a console printable version of the board. For development purposes.
def get_board_string(board):
    b = board.copy()
    string = ''
    for i in range(len(b)):
        substr = ''
        for j in range(len(b)):
            if b[i][j] is None:
                substr += " "
            elif not b[i][j].is_populated:
                substr += "{:<4}".format('O')
            else:
                substr += "{:<4}".format(str(b[i][j]))
        string += substr + '\n'
    return string

# Returns a console printable version of a neighbor list. For development purposes.
def get_neighbors_string(neighborlist):
    n = neighborlist.copy()

    for i in range(len(n)):
        if n[i] is None:
            n[i] = " "

    r1 = f'{n[5]}{n[0]} \n'
    r2 = f'{n[4]}*{n[1]}\n'
    r3 = f' {n[3]}{n[2]}'

    return r1 + r2 + r3