import numpy as np
from HexGrid import Cell, HexGrid


class PSCell(Cell):
    def __init__(self, nametag):
        Cell.__init__(self, nametag)

    # returns list of legal moves based on indecies of the direction [n, e, se, s, w, nw] = [0,1,2,3,4,5]
    def get_legal_moves(self):
        legal_moves = []

        for direction in range(len(self.neighbors)):
            if (self.neighbors[direction] is not None) and \
                    (self.neighbors[direction].is_populated) and \
                    (self.neighbors[direction].neighbors[direction] is not None) and \
                    (not self.neighbors[direction].neighbors[direction].is_populated):
                legal_moves.append(direction)
        return legal_moves

    # Cells are equal if their underlying nametags are equal
    def __eq__(self, other):
        return self.nametag == other.nametag and self.is_populated == other.is_populated

    # Making the Cells hashable
    def __hash__(self):
        return hash(str(self.nametag))


class PSBoard(HexGrid):
    remaining_pegs = None

    def __init__(self, board_size, board_shape):
        HexGrid.__init__(self, board_size, board_shape)
        self.board_size = board_size
        self.board_shape = board_shape
        populated_board = self.populate_board(PSCell)
        self.board = self.assign_neighbors(populated_board)
        self.update_remaining_pegs()

    def get_boardsize(self):
        return self.board_size

    def update_remaining_pegs(self):
        counter = 0
        for row in self.board:
            for cell in row:
                if (cell is not None) and cell.is_populated:
                    counter += 1
        self.remaining_pegs = counter
        return

    def get_remaining_pegs(self):
        return self.remaining_pegs

    # Returns tuples on the form [cell, direction]
    def get_all_legal_moves(self):
        legal_moves = []
        if self.get_remaining_pegs() > 1:
            for row in self.board:
                for cell in row:
                    if (cell is not None) and cell.is_populated:
                        for move in cell.get_legal_moves():
                            legal_moves.append(tuple([cell, move]))
        return legal_moves

    def get_cell(self, nametag):
        for i in range(self.board_size):
            for j in range(len(self.board[i])):
                # continue to next loop iteration if cell is None
                if self.board[i][j] is None:
                    continue
                else:
                    cell = self.board[i][j]
                    if cell.get_nametag() == nametag:
                        return cell
        return -1
    
    # Boards are equal if their underlying matrices are equal
    def __eq__(self, other):
        return self.__str__() == other.__str__()

    # Making the boards hashable
    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        output = ""
        for row in self.board:
            for cell in row:
                if cell is not None:
                    if cell.is_populated:
                        output += '1'
                    else:
                        output += '0'
        return output
    
    def one_hot_encode(self):
        return np.array([int(cell) for cell in list(self.__str__())])