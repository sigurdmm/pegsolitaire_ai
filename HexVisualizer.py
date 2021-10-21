from PIL import Image, ImageDraw
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Styles:
    WHITE = (255, 255, 255)
    RED = (255,0,0,255)
    BLUE = (0,0,255,255)
    BLACK = (0,0,0,255)
    GREEN = (0,255,0,255)

    PLAYER1 = GREEN
    PLAYER2 = RED
    OCCUPIED = BLACK
    EMPTY = BLACK

    BGCOLOR = WHITE

    LINEWIDTH = 10
    LINECOLOR = BLACK

    CELLRADIUS = 50
    #Must be minimum the 2xcellradius to prevent overlaps
    CELLMARGIN = 2*CELLRADIUS + 100
    #Must be minimum the cellradius to prevent overlaps
    PADDING = CELLRADIUS + 100


class HexVizualizer:
    def __init__(self, board, rotation, output_path):
        self.board = board

        self.board_dimentions = (len(board) - 1) * Styles.CELLMARGIN
        self.image_dimentions = self.board_dimentions + 2 * Styles.PADDING
        self.image_centre = self.image_dimentions / 2

        self.board = self.filter_nones(board)

        # Calculate coordinates for all the board cells.
        self.cell_coordinates = self.get_node_coordinates()

        # Render lines between neighbouring cells
        self.render_graphics(rotation, output_path)

    # loop through 2d array removing any None values from the 2darray
    def filter_nones(self, array):
        num_of_rows = len(array[0])
        clean_array = []
        for i in range(num_of_rows):
            row = []
            for j in range(num_of_rows):
                if array[i][j] is not None:
                    row.append(array[i][j])
            clean_array.append(row)
        return clean_array

    # returns a 2d-array where every grid row is an array of tuples
    # containing the node, x-, and y-coordinate: (the node, x, y)
    def get_node_coordinates(self):
        node_coordinates = []
        for i in range(len(self.board)):
            # y coordinate for current row
            y = i * Styles.CELLMARGIN + Styles.PADDING
            row_width = (len(self.board[i]) - 1) * Styles.CELLMARGIN
            row_start = self.image_centre - (row_width / 2)

            cell_coordinates_row = []
            for j in range(len(self.board[i])):
                # x coordinate for current dot
                x = row_start + (j * Styles.CELLMARGIN)
                cell_coordinates_row.append((self.board[i][j], x, y))
            node_coordinates.append(cell_coordinates_row)

        return node_coordinates

    def render_graphics(self, rotation, output_path):

        image = create_image(self.image_dimentions, self.image_dimentions)
        canvas = ImageDraw.Draw(image)

        render_lines(self.cell_coordinates, canvas)
        render_dots(self.cell_coordinates, canvas)

        image = image.rotate(rotation, Image.NEAREST, expand=1, fillcolor=Styles.WHITE)
        print(output_path)
        image.save(output_path)

        # Display image in Jupyter Notebook by using matplotlib.image
        img = mpimg.imread(output_path)
        imgplot = plt.imshow(img)
        imgplot.axes.get_xaxis().set_visible(False)
        imgplot.axes.get_yaxis().set_visible(False)


def render_lines(cell_coordinates, canvas):
    for i in range(len(cell_coordinates)):
            for j in range(len(cell_coordinates[i])):
                #render horizontal lines
                if j+1 < len(cell_coordinates[i]):
                    cell_connector(canvas, cell_coordinates[i][j], cell_coordinates[i][j+1])
                #render vertical
                if i+1 < len(cell_coordinates):
                    cell_connector(canvas, cell_coordinates[i][j], cell_coordinates[i+1][j])
                    #render diagonal
                    if j+1 < len(cell_coordinates[i+1]):
                        cell_connector(canvas, cell_coordinates[i][j], cell_coordinates[i+1][j+1])

def render_dots(cell_coordinates, canvas):
    for row in cell_coordinates:
        for (cell, x, y) in row:
            if cell.is_populated:
                occupied_cell(canvas, [x, y])
            else:
                empty_cell(canvas, [x, y])

def create_image(width, height):
    return Image.new("RGB", (width, height), Styles.BGCOLOR)

def draw_dot(drawing, x, y, r, color, is_filled, width=1):
    upper_left_point = (x-r, y-r)
    lower_right_point = (x+r, y+r)
    coordinates = [upper_left_point, lower_right_point]

    if is_filled:
        drawing.ellipse(coordinates, fill=color)
    else:
        drawing.ellipse(coordinates, fill=Styles.BGCOLOR, outline=color, width=width)

def empty_cell(drawing, coordinates):
    draw_dot(drawing, coordinates[0], coordinates[1], Styles.CELLRADIUS, Styles.EMPTY, False, 5)

def occupied_cell(drawing, coordinates):
    draw_dot(drawing, coordinates[0], coordinates[1], Styles.CELLRADIUS, Styles.OCCUPIED, True)

def draw_line(drawing, coordinates, color, width):
    drawing.line(coordinates, fill=color, width=width)

def cell_connector(drawing, cell1, cell2):
    (_, x1, y1) = cell1
    (_, x2, y2) = cell2
    coordinates = [(x1, y1),(x2, y2)]
    draw_line(drawing, coordinates, Styles.LINECOLOR, Styles.LINEWIDTH)