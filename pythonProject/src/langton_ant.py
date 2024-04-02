import copy
import numpy as np
import time
from typing import TypeAlias
from colorama import Fore, Back, Style
import pickle
import psycopg2
from psycopg2.extensions import AsIs

# some aliases to make the code better readable
GridID: TypeAlias = tuple[int, int]
DataPackage: TypeAlias = tuple[GridID, 'GridData']
GridPackage: TypeAlias = tuple[GridID, 'Grid']
AntDirections: TypeAlias = tuple[list['Ant'], list['Ant'], list['Ant'], list['Ant']]
MinMax: TypeAlias = tuple[int, int]

# easy access to database info
simulations_table = "sim_table"
iterations_table = "iter_table"
database = "langtonantdb"
user = "langtonantdb"
host = 'postgresdb'
password = "walkingant"
port = 5432


# class to store all data of which we only need 1 instance
class GlobalData:
    def __init__(self, color_range: int, simulation_id: int):
        self.color_range = color_range
        self.simulation_id = simulation_id


# class to represent the ant
class Ant:
    # arrows to show what direction the ant is facing when drawn in the terminal
    # different options possible, uncomment the one you think looks best for you
    ant_visual = ("\u2B06", "\u27A1", "\u2B07", "\u2B05")

    # ant_visual = ("\u2B89", "\u2B8A", "\u2B8B", "\u2B88")
    # ant_visual = ("\u21D1", "\u21D2", "\u21D3", "\u21D0")
    # ant_visual = ("\u2B9D", "\u2B9E", "\u2B9F", "\u2B9C")

    def __init__(self, loc=None, rot=None, orientation=0, color=Fore.BLACK):
        if loc is None:
            loc = (0, 0)
        if rot is None:
            rot = (1, -1)
        self.loc = loc
        self.old_loc = loc
        self.orientation = orientation
        self.rot = rot  # (counter) clock wise
        self.color = color
        self.move_f = default_move
        self.interaction_f = default_interaction
        self.crossed_over = False

    # Store the old location and calculate a new one.
    def move(self, grid, glob):
        self.old_loc = self.loc
        self.move_f(self, grid, glob)

    # interact with another ant and return which one won and lost
    def interaction(self, other_ant):
        return self.interaction_f(self, other_ant)


# default function to pick a direction to go to next and update the cell color
def default_move(ant: Ant, grid, glob):
    grid_cell = grid[*ant.loc]
    old_color = grid_cell.color

    grid_cell.color = (old_color + 1) % glob.color_range  # change square color to the next one

    for _ in range(4):
        ant.orientation = (ant.orientation + ant.rot[
            old_color % len(ant.rot)]) % 4  # update orientation with cluster in cycle
        new_loc = tuple(map(sum, zip(ant.loc, LangtonAnt.cardinal[ant.orientation])))  # create new location
        if grid[*new_loc].ant is None:
            ant.loc = new_loc
            return

    # ant is surrounded and can't move


# default function to handle interactions with other ants. Returns lived and dead ant
def default_interaction(ant, other_ant):
    if other_ant.orientation < ant.orientation:  # The ant with the lowest orientation number dies
        return ant, other_ant
    return other_ant, ant


# Class to hold the grid cell
class GridCell:

    def __init__(self):
        self.color = 0
        self.ant: Ant = None

    # methode to make padding work
    @staticmethod
    def grid_cell_padding(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = GridCell()
        vector[-pad_width[1]:] = GridCell()


# A glorified ndarray. The get and set index is shifted by 1 to allow for padding around the borders
# the rest is the same
class GridData:

    def __init__(self, middle, shape: tuple[int, int], ants: list[Ant], border_ants: list[Ant], border_cleanup: bool):
        self.shape = shape
        self.middle = middle
        self.ants: list[Ant] = ants
        self.border_ants: list[Ant] = border_ants
        self.border_cleanup = border_cleanup

    # create a list with a grid
    @classmethod
    def from_grid(cls, grid: 'Grid'):
        ants = grid.ants
        shape = grid.shape
        return cls(grid.grid, shape, ants, [], False)

    # create grid data from a list of border ants
    @classmethod
    def from_border_ants(cls, shape: tuple[int, int], border_ants: list[Ant]):
        return cls(None, shape, [], border_ants, True)

    # merge 2 grid data objects with the same id together
    def merge_slices(self, section: 'GridData'):
        # switch merge to make the middle section the main grid
        if section.middle is not None:
            return section.merge_slices(self)

        # combine the border ants
        self.border_ants += section.border_ants
        self.border_cleanup = True
        return self

    # add ants to
    def __update_ants(self, grid: 'Grid'):
        for ant in self.border_ants:

            # if the ant is a crossover ant put it at its old location
            if ant.crossed_over:
                ant.crossed_over = False
                cardinal = LangtonAnt.cardinal[(ant.orientation + 2) % 4]
                ant.old_loc = tuple(i + s * c for i, s, c in zip(ant.old_loc, grid.shape, cardinal))
                ant.loc = tuple(i + s * c for i, s, c in zip(ant.loc, grid.shape, cardinal))
                grid_cell: GridCell = grid[*ant.old_loc]
                grid_cell.ant = ant
                grid.ants.append(ant)
                continue

            # The ant is on the border of a neighbouring grid. W
            # get the direction from where the ant is located
            size_y, size_x = grid.shape
            direction_id = -1
            #                            down           left                  up           right
            for c, size, direction in [(0, 0, 2), (1, size_x - 1, 3), (0, size_y - 1, 0), (1, 0, 1)]:
                if ant.loc[c] == size:
                    direction_id = direction
                    break

            # a check to rule out coding errors this statement ideally should never become true
            if direction_id == -1:
                print("something is not right")
                exit(ant.loc)

            ant.loc = tuple(i + s * c for i, s, c in zip(ant.loc, grid.shape, LangtonAnt.cardinal[direction_id]))
            grid_cell: GridCell = grid[*ant.loc]

            # if there is no border ant in the assigned square place the ant else do an interaction
            if grid_cell.ant is None:
                grid_cell.ant = ant
                continue

            # assign the life ant
            grid_cell.ant, _ = ant.interaction(grid_cell.ant)

    # returns a Grid Data object containing a full connected grid
    def fix_grid(self):
        size_y, size_x = tuple(map(lambda i: i + 2, self.shape))

        # If true this cell already existed and potentially needs some cleanup.
        if self.middle is not None:

            # If demanded remove all ants from cells on the 1 wide border of this cell
            if self.border_cleanup:
                x_step = 1
                for y in range(0, size_y):
                    for x in range(0, x_step, size_x):
                        grid_cell: GridCell = self.middle[y, x]
                        grid_cell.ant = None
                    x_step = 1 if y == size_y - 1 else size_x - 1

            # if there are no border ants to add we can return here
            if not self.border_ants:
                return Grid.by_grid_data(self)

        # if there is no grid defined in this cell
        if self.middle is None:

            # No border ants we don't need to render this cell return null to remove it later.
            if not self.border_ants:
                return

            # we need to render this cell. Give it a grid.
            self.middle = np.fromiter((GridCell() for _ in range(size_x * size_y)), dtype=GridCell).reshape(size_y,
                                                                                                            size_x)

        # convert to a real grid and add all the new ants
        grid = Grid.by_grid_data(self)
        self.__update_ants(grid)

        return grid


# class to represent the grid on which the ants move.
# The grid contained in this object is a grid of shape with a padding of 1 around the edges.
# The padding represents the bordering cells of the neighbours on that side of this cell.
# This border will be referred to as crossover border.
class Grid:

    def __init__(self, grid, ants: list[Ant], shape):
        self.grid = grid
        self.ants: list[Ant] = ants
        self.shape = shape

    # create the grid based on a list of ants and a shape
    @classmethod
    def by_size(cls, shape: tuple[int, int], ants: list[Ant]):
        size_y, size_x = shape
        grid = np.fromiter((GridCell() for _ in range(size_x * size_y)), dtype=GridCell).reshape(shape)
        for ant in ants:
            grid_cell: GridCell = grid[*ant.loc]
            grid_cell.ant = ant
        grid = np.pad(grid, 1, GridCell.grid_cell_padding)
        return cls(grid, ants, shape)

    # create the grid from a GridData object
    @classmethod
    def by_grid_data(cls, grid_data: GridData):
        grid = grid_data.middle
        ants = grid_data.ants
        shape = grid_data.shape
        return cls(grid, ants, shape)

    # getter and setter to allow easy access to the internal numpy grid.
    # Since we don't want to complicate the computations and ant positions by having to account
    # for the crossover border the index is shifted by +1. This makes the grid itself accessible by
    # its normal indexes and the crossover border by -1 and size.
    def __getitem__(self, item):
        if isinstance(item, tuple) and all(isinstance(i, int) for i in item):
            item = tuple(i + 1 for i in item)
        return self.grid.__getitem__(item)

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and all(isinstance(i, int) for i in key):
            key = tuple(i + 1 for i in key)
        return self.grid.__setitem__(key, value)


# Class to hold the langton's ant simulation
class LangtonAnt:
    cardinal = ((-1, 0), (0, 1), (1, 0), (0, -1))

    @staticmethod
    def __border_ant(grid: Grid, border_ants: AntDirections, border_cleanup: list[bool], ant):
        size_y, size_x = grid.shape

        # direction presets (direction, coordinate, inner border, outer border)
        #               up                   right                       down                 left
        dir_pre = ((0, 0, 0, -1), (1, 1, size_x - 1, size_x), (2, 0, size_y - 1, size_y), (3, 1, 0, -1))

        # the ant can be close to multiple borders therefor we have to check at least 2 but all is easier
        for direction, cord, i_border, o_border in dir_pre:

            # check if ant entered the border region or stayed in the border region if so send ant to neighbour and
            # force border update
            if ant.loc[cord] == i_border:
                border_ants[direction].append(copy.deepcopy(ant))
                border_cleanup[direction] = True
                continue

            # check if an ant left the border region if so force border update on neighbour
            # and check if the ant is a crossing ant
            if ant.old_loc[cord] == i_border:
                border_cleanup[direction] = True

                # if the ant is a crossing ant mark it, send it to the neighbour
                # and return we don't need to check the other directions anymore
                if ant.loc[cord] == o_border:
                    ant.crossed_over = True
                    border_ants[direction].append(copy.deepcopy(ant))
                    return

    @staticmethod
    def __id_package(grid_id: GridID, shape: tuple[int, int], border_ants: list[Ant], direction_id: int):
        return (tuple[int, int](i + c for i, c in zip(grid_id, LangtonAnt.cardinal[direction_id])),
                GridData.from_border_ants(shape, border_ants))

    # The first part of the simulation. This function calculates where each ant will go and
    # informs the neighbours where needed. Note the ants don't actually move yet.
    @staticmethod
    def advance_one_p1(grid_package: GridPackage, glob: GlobalData):
        grid_id, grid = grid_package
        border_ants: AntDirections = ([], [], [], [])
        border_cleanup: list[bool] = [False, False, False, False]
        for ant in grid.ants:
            # move the ant and change the square color
            ant.move(grid, glob)

            # mark ant if it's on the border
            LangtonAnt.__border_ant(grid, border_ants, border_cleanup, ant)

        # create a datapackage for itself and each neighbour that needs an update.
        # a datapackage contains the movement of the ants near the border and for the grid itself its grid.
        output: list[DataPackage] = [(grid_id, GridData.from_grid(grid))]
        for i in range(4):
            if border_cleanup[i]:
                output.append(LangtonAnt.__id_package(grid_id, grid.shape, border_ants[i], i))

        return output

    # The second part of the simulation. This function moves all the ants to their new location,
    # handles crossed ants, calculates interactions and removes dead ants.
    @staticmethod
    def advance_one_p2(data_package: DataPackage, glob: GlobalData):
        grid_id, grid_data = data_package
        grid = grid_data.fix_grid()
        if grid is None:
            return
        death_ants: list[Ant] = []
        for ant in grid.ants:

            # remove ant from old location if it's still registered there
            old_grid_cell: GridCell = grid[*ant.old_loc]
            if old_grid_cell.ant is ant:
                old_grid_cell.ant = None

            grid_cell = grid[*ant.loc]
            other_ant = grid_cell.ant

            # handle the crossover ant
            if ant.crossed_over:
                ant.crossed_over = False
                death_ants.append(ant)
                if other_ant is not None:
                    grid_cell.ant, _ = ant.interaction(other_ant)
                continue

            # There is no ant on the new square no interaction
            if other_ant is None or other_ant.loc != ant.loc:
                grid_cell.ant = ant
                continue

            # There is a moved ant at the new square interact with it
            grid_cell.ant, death_ant = ant.interaction(other_ant)  # place the sole survivor on this grid square
            death_ants.append(death_ant)  # kill the other ant

        if death_ants:  # Remove all death and crossover ants from our list
            grid.ants = list(set(grid.ants).difference(death_ants))

        return grid_id, grid

    # initialize the database
    @staticmethod
    def init_db():

        # open connection
        conn = psycopg2.connect(database=database,
                                user=user,
                                host=host,
                                password=password,
                                port=port)
        cur = conn.cursor()

        # create a table to contain info for each simulation
        cur.execute("""CREATE TABLE IF NOT EXISTS %s (
                                   simulation_id bigserial PRIMARY KEY,
                                   y_min bigint NOT NULL,
                                   y_max bigint NOT NULL,
                                   x_min bigint NOT NULL,
                                   x_max bigint NOT NULL,
                                   iterations bigint NOT NULL)""", (AsIs(simulations_table),))

        # create a table to contain all iterations of all simulations
        cur.execute("""CREATE TABLE IF NOT EXISTS %s (
                            simulation_id bigint NOT NULL,
                            y_id bigint NOT NULL,
                            x_id bigint NOT NULL,
                            iteration bigint NOT NULL,
                            data bytea NOT NULL,
                            PRIMARY KEY(simulation_id, y_id, x_id, iteration),
                            CONSTRAINT fk_simulation
                            FOREIGN KEY(simulation_id) 
                            REFERENCES %s (simulation_id))""", (AsIs(iterations_table), AsIs(simulations_table)))

        # commit and close connection
        conn.commit()
        cur.close()
        conn.close()

    # create a new simulation in the database
    @staticmethod
    def init_simulation():
        sql_inputs = {'s_table': AsIs(simulations_table)}

        # connect
        conn = psycopg2.connect(database=database,
                                user=user,
                                host=host,
                                password=password,
                                port=port)
        cur = conn.cursor()

        # add simulation
        cur.execute("""INSERT INTO %(s_table)s (y_min, y_max, x_min, x_max, iterations) 
                VALUES (0, 0, 0, 0, -1)
                RETURNING simulation_id""", sql_inputs)

        # fetch simulation id for reference
        s_id = cur.fetchone()[0]

        # commit and close connection
        conn.commit()
        cur.close()
        conn.close()

        return s_id

    # store grid with simulation id , coordinates and iteration in the database
    @staticmethod
    def store_data(grid_package: GridPackage, glob: GlobalData, iteration: int) -> tuple[MinMax, MinMax]:
        grid_c = grid_package[0]
        sql_inputs = {'i_table': AsIs(iterations_table),
                      's_id': glob.simulation_id,
                      'y_id': grid_c[0],
                      'x_id': grid_c[1],
                      'iteration': iteration,
                      'grid_data': pickle.dumps(grid_package)}

        # connect
        conn = psycopg2.connect(database=database,
                                user=user,
                                host=host,
                                password=password,
                                port=port)
        cur = conn.cursor()

        # add grid to the database
        cur.execute("""INSERT INTO %(i_table)s (simulation_id, y_id, x_id, iteration, data) 
                    VALUES(%(s_id)s, %(y_id)s, %(x_id)s, %(iteration)s, %(grid_data)s)""", sql_inputs)

        # commit and close
        conn.commit()
        cur.close()
        conn.close()

        # return grid coordinates y(min, max) x(min, max) to update total grid dimension
        return (grid_c[0], grid_c[0]), (grid_c[1], grid_c[1])

    # update the info for the given simulation. Set the new bounding y and x and increase the iteration with 1
    @staticmethod
    def update_simulation_info(dimensions: tuple[MinMax, MinMax], simulation_id: int):
        sql_inputs = {'s_table': AsIs(simulations_table),
                      'min_y': dimensions[0][0],
                      'max_y': dimensions[0][1],
                      'min_x': dimensions[1][0],
                      'max_x': dimensions[1][1],
                      's_id': simulation_id}

        # connect
        conn = psycopg2.connect(database=database,
                                user=user,
                                host=host,
                                password=password,
                                port=port)
        cur = conn.cursor()

        # update the information
        cur.execute("""UPDATE %(s_table)s 
                        SET y_min = %(min_y)s,
                        y_max = %(max_y)s,
                        x_min = %(min_x)s,
                        x_max = %(max_x)s,
                        iterations = iterations + 1
                        WHERE simulation_id = %(s_id)s""", sql_inputs)

        # commit and close
        conn.commit()
        cur.close()
        conn.close()

    # Visualization function to show just this grid in the terminal
    @staticmethod
    def visualize_grid(grid: Grid, iteration, colors):

        # print iteration and ant locations
        print(f"iteration: {iteration}")

        size_y, size_x = grid.shape
        # print the grid with ant = arrow, squares = colors
        for y in range(0, size_y):
            for x in range(0, size_x):
                grid_cell: GridCell = grid[y, x]
                cell_color = colors[grid_cell.color]
                ant = grid_cell.ant
                cell_infill = "   " if ant is None else (ant.color + f" {Ant.ant_visual[ant.orientation]} ")
                print(cell_color + cell_infill, end="")
            print(Style.RESET_ALL + f" {y}")
        for x in range(0, size_x):
            print(f" {x} " if -1 < x < 10 else f"{x} ", end="")
        print("")

    # example run. Not sure if it still works
    @staticmethod
    def example_run():
        main_colors = (Back.WHITE, Back.BLUE, Back.GREEN, Back.MAGENTA)
        loops = 10000
        glob = GlobalData(len(main_colors), 1)
        main_grid = Grid.by_size((40, 40), [Ant((16, 16), [-1, 1, -1, 1]),
                                            Ant((16, 19), [-1, 1, -1, 1]), Ant((15, 18), [-1, 1, -1, 1])])

        loop_package = ((0, 0), main_grid)
        for i in range(loops):
            if not i % 50 or (900 < i < 950 and not i % 10):
                LangtonAnt.visualize_grid(loop_package[1], i, main_colors)
                time.sleep(1)
            new_grid_package = LangtonAnt.advance_one_p1(loop_package, glob)[0]
            new_grid_package[1].border_cleanup = True
            loop_package = LangtonAnt.advance_one_p2(new_grid_package, glob)

        LangtonAnt.visualize_grid(loop_package[1], loops - 1, main_colors)


if __name__ == "__main__":
    LangtonAnt.example_run()
