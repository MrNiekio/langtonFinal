import copy
import numpy as np
import time
from typing import TypeAlias
from colorama import Fore, Back, Style

GridID: TypeAlias = tuple[int, int]
DataPackage: TypeAlias = tuple[GridID, 'GridData']
GridPackage: TypeAlias = tuple[GridID, 'Grid']
AntDirections: TypeAlias = tuple[list['Ant'], list['Ant'], list['Ant'], list['Ant']]


# class to store all data of which we only need 1 instance
class GlobalData:
    def __init__(self, color_range):
        self.color_range = color_range


# class to represent the ant
class Ant:
    # arrows to show what direction the ant is facing when drawn in the terminal
    # different options possible, uncomment the one you think looks best for you
    ant_visual = ("\u2B06", "\u27A1", "\u2B07", "\u2B05")
    # ant_visual = ("\u2B89", "\u2B8A", "\u2B8B", "\u2B88")
    # ant_visual = ("\u21D1", "\u21D2", "\u21D3", "\u21D0")
    # ant_visual = ("\u2B9D", "\u2B9E", "\u2B9F", "\u2B9C")

    # possible directions

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

    def move(self, grid, glob):
        self.old_loc = self.loc
        self.move_f(self, grid, glob)

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

    @classmethod
    def from_grid(cls, grid: 'Grid'):
        ants = grid.ants
        shape = grid.shape
        return cls(grid.grid, shape, ants, [], False)

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

        if self.middle is not None:

            # run border cleanup
            if self.border_cleanup:
                x_step = 1
                for y in range(0, size_y):
                    for x in range(0, x_step, size_x):
                        grid_cell: GridCell = self.middle[y, x]
                        grid_cell.ant = None
                    x_step = 1 if y == size_y - 1 else size_x - 1

            if not self.border_ants:
                return Grid.by_grid_data(self)

        if self.middle is None:
            if not self.border_ants:
                return

            self.middle = np.fromiter((GridCell() for _ in range(size_x * size_y)), dtype=GridCell).reshape(size_y,
                                                                                                            size_x)

        grid = Grid.by_grid_data(self)
        self.__update_ants(grid)

        return grid


class Grid:

    def __init__(self, grid, ants: list[Ant], shape):
        self.grid = grid
        self.ants: list[Ant] = ants
        self.shape = shape

    @classmethod
    def by_size(cls, shape: tuple[int, int], ants: list[Ant]):
        size_y, size_x = shape
        grid = np.fromiter((GridCell() for _ in range(size_x * size_y)), dtype=GridCell).reshape(shape)
        for ant in ants:
            grid_cell: GridCell = grid[*ant.loc]
            grid_cell.ant = ant
        grid = np.pad(grid, 1, GridCell.grid_cell_padding)
        return cls(grid, ants, shape)

    @classmethod
    def by_grid_data(cls, grid_data: GridData):
        grid = grid_data.middle
        ants = grid_data.ants
        shape = grid_data.shape
        return cls(grid, ants, shape)

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

    # Do the simulation
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

        output: list[DataPackage] = [(grid_id, GridData.from_grid(grid))]
        for i in range(4):
            if border_cleanup[i]:
                output.append(LangtonAnt.__id_package(grid_id, grid.shape, border_ants[i], i))

        return output

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

    @staticmethod
    def example_run():
        main_colors = (Back.WHITE, Back.BLUE, Back.GREEN, Back.MAGENTA)
        loops = 10000
        glob = GlobalData(len(main_colors))
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
