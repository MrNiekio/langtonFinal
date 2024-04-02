from typing import TypeAlias
from pyspark import RDD, Broadcast
from pyspark.sql import SparkSession
from langton_ant import LangtonAnt, Grid, GridCell, GridData, GlobalData, Ant, GridPackage, GridID, MinMax
from colorama import Back
import numpy as np

IndexAnts: TypeAlias = tuple[GridID, list[Ant]]

# initialization of the grid from a list of ants, shape of each subgrid and the amount of colors
def initialize_data(spark, ants: list[Ant], shape: tuple[int, int], color_range: int):
    # if the spark session was not created return
    if not isinstance(spark, SparkSession):
        return

    # loop over all the ants and map them to placeholder grids. (Create the grids if they don't exist yet)
    # placeholder grids are objects containing subgrid coordinates and a list of ants that are on them.
    simple_grids: list[IndexAnts] = []
    for ant in ants:
        grid_id = GridID(c // s for c, s in zip(ant.loc, shape))
        ant.loc = tuple((c % s + s) % s for c, s in zip(ant.loc, shape))
        for grid in simple_grids:
            if grid[0] == grid_id:
                grid[1].append(ant)
                break
        else:
            simple_grids.append((grid_id, [ant]))

    # convert the placeholder grids to full sub grids with their corresponding ant list
    la_grids: list[GridPackage] = []
    for grid in simple_grids:
        la_grids.append((grid[0], Grid.by_size(shape, grid[1])))

    # create an instance of this simulation in the database and get its id
    s_id = LangtonAnt.init_simulation()

    # create the spark objects from the obtained data
    # global_b holds all information that's the same for all nodes
    global_b = spark.sparkContext.broadcast(GlobalData(color_range, s_id))
    # grid_data is a rdd with each sub grid on a row
    grid_data = spark.sparkContext.parallelize(la_grids)

    # store the starting conditions
    iteration_b = spark.sparkContext.broadcast(-1)
    store_data(global_b, iteration_b, grid_data)

    return global_b, grid_data


# advance the whole grid 1 iteration.
def advance_one(global_b: Broadcast[GlobalData], grid_data: RDD[GridPackage]) -> RDD[GridPackage]:
    return (grid_data.flatMap(lambda grid_package: LangtonAnt.advance_one_p1(grid_package, global_b.value))
            .reduceByKey(lambda grid_1, grid_2: GridData.merge_slices(grid_1, grid_2))
            .map(lambda grid_package: LangtonAnt.advance_one_p2(grid_package, global_b.value))
            .filter(lambda x: x is not None))


# store the new grid in the databases history and update the simulation info.
def store_data(global_b: Broadcast[GlobalData], iteration_b: Broadcast[int], grid_data: RDD[GridPackage]):
    min_max = tuple[MinMax, MinMax](
        grid_data.map(lambda grid_package: LangtonAnt.store_data(grid_package, global_b.value, iteration_b.value))
        .reduce(lambda lmm, rmm: tuple((min(l), max(r)) for (l, r) in (zip(*c) for c in zip(lmm, rmm)))))
    LangtonAnt.update_simulation_info(min_max, global_b.value.simulation_id)


# a function to visualize the full grid useful to debug but costly when ran with large grids
def visualize(iteration: int, input_rdd: RDD[GridPackage], shape: tuple[int, int], colors):
    # get all the data in 1 place and sort it by grid coords
    data = input_rdd.sortByKey().collect()

    # prepare all variables for the visualization loop
    data_index = 0
    size_y, size_x = shape
    y_s, y_e = 0, size_y + 1
    y_shape = size_y + 1
    min_max = tuple[MinMax, MinMax](map(lambda g_id: (min(g_id), max(g_id)), zip(*(grid_id for grid_id, _ in data))))
    (start_y, end_y), (start_x, end_x) = min_max
    scale_y, scale_x = tuple((end - start + 1) * s for (start, end), s in zip(min_max, shape))

    # create the object to hold the grid to be visualized and use its blank grid as a template for the empty grids.
    print_grid = Grid.by_size(shape, [])
    template_grid = print_grid.grid

    # create base grid from which we can build the full grid
    output_grid = np.array([], dtype=GridCell).reshape(0, scale_x + 2)

    # Create the full grid by: starting at min y and x, placing empty grids at current y and x until
    # the y and x matches the first next grid in the data, repeat until max y and x.
    for y in range(start_y, end_y + 1):
        if y == end_y:
            y_e = size_y + 2
            y_shape += 1
        x_s, x_e = 0, size_x + 1
        x_grid = np.array([], dtype=GridCell).reshape(y_shape, 0)
        for x in range(start_x, end_x + 1):
            if x == end_x:
                x_e += 1
            if data_index < len(data) and data[data_index][0] == (y, x):
                x_grid = np.hstack((x_grid, data[data_index][1].grid[y_s:y_e, x_s:x_e]))
                data_index += 1
            else:
                x_grid = np.hstack((x_grid, template_grid[y_s:y_e, x_s:x_e]))
            x_s, x_e = 1, size_x + 1
        output_grid = np.vstack((output_grid, x_grid))
        y_s, y_e = 1, size_y + 1
        y_shape = size_y

    # Add the full grid to the grid object, set the right shape for the grid and send it to the visualize function.
    print_grid.grid = output_grid
    print_grid.shape = (scale_y, scale_x)
    LangtonAnt.visualize_grid(print_grid, iteration, colors)


# example simulation should not be the one run by default.
# Ideally there are functions that can be called with parameters to start different configurations.
def run_spark_session():

    # initialize the database
    LangtonAnt.init_db()

    # set some data for this simulation
    colors = (Back.WHITE, Back.BLUE, Back.GREEN, Back.MAGENTA)
    loops = 200
    visualize_val = 50
    shape = (5, 5)
    # ants = [Ant([1, 10], orientation=3), Ant([5, 7], orientation=3), Ant([-6, 8], orientation=2),
    #         Ant([0, 20], orientation=0)]
    ants = [Ant((16, 16), [-1, 1, -1, 1]), Ant((16, 19), [-1, 1, -1, 1]),
            Ant((15, 18), [-1, 1, -1, 1])]

    # create spark session
    spark = SparkSession.builder.appName("LangtonAnt").getOrCreate()

    # convert the data to data spark understands and store it in the database
    global_b, grid_data = initialize_data(spark, ants, shape, len(colors)
                                          )  # type: Broadcast[GlobalData], RDD[tuple[tuple[int, int], Grid]]

    # loop that runs the simulation i times and visualizes every visualize_val times
    for i in range(loops):
        if not i % visualize_val:
            visualize(i, grid_data, shape, colors)
        iteration_b = spark.sparkContext.broadcast(i)
        grid_data = advance_one(global_b, grid_data)
        store_data(global_b, iteration_b, grid_data)

    visualize(loops, grid_data, shape, colors)

    spark.stop()


if __name__ == "__main__":
    run_spark_session()
