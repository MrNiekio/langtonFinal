from typing import TypeAlias
from pyspark import RDD, Broadcast
from pyspark.sql import SparkSession
from langton_ant import LangtonAnt, Grid, GridCell, GridData, GlobalData, Ant, GridPackage, GridID
from colorama import Back
import numpy as np

IndexAnts: TypeAlias = tuple[GridID, list[Ant]]
MinMax: TypeAlias = tuple[int, int]


def initialize_data(spark, ants: list[Ant], shape: tuple[int, int], color_range: int):
    if not isinstance(spark, SparkSession):
        return

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

    la_grids: list[GridPackage] = []
    for grid in simple_grids:
        la_grids.append((grid[0], Grid.by_size(shape, grid[1])))

    broadcast = spark.sparkContext.broadcast(GlobalData(color_range))
    grid_data = spark.sparkContext.parallelize(la_grids)
    return broadcast, grid_data


def advance_one(broadcast: Broadcast[GlobalData], grid_data: RDD[GridPackage]) -> RDD[GridPackage]:
    return (grid_data.flatMap(lambda grid_package: LangtonAnt.advance_one_p1(grid_package, broadcast.value))
            .reduceByKey(lambda grid_1, grid_2: GridData.merge_slices(grid_1, grid_2))
            .map(lambda grid_package: LangtonAnt.advance_one_p2(grid_package, broadcast.value))
            .filter(lambda x: x is not None))


def visualize(iteration: int, input_rdd: RDD[GridPackage], shape: tuple[int, int], colors):
    # data = input_rdd.collect()
    # for row in data:
    #     LangtonAnt.visualize_grid(row[1], row[0][1], colors)

    data = input_rdd.sortByKey().collect()
    min_max = tuple[MinMax, MinMax](map(lambda g_id: (min(g_id), max(g_id)),
                                        zip(*(grid_id for grid_id, _ in data))))

    print_grid = Grid.by_size(shape, [])
    (start_y, end_y), (start_x, end_x) = min_max
    scale_y, scale_x = tuple((end - start + 1) * s for (start, end), s in zip(min_max, shape))
    size_y, size_x = shape
    template_grid = print_grid.grid
    data_index = 0
    output_grid = np.array([], dtype=GridCell).reshape(0, scale_x + 2)
    y_s, y_e = 0, size_y + 1
    y_shape = size_y + 1
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

    print_grid.grid = output_grid
    print_grid.shape = (scale_y, scale_x)
    LangtonAnt.visualize_grid(print_grid, iteration, colors)


def run_spark_session():
    colors = (Back.WHITE, Back.BLUE, Back.GREEN, Back.MAGENTA)
    loops = 200
    shape = (5, 5)

    ants = [Ant([1, 10], orientation=3), Ant([5, 7], orientation=3), Ant([-6, 8], orientation=2),
            Ant([0, 20], orientation=0)]
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

    broadcast, grid_data = initialize_data(spark, ants, shape, len(colors))

    for i in range(loops):
        if not i % 50:
            visualize(i, grid_data, shape, colors)
        grid_data = advance_one(broadcast, grid_data)

    visualize(loops, grid_data, shape, colors)

    spark.stop()


if __name__ == "__main__":
    run_spark_session()
