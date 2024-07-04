import pytest
import numpy as np

from differential_evolution import DifferentialEvolution


# CONSTANTS

def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (
            array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))


BOUNDS = np.array([[-20, 20], [-20, 20]])
FOBJ = rastrigin

"""
Ваша задача добиться 100% покрытия тестами DifferentialEvolution
Различные этапы тестирования логики разделяйте на различные функции
Запуск команды тестирования:
pytest -s test_de.py --cov-report=json --cov
"""

np.random.seed(42)
mutation_coefficient = 0.6
crossover_coefficient = 0.3
population_size = 30


def test_DifferentialEvolution_initialization():

    de_solver = DifferentialEvolution(FOBJ, BOUNDS,
                                      mutation_coefficient=mutation_coefficient,
                                      crossover_coefficient=crossover_coefficient,
                                      population_size=population_size)

    assert isinstance(de_solver, DifferentialEvolution)
    assert callable(de_solver.fobj)
    assert np.array_equal(de_solver.bounds, BOUNDS)
    assert isinstance(de_solver.mutation_coefficient, float)
    assert isinstance(de_solver.crossover_coefficient, float)
    assert isinstance(de_solver.population_size, int)
    assert len(de_solver.fitness) == 0
    assert de_solver.best_idx is None
    assert de_solver.best is None

    assert np.allclose(de_solver.bounds, np.array([[-20, 20], [-20, 20]]))
    assert de_solver.mutation_coefficient == pytest.approx(0.6)
    assert de_solver.crossover_coefficient == pytest.approx(0.3)
    assert de_solver.population_size == 30

    with pytest.raises(TypeError):
        de_solver1 = DifferentialEvolution(FOBJ, 1,
                                          mutation_coefficient=mutation_coefficient,
                                          crossover_coefficient=crossover_coefficient,
                                          population_size=population_size)





def test_init_population():

    de_solver = DifferentialEvolution(FOBJ, BOUNDS,
                                      mutation_coefficient=mutation_coefficient,
                                      crossover_coefficient=crossover_coefficient,
                                      population_size=population_size)

    de_solver._init_population()
    assert all(
        np.all(np.logical_and(individual >= min_bound, individual <= max_bound)) for individual, (min_bound, max_bound)
        in zip(de_solver.population, BOUNDS))

    assert np.allclose(de_solver.min_bound, np.array([-20, -20]))
    assert np.allclose(de_solver.max_bound, np.array([20, 20]))

    assert np.allclose(de_solver.diff, np.array([40., 40.]))

    assert np.allclose(de_solver.population_denorm, np.array([[ -5.01839525,  18.02857226],
     [  9.27975767,   3.94633937],
     [-13.75925438, -13.76021919],
     [-17.67665551,  14.64704583],
     [  4.04460047,   8.32290311],
     [-19.17662023,  18.79639409],
     [ 13.29770563, -11.50643557],
     [-12.72700131, -12.66381961],
     [ -7.83031028,   0.99025727],
     [ -2.72219925,  -8.35083439],
     [  4.47411579, -14.42024557],
     [ -8.31421406,  -5.34552627],
     [ -1.75720063,  11.40703846],
     [-12.01304871,   0.56937754],
     [  3.69658275, -18.14198349],
     [  4.30179408, -13.17903505],
     [-17.39793628,  17.95542149],
     [ 18.62528132,  12.33589392],
     [ -7.81544923, -16.09311544],
     [  7.36932106,  -2.39390025],
     [-15.11847061,  -0.1929236 ],
     [-18.62445916,  16.37281608],
     [ -9.64880074,   6.50089137],
     [ -7.53155696,   0.80272085],
     [  1.86841117, -12.60582178],
     [ 18.78338511,  11.00531293],
     [ 17.57995766,  15.79309402],
     [  3.91599915,  16.8749694 ],
     [-16.46029992, -12.1606855 ],
     [-18.19090844,  -6.98678677]]))

    assert np.allclose(de_solver.fitness, np.array([350.44114206, 114.10937098, 397.43792152, 557.474079, 100.44172204, 733.72431091, 342.17161034, 348.94302733,  67.47848767, 104.80489218, 266.59972432, 127.27506805, 161.09812272, 163.73597483, 359.81221309, 211.07700301, 643.48833144, 531.27262006, 327.73546265, 94.71097345, 237.74040415, 649.00801764, 171.29943357,  83.92048371, 183.49514216, 481.85574135, 584.56636567, 304.39072609, 443.19239418, 386.1306672]))

    assert de_solver.best_idx == 8
    assert np.allclose(de_solver.best, np.array([-7.83031028,  0.99025727]))

    assert len(de_solver.population) == population_size
    assert len(de_solver.fitness) == population_size
    assert isinstance(de_solver.best, np.ndarray)


def test_mutation():

    de_solver = DifferentialEvolution(FOBJ, BOUNDS,
                                      mutation_coefficient=mutation_coefficient,
                                      crossover_coefficient=crossover_coefficient,
                                      population_size=population_size)

    de_solver._init_population()
    population_index = 29
    de_solver.idxs = [idx for idx in range(de_solver.population_size) if idx != population_index]
    de_solver._mutation()


    assert np.allclose(de_solver.mutant, np.clip(de_solver.a + de_solver.mutation_coefficient * (de_solver.b - de_solver.c), 0, 1))
    assert (de_solver.mutant <= 1).all()
    assert (de_solver.mutant >= 0).all()
    assert isinstance(de_solver.mutant, np.ndarray)


def test_crossover():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS,
                                      mutation_coefficient=mutation_coefficient,
                                      crossover_coefficient=crossover_coefficient,
                                      population_size=population_size)

    de_solver._init_population()
    population_index = 29
    de_solver.idxs = [idx for idx in range(de_solver.population_size) if idx != population_index]
    de_solver._mutation()

    with pytest.raises(TypeError):
        de_solver.dimensions = []
        de_solver._crossover()

    with pytest.raises(TypeError):
        de_solver.crossover_coefficient = []
        de_solver._crossover()



def test_recombination():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS,
                                      mutation_coefficient=mutation_coefficient,
                                      crossover_coefficient=crossover_coefficient,
                                      population_size=population_size)

    de_solver._init_population()
    population_index = 29
    de_solver.idxs = [idx for idx in range(de_solver.population_size) if idx != population_index]

    with pytest.raises(AttributeError):
        isinstance(de_solver.trial, np.ndarray)

    de_solver.mutant = de_solver._mutation()
    de_solver.cross_points = de_solver._crossover()

    de_solver.trial, de_solver.trial_denorm = de_solver._recombination(population_index)


    assert len(de_solver.trial) == len(de_solver.bounds)
    assert np.all(np.logical_and(de_solver.trial >= de_solver.bounds[:, 0], de_solver.trial <= de_solver.bounds[:, 1]))
    assert isinstance(de_solver.trial_denorm, np.ndarray)
    # assert np.array_equal(de_solver.trial_denorm, de_solver.denormalize(de_solver.trial))


    with pytest.raises(IndexError):
        de_solver._recombination("kj")



    with pytest.raises(TypeError):
        de_solver.min_bound = "p"
        de_solver._crossover(population_index)



def test_evaluate():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS,
                                      mutation_coefficient=mutation_coefficient,
                                      crossover_coefficient=crossover_coefficient,
                                      population_size=population_size)

    de_solver._init_population()
    population_index = 29
    de_solver.best_idx = 2
    de_solver.idxs = [idx for idx in range(de_solver.population_size) if idx != population_index]

    de_solver.mutant = de_solver._mutation()
    de_solver.cross_points = de_solver._crossover()
    de_solver.trial, de_solver.trial_denorm = de_solver._recombination(population_index)
    result_of_evolution = de_solver.fobj(de_solver.trial_denorm)

    assert de_solver.trial is not None

    de_solver.fitness[population_index] = 10 ** 9 / 1.0
    de_solver.fitness[de_solver.best_idx] = 10 ** 9 / 1.0
    print("here")
    de_solver._evaluate(result_of_evolution, population_index)



def test_iterate():
    de_solver = DifferentialEvolution(FOBJ, BOUNDS,
                                      mutation_coefficient=mutation_coefficient,
                                      crossover_coefficient=crossover_coefficient,
                                      population_size=population_size)

    de_solver._init_population()
    de_solver.iterate()
    assert np.all(
        de_solver.idxs == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                           26, 27, 28])

