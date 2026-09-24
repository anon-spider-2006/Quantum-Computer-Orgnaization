# project/
# |-- quantum_sim/
# |   |-- __init__.py
# |   |-- gates.py
# |   |-- statevector.py
# |   |-- decomposition.py
# |   |-- measurement.py
# |   |-- utils.py
# |-- tests/
# |   |-- test_gates.py
# |   |-- test_statevector.py
# |   |-- test_measurement.py
# |   |-- test_decomposition.py
# |-- examples/
# |   |-- quantum_rng.py
# |   |-- euler_demo.py
# |-- README.md

## Running the programs and tests

Run all commands from the project root directory, which is the directory that
contains the `quantum_sim/`, `tests/`, and `examples/` directories.

The project requires Python, NumPy, and pytest.  Numpy should already
be installed along with Qiskit.

On CCI Power PC install pytest from conda-forge
```bash
conda install conda-forge::pytest --yes
```

On a laptop use pip.
```bash
python -m pip install pytest
```
 Run the Euler-decomposition demonstration program with:

```bash
python examples/euler_demo.py
```

Run the complete automated test suite with:

```bash
pytest -q
```

To run an individual test module, use a command such as:

```bash
pytest -q tests/test_gates.py
```

or:

```bash
pytest -q tests/test_decomposition.py
```

For more detailed output, including the name of every passing test, run:

```bash
pytest -v
```

## Test coverage

The tests are intended both to verify correct behavior and to document
the required interface and numerical conventions.  You may add more
tests as needed.

