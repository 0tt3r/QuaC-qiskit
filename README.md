# Qiskit QuaC Provider and Backends

This Python package serves as an easily-installable interface between Qiskit and QuaC (Quantum in C), a fast, highly parallel noisy quantum system simulator. This interface is useful for two primary reasons.

1.  It allows potential users to interact with QuaC via an already-familar interface (i.e., Qiskit)

2.  It allows QuaC to leverage the many excellent features (like circuit optimization) present in Qiskit

The code in this repository is based on qiskit provider/backend code structure for maximum compatibility and ease of use (current qiskit providers include Aer and IBMQ).

## How do I install the QuaC Qiskit Provider?
Neither QuaC nor its Qiskit Provider are on PyPI yet. In order to install QuaC, please proceed to the
QuaC website (linked below under more information) and follow the `README` instructions on `master` to install
QuaC. Then, clone this repository and run `pip install .` in its root directory.

## How do I use QuaC as a provider?
Just the same as you would use Aer or IBMQ as your provider. An object `Quac` is provided upon import for interfacing with and managing backends.
```python
from qiskit import Aer
from qiskit import IBMQ
from qiskit.providers.quac import Quac
```

## What kinds of backends are supported?
You can see the supported backends by running the following command:
```python
from qiskit.providers.quac import Quac
Quac.backends()
```
One backend called `density_simulator` is currently supported, but others will soon be added.
The `density_simulator` backend prints a matrix representing the frequencies of given
quantum experiment outcomes.

## Where can I find more information about QuaC?
The main QuaC repository is located on GitHub [here](https://github.com/0tt3r/QuaC/). A branch called `python-interface` contains Python bindings.
Additionally, you may also wish to check the QuaC website [here](https://0tt3r.github.io/QuaC/).

## Where can I find documentation of this package?
Please check the `docs` folder in this repository.

## Where can I send feature requests?
Please open an `issue` describing the feature you would like to see added.
