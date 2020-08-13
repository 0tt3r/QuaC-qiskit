# The QuaC-Qiskit Plugin

## Introduction and Motivation
This Python package serves as an easily-installable interface between the Qiskit and QuaC libraries. QuaC (Quantum in C) is a fast, highly parallel noisy quantum system simulator developed at Argonne National Laboratory. QuaC evolves the density matrix describing a quantum system by time-stepping through the Lindblad master equation. Therefore, it posseses the benefits of treating time explicitly and modeling noise in a physics-first fashion. In addition, QuaC is specifically designed with scalability in mind, making it an excellent candidate for studying noisy quantum circuits in the NISQ era. This interface is useful for four primary reasons.

1.  It allows researchers to interact with QuaC via an already-familiar interface (i.e., Qiskit)
2.  It requires researchers to perform very little additional work to run noisy quantum simulations on large-scale hardware
3.  It allows researchers to easily observe the effects of physics-based noise on quantum circuits
4.  It provides researchers with new tools to explicitly treat time in their quantum simulations

In short, this plugin allows for the synergistic use of Qiskit and QuaC together. The code in this repository is based on the Qiskit provider/backend code structure for maximum compatibility and ease of use (current Qiskit providers include BasicAer, Aer, and IBMQ).

## Installation
Neither QuaC nor this plugin are on PyPI yet. Please run the following commands to build and install QuaC and its Python bindings (instructions also available on the QuaC repository):

```
git clone -b maint https://bitbucket.org/petsc/petsc petsc
cd petsc

export PETSC_DIR=${PWD}
export PETSC_ARCH=linux-gnu-c-complex 

./configure --with-scalar-type=complex --download-mpich --download-fblaslapack=1 \
  --with-debugging=no COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3 --with-64-bit-indices
make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} all

cd ..
git clone -b maint https://bitbucket.org/slepc/slepc
cd slepc

export SLEPC_DIR=${PWD}
./configure
make SLEPC_DIR=${SLEPC_DIR}

cd ..
git clone -b diagonalize-python-interface https://github.com/0tt3r/QuaC
cd QuaC
make
cd python
make
python setup.py install
```

Now that QuaC is installed, you can install the Qiskit-Quac plugin:

```
git clone https://github.com/0tt3r/QuaC-qiskit
cd QuaC-qiskit
pip install .
```

## Features Overview
The features available in this plugin fall under three primary umbrellas.
#### 1. Noise modeling
The plugin supports the modeling of four types of noise: T1 noise, T2 noise, measurement error, and ZZ coupling error. Noise is defined by using a QuaC noise model object of type `QuacNoiseModel`. The `QuacNoiseModel` constructor takes T1 and T2 noise as lists. The index of a given time constant is its associated qubit index (so, t1=[50000, 60000] would indicate that qubit 0 has a T1 constant of 50000 nanoseconds, for example). T1 and T2 noise must be specified; if you would like T1 and T2 noise to be ignored for a certain qubit, you can set its corresponding time constant to `float('inf')`. 

The user has the option to specify measurement error and ZZ coupling terms. The constructor takes measurement error as a list of 2x2 matrices, with index again specifying the corresponding qubit. The matrices should be formulated as follows:

| P(measure 0 | prepped 0) P(measure 0 | prepped 1) |
| P(measure 1 | prepped 0) P(measure 1 | prepped 1) |

The constructor takes ZZ coupling terms as a dictionary mapping ordered pairs of qubit indices to GHz ZZ values. Due to the symmetry of adding ZZ coupling to qubit pair Hamiltonians, ZZ coupling should only be specified for qubit index pairs where the first element is smaller than the second. As an example, the ZZ coupling between qubits 0 and 1 should be specified in the dictionary as {(0, 1): ZZ_val}, while (1, 0) should be left out.

Here is an example of specifying a QuaC noise model with all four noise components present:
```python
import numpy as np
from qiskit.providers.quac.models import QuacNoiseModel

noise_model = QuacNoiseModel([1234, 1324, 1432, 1342, 1243],
                             [100123, 100432, 10234, 10233, 12543],
                             [np.array([[0.99, 0.02], [0.01, 0.98]]), np.eye(2), np.eye(2), np.eye(2), np.eye(2)],
                             [(0, 1): 2e-5, (0, 2): 1e-5 ... (3, 4): 9e-5])
```
Now, the noise model can be injected into the Qiskit native command `execute` via the `quac_noise_model` key:
```python
from qiskit import QuantumCircuit, execute
from qiskit.providers.quac import Quac
from qiskit.visualization import plot_histogram

backend = Quac.get_backend('fake_yorktown_density_simulator')
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

quac_job = execute(quantum_circuit, backend, quac_noise_model=noise_model)
plot_histogram(quac_job.result().get_counts())
```

**Please note** that if you wish to include ZZ coupling but not measurement error, you still must specify measurement matrices as 2x2 identities in the constructor! **Please also note** that the standard time unit of the plugin is nanoseconds.

#### 2. Optimization
The plugin also offers tools for optimizing noise models against real hardware results. Due to the fact that noise changes on the order of minutes, it would be useful to have a method of extracting the correct noise parameters from a set of circuits run on a quantum computer.

The plugin attempts to solve this problem by allowing the user to guess an initial noise model somewhere near the correct range. The plugin will then simulate a set of user-provided circuits with the initial noise model guess and measure the sum of a divergence metric between the calculated probability distributions and hardware probability distributions of the same circuits. The parameters are optimized via a gradient-free optimizer until the divergence metric is minimized.

The user may specify their own divergence metric, or use one of the three provided in the `qiskit.providers.quac.optimization` package. Please see an example of using plugin optimization tools below. Please note that in this case, the plugin is used in place of hardware, and the circuits provided are calibration circuits. This is a proof-of-concept to show that an optimizer can recover T1 and T2 times from a given noise model, although it is user-defined.

```python
from qiskit import IBMQ, execute
from qiskit.providers.quac import Quac
from qiskit.test.mock import FakeBurlington
from qiskit.ignis.characterization.coherence import t1_circuits, t2_circuits
from qiskit.providers.quac.optimization import *
from qiskit.providers.quac.models import QuacNoiseModel

# Get backends
IBMQ.load_account()
provider = IBMQ.get_provider(hub="ibm-q-ornl", group="anl", project="chm168")
backend = provider.get_backend("ibmq_burlington")
plugin_backend = Quac.get_backend("fake_burlington_density_simulator", t1=True, t2=True, meas=False, zz=False)

# Get calibration circuits
hardware_properties = FakeBurlington().properties()
num_gates = np.linspace(10, 500, 30, dtype='int')
qubits = list(range(len(backend.properties().qubits)))

t1_circs, t1_delay = t1_circuits(num_gates,
                                 hardware_properties.gate_length('id', [0]) * 1e9,
                                 qubits)

t2_circs, t2_delay = t2_circuits((num_gates / 2).astype('int'),
                                 hardware_properties.gate_length('id', [0]) * 1e9,
                                 qubits)

# Formulate real noise model
real_noise_model = QuacNoiseModel(
    t1_times=[1234, 2431, 2323, 2222, 3454],
    t2_times=[12000, 14000, 14353, 20323, 30232]
)

# Formulate initial guess noise model (only same order of magnitude)
guess_noise_model = QuacNoiseModel(
    t1_times=[1000, 1000, 1000, 1000, 1000],
    t2_times=[10000, 10000, 10000, 10000, 10000]
)

# Calculate calibration circuit reference results
reference_job = execute(t1_circs + t2_circs, plugin_backend,
                        quac_noise_model=real_noise_model,
                        optimization_level=0)
reference_result = reference_job.result()

# Calculate optimized noise model
new_noise_model = optimize_noise_model_ng(
    guess_noise_model=guess_noise_model,
    circuits=t1_circs + t2_circs,
    backend=plugin_backend,
    reference_result=reference_result,
    loss_function=angle_objective_function
)

# Show original noise model and optimized noise model
print(f"Original noise model: {real_noise_model}")
print(f"New noise model: {new_noise_model}")
```

These tools are currently under construction and testing, but are nevertheless available for use.

#### 3. Utilities
The QuaC-qiskit plugin offers a variety of utilities designed to improve user experience. These utilities can be found in the `qiskit.providers.quac.format` and `qiskit.providers.quac.stat` packages. Please see the documentation pages for more information about these utilities.

#### 4. Customized Circuit Scheduling and Timing
A user wishing to exert fine-grain control over the specific time each gate in a given circuit is executed may inject a list of gate times into the Qiskit `execute` function using the `gate_times` key. Please note that the circuit must be pre-transpiled; otherwise, the list of times will not match the gates generated after swaps are generated to conform to device coupling maps, for example.

Other injectable parameters are `dt`, which specifies that time step the Lindblad solver should use, and `simulation_length`, which specifies the length of time for which time stepping should proceed. These parameters should be added into the `kwargs` of the `execute` function, just as gate times are above.

## Examples
Examples can be found under the `examples` folder in the plugin source code.

## Hardware Comparisons
(In progress!)
