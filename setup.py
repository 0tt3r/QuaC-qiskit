import setuptools

with open("README.md", "r") as desc_file:
    long_description = desc_file.read()

with open("requirements.txt", "r") as req_file:
    requirements = req_file.read().split("\n")[:-1]  # by PEP-8, the last line is blank

setuptools.setup(
    name="quac-qiskit",
    version="0.0.1",
    author="The QuaC/Qiskit Integration Development Team",
    author_email="",
    description="A package to allow simulation of quantum systems built in Qiskit on quac simulators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/0tt3r/QuaC-qiskit",
    packages=['qiskit.providers.quac'],
    requirements=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="qiskit quac integration backend",
    python_requires='>=3.7'
)

print("success")
