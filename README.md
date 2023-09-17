# ADIA Project: Exogenous Information in HFT

This repository contains the Python code related to the Applied Finance Project (AFP) by students Felipe Montenegro and Daniel Trivino in partnership with the Abu Dhabi Investment Authority (ADIA).

The project's objective was to understand the role of endogenous and exogenous information effects in high-frequency dynamics, particularly in the context of limit order books.

I invite you to read the [AFP Report](https://acrobat.adobe.com/link/review?uri=urn:aaid:scds:US:2e1301a0-ea68-3e2b-bd25-1350c12ee3f7) for further details about the project and the development of this Python tool.

- **Layer 0 (L0_Library):** The scripts in Layer 0 are used to import all the Python libraries required for the code. Therefore, all Layer 1 code starts with an "import" of Layer 0 files. Additionally, Layer 0 contains the `requirements.txt` file, which manages package dependencies and their respective versions to enable the execution of the set of scripts.

- **Layer 1 (L1_Dev):** After importing all the libraries mentioned in Layer 0, the environment is ready for development. Layer 1 contains all the classes and functions used in the project.

- **Layer 2 (L2_Execution):** Finally, Layer 2 includes all the Python files for execution and Jupyter notebooks displaying the final results.

- **Data:** This folder contains all the data used for the development of the project.

- **Documents:** The PDFs in this folder represent all the papers and notes used as inspiration for this project.
