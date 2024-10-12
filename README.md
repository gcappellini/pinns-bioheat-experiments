README for PINNs Bioheat Experiments Repository

Overview

This repository contains code, models, and results for the observation of bioheat transfer problems using Physics-Informed Neural Networks (PINNs) and comparing their performance with ground truth simulations. The repository is organized into three main folders: models, tests, and src.

Folder Structure

	•	models:
This folder contains trained PINNs models. Each model is uniquely identified by a random hash. For each model, the following files are included:
	•	A trained model file.
	•	A plot showing the evolution of the loss components during training.
	•	A corresponding .yaml configuration file that defines the experiment setup.
	•	tests:
This folder contains the results of the experiments, subdivided into three main categories:
	1.	cooling_simulation:
Contains both MATLAB ground truth and PINNs predictions:
	•	1obs: PINNs validation with a single observer (comparison with MATLAB ground truth).
	•	8obs: PINNs with a multiple-model observer, aiming to outperform MATLAB.
	2.	cooling_meas_1 and cooling_meas_2:
These folders contain the PINNs predictions on measurements performed at AMC Hospital.
	•	src:
Contains the source code for running experiments and additional files:
	•	run.py: The main script to execute the complete set of experiments.
	•	main.py: Used to run individual experiments. Before running main.py, adjust the configurations in the config.yaml file within the src folder.
	•	configs/: This folder contains configuration files (.yaml) for specific experiments.
	•	matlab/: Contains MATLAB scripts for generating ground truth simulations.
	•	data/: Contains measurement data used in some experiments.
	•	Other subfolders are not relevant at the moment.

Additional Files

	•	requirements.txt:
This file lists all the Python dependencies needed to run the experiments. You can install them by running:

pip install -r requirements.txt


	•	Dockerfile:
Note: This file is currently not updated. The Dockerfile is intended for containerizing the project but needs to be revised to match the latest setup.

Usage

Running Experiments

	1.	Complete Experiments:
To run the full set of experiments, simply execute:

python src/run.py


	2.	Single Experiment:
To run a single experiment, modify the config.yaml file located in the src folder. This file controls the parameters for the experiment, including which model, data, and configurations to use. Predefined configurations can be found in the src/configs/ folder. After updating the config.yaml, execute:

python src/main.py

