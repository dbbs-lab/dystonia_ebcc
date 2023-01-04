# dystonia_ebcc
Code to simulate mouse models of dystonia during EyeBlink Classical Conditioning (EBCC), described in Geminiani et al., *Front Syst Neurosci*, 2022.

Requirements:
- BSB v3.8+
- NEST simulator v 2.18
- cereb-nest NEST extension module (available at https://github.com/dbbs-lab/cereb-nest)

Repository content:
- `simulation` includes all the script to run simulations in control and pathological conditions. The main simulation file is `simulate_network.py`; different simulation configurations are described in the .json files
- `simulation_analysis.py` allows to plot figures from simulation results (which are saved as .hdf5 files) 
