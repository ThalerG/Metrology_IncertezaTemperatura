# Monte Carlo Simulation for Temperature Uncertainty Analysis
This repository contains scripts for performing Monte Carlo simulations to analyze the uncertainty in temperature measurements. The main files are `fast_montecarlo.py` and `montecarlo_noplot.py`.

## Main Files

### [`fast_montecarlo.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fd%3A%2FDocumentos%2FLIAE%2FMetrology_IncertezaTemperatura%2Ffast_montecarlo.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "d:\Documentos\LIAE\Metrology_IncertezaTemperatura\fast_montecarlo.py")

This script performs Monte Carlo simulations in parallel to speed up the computation. It reads the data from [`Dados/data.csv`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fd%3A%2FDocumentos%2FLIAE%2FMetrology_IncertezaTemperatura%2FDados%2Fdata.csv%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "d:\Documentos\LIAE\Metrology_IncertezaTemperatura\Dados\data.csv") and generates a Monte Carlo matrix. The results are saved in the specified file.

#### Usage

```sh
python fast_montecarlo.py --N_montecarlo <number_of_simulations> --fname <output_filename> [--Plot] [--NoSave] [--execTime]
```

Arguments
--N_montecarlo: Number of Monte Carlo simulations (default: 1,000,000)
--fname: Name of the file to save the results (default: montecarlo_results)
--Plot: Enable plotting (optional)
--NoSave: Disable saving results (optional)
--execTime: Enable execution time measurement (optional)


### [`montecarlo_noplot.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fd%3A%2FDocumentos%2FLIAE%2FMetrology_IncertezaTemperatura%2Fmontecarlo_noplot.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "d:\Documentos\LIAE\Metrology_IncertezaTemperatura\montecarlo_noplot.py")

This script performs Monte Carlo simulations without generating plots. It reads the data from `Dados/data.csv` and saves the results in the specified folder.

#### Usage

```sh
python montecarlo_noplot.py --fsave <output_folder> --N_montecarlo <number_of_simulations>
```

Arguments
--fsave: Folder to save the results (default: Resultados)
--N_montecarlo: Number of Monte Carlo simulations (default: 200)

## Additional Files

- `analysis_montecarlo.py`: Script for generating HTML reports of the Monte Carlo analysis.
- `fcn.py`: Contains utility functions used across different scripts.
- `plot_convergence.py`: Script for plotting convergence results.
- `map_montecarlo.py`: Script for mapping Monte Carlo results.
- `montecarlo_temperature.py`: Script for performing Monte Carlo simulations specifically for temperature analysis.

## Data

The data used for the simulations is stored in the `Dados/` directory. The main data file is `data.csv`.

## Results

The results of the simulations are saved in the `Resultados/` directory. Each analysis generates a `.feather` and a `.txt` file.

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- tqdm
- argparse
- multiprocessing

## Installation

To install the required dependencies and set up a virtual environment, follow these steps:

1. Create a virtual environment:
```sh
python -m venv myenv
```

2. Activate the virtual environment:
```sh
source myenv/bin/activate
```

3. Install the dependencies:
```sh
pip install pandas numpy matplotlib tqdm pyarrow scipy
```

Remember to activate the virtual environment every time you work on this project.


## License

This project is licensed under the MIT License.
