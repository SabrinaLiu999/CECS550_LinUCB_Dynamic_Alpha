# CECS550 LinUCB Dynamic Alpha

The traditional LinUCB Disjoint algorithm is a contextual bandit algorithm commonly used in recommender systems. Our project implements the traditional LinUCB Disjoint algorithm with tunable alpha parameters to simulate LinUCB policies, evaluate their performance, and visualize the results.



## Files
- **LinUCB_Dynamic_Alpha.ipynb**: Contains input values for different alpha and compares the results of the two algorithms in a single simulation and plotting.
- **50_simulation_comparison.ipynb**: Contains input alpha = 0.25 and compares the results of the two algorithms in 50 times simulations and plotting.
- **code.py**: Contains all code that can be executed locally: comparisons of the original LinUCB Disjointed and LinUCB Disjointed with dynamic alpha.

## Requirements
Environment, Python 3.8
The following Python libraries are required to run the code:
- `numpy`
- `pandas`
- `matplotlib`
- `time`
- `scikit-learn`

You can install these libraries using pip:
```sh

pip install numpy pandas matplotlib scikit-learn
```


 ## Operation 
  1. Download all files
  2. Open Terminal
  3. Change the directory to the folder code.py belongs to folder
  4. Enter the following command: `python code.py`
