# Quantifying Unfairness using EMD
This code implements the _BALANCED_ and _UNBALANCED_ algorithms described in the following paper:

Shady Elbassuoni, Sihem Amer-Yahia, Ahmad Ghizzawi, Christine El Atie. 
[Exploring Fairness of Ranking in Online Job Marketplaces](https://openproceedings.org/2019/conf/edbt/EDBT19_paper_230.pdf). 

# Prerequisites
- Python 3
- MongoDB

## Dependencies
```pip install -r requirements.txt```

## MongoDB
You should have a local MongoDB instance to run the experiments. Please refer to 
https://www.mongodb.com/download-center?jmp=nav#community for Community Server installation 
instructions.

## Setup
After installing and running your local MongoDB instance, run ```python common/add_100k_workers.py```.

# Usage
```
python run_experiments.py [-h] [-c {transparent,opaque_process}]
                          [-w WORKERS] [-b {auto,preset}] [-n NORMALIZE]
                          [-r {min,max,avg}]
```

The following params can be set:
```
General arguments:
  -h, --help            show this help message and exit
  -c {transparent,opaque_process}, --config {transparent,opaque_process}
                        Experiments configuration. (default: transparent)
  -w WORKERS, --workers WORKERS
                        Number of workers. (default: 50)
  -b {auto,preset}, --bins {auto,preset}
                        If bins is auto and quantity is EMD, numpy will decide
                        on the binning strategy for each partition. This will
                        also be used to generate histograms of the function
                        values per partition. (default: preset)

EMD specific arguments.:
  -n NORMALIZE, --normalize NORMALIZE
                        Indicates whether per partition values should be
                        normalized when using EMD. (default: True)
  -r {min,max,avg}, --criterion {min,max,avg}
                        Criterion to be used when (default: avg)
```

## Example
If you want to run the experiments with a transparent configuration using EMD, with 500 workers, 
with normalization, with auto bins, and with avg as the criterion you'll run the following command:

```python run_experiments.py -c transparent -w 500 -n True -b preset -r avg```

Please note that this might take up to 1 hour to terminate. A txt file will be generated at the end of the run containing
a table of the results.
