# Quantifying Unfairness using EMD

# Prerequisites
- Python 3

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
python run_experiments.py [-h] -q {emd,kl} [-c {transparent,opaque_process}]
                          [-w WORKERS] [-b {auto,preset}] [-n NORMALIZE]
                          [-r {min,max,avg}] [-s {standardization,minmax}]
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
If you want to run the experiments with an opaque_process configuration using EMD, with 500 workers, 
with normalization, with auto bins, and with avg as the criterion you'll run the following command:

```python run_experiments.py -q emd -c opaque_process -w 500 -n True -b preset -r avg```