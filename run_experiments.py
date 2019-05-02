import argparse

from disparity.emd import EMD
from disparity.helpers import Helper

# Opaque Process
percentages = {
    10: 0.1,
    30: 0.3,
    50: 0.5,
    # 'gender': 'g',
    # 'gender_country': 'gc'
}

# Opaque Dataset
K = {
    50: [10],
    500: [10],
    7300: [10]
}

## movielens
# F = {
#     1: 1,
#     2: 2,
#     3: 3,
# }


def export_tables(name, content):
    with open(name + '.txt', 'w') as w:
        w.write(content)


def run(bins, config, criterion, normalize, workers):
    db = "WorkerSet100K"
    collection = 'workers'
    ## simulated
    F = {
        1: [0.3, 0.7],
        2: [0.7, 0.3],
        3: [0.5, 0.5],
        4: [1, 0],
        5: [0, 1],
        6: '6'
    }
    helper = Helper(configuration=config, N=workers, db_name=db, collection_name=collection)
    workers = helper.get_documents()
    attributes = helper.get_attributes(workers)
    quantify_disparity_metric = EMD

    name, values, time_values = helper.run_experiments(quantify_disparity_metric, workers, attributes, functions=F,
                                                       percentages=percentages, bins=bins, criterion=criterion,
                                                       normalize=normalize,)

    table, timetable = helper.build_tables(name, values, time_values, functions=F, percentages=percentages)
    export_tables(name, str(table) + '\n' + str(timetable))


def main():
    """Main
    """

    parser = argparse.ArgumentParser(description='Run fairness experiment on the 100K simulated dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', "--config", type=str, help='Experiments configuration.',
                        choices=['transparent', 'opaque_process'], default='transparent')
    parser.add_argument('-w', "--workers", type=int, help='Number of workers.', default=50)
    parser.add_argument('-b', "--bins", type=str, help='If bins is auto and quantity is EMD, numpy will decide on '
                                                       'the binning strategy for each partition. '
                                                       'This will also be used to generate histograms of the function '
                                                       'values per partition.',
                        default='preset', choices=['auto', 'preset'])

    emd_group = parser.add_argument_group('EMD specific arguments.')
    emd_group.add_argument('-n', '--normalize', type=lambda x: (str(x).lower() == 'true'),
                           help='Indicates whether per partition values should be normalized when using EMD.',
                           default=True)
    emd_group.add_argument('-r', "--criterion", type=str, help='Criterion to be used when ', default='avg',
                           choices=['min', 'max', 'avg'])

    args = parser.parse_args()  # parse arguments from command line

    run(args.bins, args.config, args.criterion, args.normalize, args.workers)


if __name__ == "__main__":
    # execute only if run as a script
    main()
