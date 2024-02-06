import os
import pickle
import sys
from dataclasses import dataclass, field

import torch
from transformers import HfArgumentParser


@dataclass
class DataArguments:
    clusters_dump_name: str = field(
        default='clusters/cv16.pkl',
        metadata={'help': 'Name of pkl dump file of clusters list.'}
    )


def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args = parser.parse_args_into_dataclasses()

    # Remove this line if multiple dataclasses exist in parser
    data_args = data_args[0]

    with open(data_args.clusters_dump_name, 'rb') as f:
        clusters_dicts = pickle.load(f)

    print(clusters_dicts)


if __name__ == '__main__':
    main()
