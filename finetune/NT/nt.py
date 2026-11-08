"""Entry point for VeriComp's Nucleotide Transformer benchmark tasks."""

from finetune.sequence_classification import build_parser, train_task
from vericomp.benchmark_data import load_nt_splits


def main():
    parser = build_parser("VeriComp Nucleotide Transformer downstream benchmark")
    args = parser.parse_args()
    train_task(args, task_family="NT", dataset_loader=load_nt_splits)


if __name__ == "__main__":
    main()
