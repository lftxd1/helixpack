"""Entry point for VeriComp's GenomicBenchmarks tasks."""

from finetune.sequence_classification import build_parser, train_task
from vericomp.benchmark_data import load_gb_splits


def main():
    parser = build_parser("VeriComp GenomicBenchmarks downstream benchmark")
    args = parser.parse_args()
    train_task(args, task_family="GB", dataset_loader=load_gb_splits)


if __name__ == "__main__":
    main()
