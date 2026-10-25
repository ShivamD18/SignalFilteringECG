"""Command-line interface: ``python -m ecglab <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_download(args):
    from . import data

    data.download(force=args.force)


def cmd_train(args):
    from . import model

    model.train_and_evaluate(args.models)


def cmd_detector(args):
    from . import data, detect, filters

    tp = fp = fn = 0
    print(f"{'record':>6} {'beats':>6} {'TP':>6} {'FP':>5} {'FN':>5} {'Se':>7} {'+P':>7}")
    for name in data.ALL_RECORDS:
        rec = data.load_record(name)
        peaks = detect.detect_r_peaks(filters.preprocess(rec.signal, rec.fs), rec.fs)
        m = detect.match_beats(rec.beat_samples, peaks, rec.fs)
        tp, fp, fn = tp + m.tp, fp + m.fp, fn + m.fn
        print(f"{name:>6} {len(rec.beat_samples):>6} {m.tp:>6} {m.fp:>5} {m.fn:>5} "
              f"{m.sensitivity:>7.2%} {m.ppv:>7.2%}")
    print(f"{'total':>6} {tp + fn:>6} {tp:>6} {fp:>5} {fn:>5} {tp / (tp + fn):>7.2%} {tp / (tp + fp):>7.2%}")


def cmd_report(args):
    from . import report

    if args.record:
        an = report.analyze_record(args.record, args.start, args.duration)
        stem = f"record_{args.record}"
    else:
        signal = report.load_csv(Path(args.csv), args.column if not args.column.isdigit() else int(args.column))
        signal = signal * args.scale
        an = report.analyze(signal, args.fs, title=Path(args.csv).name)
        stem = Path(args.csv).stem
    path = report.write_report(an, Path(args.out), stem)
    print(report.to_markdown(an))
    print(f"\nSaved {path} and {path.with_suffix('.png')}")


def cmd_export_web(args):
    from . import export_web

    export_web.export_all()


def cmd_calibrate_af(args):
    from . import rhythm

    print(json.dumps(rhythm.calibrate_irregularity(), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ecglab", description="ECG Signal Lab")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("download", help="download MIT-BIH data from PhysioNet")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("train", help="train + evaluate classifiers (inter-patient DS1 -> DS2)")
    s.add_argument("--models", nargs="*", help="subset of: logistic_regression random_forest gradient_boosting mlp")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate-detector", help="Pan-Tompkins sensitivity / PPV on all 48 records")
    s.set_defaults(func=cmd_detector)

    s = sub.add_parser("report", help="analyse an MIT-BIH record or your own CSV and write a report")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--record", help="MIT-BIH record name, e.g. 208")
    src.add_argument("--csv", help="CSV/TXT file with one ECG column")
    s.add_argument("--start", type=float, default=0.0, help="start time in seconds (records)")
    s.add_argument("--duration", type=float, default=60.0, help="seconds to analyse (records)")
    s.add_argument("--fs", type=float, default=360.0, help="sampling rate of the CSV in Hz")
    s.add_argument("--column", default="0", help="CSV column index or header name")
    s.add_argument("--scale", type=float, default=1.0, help="multiply CSV values to get millivolts")
    s.add_argument("--out", default="reports", help="output folder")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("export-web", help="export sample data + trained model for the website")
    s.set_defaults(func=cmd_export_web)

    s = sub.add_parser("calibrate-af", help="re-derive the atrial fibrillation irregularity threshold")
    s.set_defaults(func=cmd_calibrate_af)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
