# dummy
#!/usr/bin/env python3
"""
Regression Summary Extractor (UIO configs)

What it does (per your description):
- Input: regression root path
- Detect configs: any directory under regression root whose name CONTAINS 'uio'
  and has a 'sim/' subdirectory.
- Inside each config/sim:
  - Compile dirs are dynamic but always start with: 'compile_dw_'
  - There will be (typically) a VDUT compile dir and a VIP compile dir
    (we detect VDUT vs VIP by substring in directory name: 'vdut' or 'vip')
  - Compile status in <compile_dir>/test.log:
      PASS if second-last non-empty line contains "database successfully generated"
      FAIL otherwise, and we extract errors by grepping "Error-["
- If compile is clean (VDUT=PASS and VIP=PASS):
  - Simulation test dirs start with test name and usually contain 'vtb'
    (we detect any dir under sim/ that is NOT compile_dw_* and contains 'vtb')
  - For each test dir, check <test_dir>/test.log:
      FAIL if contains UVM_ERROR or UVM_FATAL (count them)
      PASS otherwise
- Output:
  - TSV file: <reg_root>/regression_summary.tsv (Excel-friendly)
  - Adds a header line including number of detected configs

Usage:
  python3 regress_summary_uio.py /path/to/regression
  python3 regress_summary_uio.py /path/to/regression -o out.tsv
  python3 regress_summary_uio.py /path/to/regression --no-per-test
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


DB_OK_STR = "database successfully generated"
ERR_PATTERN = re.compile(r"Error-\[", re.IGNORECASE)
UVM_ERR_PATTERN = re.compile(r"\bUVM_ERROR\b")
UVM_FATAL_PATTERN = re.compile(r"\bUVM_FATAL\b")


@dataclass
class CompileResult:
    lp: str  # "VDUT" or "VIP"
    status: str  # "PASS" / "FAIL" / "NA"
    log_path: Optional[Path]
    errors: List[str]
    chosen_dir: Optional[str]


@dataclass
class TestResult:
    test_dir: str
    status: str  # PASS/FAIL
    uvm_error_count: int
    uvm_fatal_count: int
    log_path: Optional[Path]


def safe_read_lines(log_path: Path, max_bytes: int = 10_000_000) -> List[str]:
    """Read log lines safely with a size cap."""
    try:
        with log_path.open("rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            data = data[:max_bytes]
        return data.decode(errors="replace").splitlines()
    except (FileNotFoundError, OSError):
        return []


def second_last_nonempty_line_contains(lines: List[str], needle: str) -> bool:
    """Check if second last non-empty line contains needle."""
    if not lines:
        return False
    trimmed = [l for l in lines if l.strip() != ""]
    if not trimmed:
        return False
    if len(trimmed) < 2:
        return needle in trimmed[-1]
    return needle in trimmed[-2]


def extract_compile_errors(lines: List[str], max_lines: int = 30) -> List[str]:
    """Extract up to max_lines unique compile errors containing Error-[."""
    errs = [l.strip() for l in lines if ERR_PATTERN.search(l)]
    seen = set()
    uniq: List[str] = []
    for e in errs:
        if e not in seen:
            uniq.append(e)
            seen.add(e)
        if len(uniq) >= max_lines:
            break
    return uniq


def count_uvm(lines: List[str]) -> Tuple[int, int]:
    """Count UVM_ERROR and UVM_FATAL occurrences."""
    err_cnt = 0
    fatal_cnt = 0
    for l in lines:
        if UVM_ERR_PATTERN.search(l):
            err_cnt += 1
        if UVM_FATAL_PATTERN.search(l):
            fatal_cnt += 1
    return err_cnt, fatal_cnt


def find_compile_dirs(sim_dir: Path) -> List[Path]:
    """Compile directories begin with compile_dw_."""
    if not sim_dir.is_dir():
        return []
    return [p for p in sim_dir.iterdir() if p.is_dir() and p.name.startswith("compile_dw_")]


def classify_lp_from_dirname(name: str) -> Optional[str]:
    n = name.lower()
    if "vdut" in n:
        return "VDUT"
    if "vip" in n:
        return "VIP"
    return None


def pick_newest(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def compile_status_for_lp(sim_dir: Path, lp: str) -> CompileResult:
    """
    Detect compile status for given LP (VDUT/VIP).
    Picks the newest compile dir matching that LP.
    """
    compile_dirs = find_compile_dirs(sim_dir)
    candidates = [d for d in compile_dirs if classify_lp_from_dirname(d.name) == lp]

    chosen = pick_newest(candidates)
    if chosen is None:
        return CompileResult(lp=lp, status="NA", log_path=None, errors=[], chosen_dir=None)

    log_path = chosen / "test.log"
    lines = safe_read_lines(log_path)

    if second_last_nonempty_line_contains(lines, DB_OK_STR):
        return CompileResult(lp=lp, status="PASS", log_path=log_path if log_path.exists() else None,
                             errors=[], chosen_dir=chosen.name)
    else:
        errs = extract_compile_errors(lines)
        return CompileResult(lp=lp, status="FAIL", log_path=log_path if log_path.exists() else None,
                             errors=errs, chosen_dir=chosen.name)


def find_test_dirs(sim_dir: Path) -> List[Path]:
    """
    Test directories:
    - under sim/
    - NOT starting with compile_dw_
    - contains 'vtb' in name (based on your examples)
    """
    if not sim_dir.is_dir():
        return []
    out: List[Path] = []
    for p in sim_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("compile_dw_"):
            continue
        if "vtb" not in p.name.lower():
            continue
        out.append(p)
    return sorted(out, key=lambda x: x.name)


def parse_test_results(sim_dir: Path) -> List[TestResult]:
    results: List[TestResult] = []
    for tdir in find_test_dirs(sim_dir):
        log_path = tdir / "test.log"
        lines = safe_read_lines(log_path)
        uerr, ufat = count_uvm(lines)
        status = "PASS" if (uerr == 0 and ufat == 0) else "FAIL"
        results.append(
            TestResult(
                test_dir=tdir.name,
                status=status,
                uvm_error_count=uerr,
                uvm_fatal_count=ufat,
                log_path=log_path if log_path.exists() else None,
            )
        )
    return results


def is_config_compile_clean(vdut: CompileResult, vip: CompileResult) -> bool:
    """Define compile clean as both VDUT and VIP compile PASS."""
    return (vdut.status == "PASS") and (vip.status == "PASS")


def discover_configs(reg_root: Path) -> List[Path]:
    """
    Config dirs:
    - immediate subdirs under regression root
    - directory name contains 'uio' (case-insensitive)
    - contains sim/ directory
    """
    if not reg_root.is_dir():
        return []
    configs: List[Path] = []
    for p in reg_root.iterdir():
        if not p.is_dir():
            continue
        if "uio" not in p.name.lower():
            continue
        if (p / "sim").is_dir():
            configs.append(p)
    return sorted(configs, key=lambda x: x.name)


def format_compile_line(cfg: str, lp: str, comp: CompileResult, today: str) -> str:
    details = comp.status
    if comp.chosen_dir:
        details += f" | dir: {comp.chosen_dir}"
    if comp.status == "FAIL" and comp.errors:
        details += " | " + "; ".join(comp.errors[:5])
        if len(comp.errors) > 5:
            details += " ..."
    if comp.log_path:
        details += f" | log: {comp.log_path}"
    return f"{cfg}\t{lp}\tCompilation Status: {details}\t{today}"


def format_run_lines(tests: List[TestResult], today: str, per_test: bool) -> List[str]:
    lines: List[str] = []
    if not tests:
        return [f"\t\tRun Results: NA (no test dirs found)\t{today}"]

    passed = sum(1 for t in tests if t.status == "PASS")
    failed = sum(1 for t in tests if t.status == "FAIL")
    lines.append(f"\t\tRun Results: TOTAL={len(tests)} PASS={passed} FAIL={failed}\t{today}")

    if per_test:
        for t in tests:
            extra = f"{t.test_dir} => {t.status} (UVM_ERROR={t.uvm_error_count}, UVM_FATAL={t.uvm_fatal_count})"
            if t.log_path:
                extra += f" | log: {t.log_path}"
            lines.append(f"\t\t{extra}\t{today}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize regression results for configs containing 'uio'.")
    ap.add_argument("regression_path", help="Regression root directory path")
    ap.add_argument(
        "-o", "--out",
        default=None,
        help="Output TSV file path (default: <regression_path>/regression_summary.tsv)"
    )
    ap.add_argument(
        "--no-per-test",
        action="store_true",
        help="Do not include per-test breakdown lines; only totals."
    )
    args = ap.parse_args()

    reg_root = Path(args.regression_path).resolve()
    out_path = Path(args.out).resolve() if args.out else (reg_root / "regression_summary.tsv")
    today = dt.date.today().isoformat()

    configs = discover_configs(reg_root)
    cfg_count = len(configs)

    if cfg_count == 0:
        raise SystemExit(
            f"No config directories found under: {reg_root}\n"
            f"Expected: subdirs whose names contain 'uio' and have a 'sim/' directory."
        )

    rows: List[str] = []
    # Extra header lines (nice for humans; Excel still fine)
    rows.append(f"# Regression path:\t{reg_root}")
    rows.append(f"# Detected configs (name contains 'uio'):\t{cfg_count}")
    rows.append("")  # spacer

    rows.append("CONFIG NAME\tLP\tCompilation /Run status\tTodays Date")

    for cfg_dir in configs:
        cfg = cfg_dir.name
        sim_dir = cfg_dir / "sim"

        vdut = compile_status_for_lp(sim_dir, "VDUT")
        vip = compile_status_for_lp(sim_dir, "VIP")

        # Print compile status rows
        rows.append(format_compile_line(cfg, "VDUT", vdut, today))
        rows.append(f"\t\tRun Results:\t{today}")  # placeholder to mimic your screenshot layout

        rows.append(format_compile_line(cfg, "VIP", vip, today))
        rows.append(f"\t\tRun Results:\t{today}")  # placeholder

        # Run results
        if is_config_compile_clean(vdut, vip):
            tests = parse_test_results(sim_dir)
            rows.extend(format_run_lines(tests, today, per_test=not args.no_per_test))
        else:
            rows.append(f"\t\tRun Results: SKIPPED (compile not clean)\t{today}")

        rows.append("")  # separator between configs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"Report generated: {out_path}")
    print(f"Configs detected: {cfg_count}")


if __name__ == "__main__":
    main()
