#!/usr/bin/env python3
"""
Unified test runner for GraphVAE project.

Runs all tests in organized test suite.
"""

import sys
import subprocess
from pathlib import Path


def run_test_file(test_path: Path, verbose: bool = True):
    """Run a single test file."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running: {test_path.relative_to(Path.cwd())}")
        print('='*70)

    result = subprocess.run(
        [sys.executable, str(test_path)],
        capture_output=not verbose,
        text=True
    )

    return result.returncode == 0


def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description='Run GraphVAE test suite')
    parser.add_argument('--suite', choices=['all', 'unit', 'spec', 'integration'],
                       default='all', help='Which test suite to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--fast', action='store_true',
                       help='Skip slow tests')

    args = parser.parse_args()

    tests_dir = Path('tests')

    # Define test suites
    test_suites = {
        'unit': [
            tests_dir / 'unit' / 'test_dataset.py',
            tests_dir / 'unit' / 'test_models.py',
            tests_dir / 'unit' / 'test_losses.py',
            tests_dir / 'unit' / 'test_component_substitution.py',
        ],
        'spec': [
            tests_dir / 'spec_generation' / 'test_spec_basic.py',
            tests_dir / 'spec_generation' / 'test_bandpass_spec.py',
            tests_dir / 'spec_generation' / 'test_spec_generation.py',
        ],
        'integration': [
            # Add integration tests here when created
        ]
    }

    # Select tests to run
    if args.suite == 'all':
        tests_to_run = []
        for suite_tests in test_suites.values():
            tests_to_run.extend(suite_tests)
    else:
        tests_to_run = test_suites[args.suite]

    # Run tests
    print("\n" + "="*70)
    print("GRAPHVAE TEST SUITE")
    print("="*70)
    print(f"\nRunning {len(tests_to_run)} test files...")

    results = {}
    for test_path in tests_to_run:
        if not test_path.exists():
            print(f"\n⚠️  Skipping {test_path.name} (not found)")
            continue

        passed = run_test_file(test_path, verbose=args.verbose)
        results[test_path.name] = passed

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed_count = sum(results.values())
    total_count = len(results)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<35} {status}")

    print(f"\nPassed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
