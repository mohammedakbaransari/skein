#!/usr/bin/env bash
# scripts/run_tests.sh
# =====================
# Run SKEIN test suite with reporting.
#
# Usage:
#   ./scripts/run_tests.sh              # all tests
#   ./scripts/run_tests.sh unit         # unit tests only
#   ./scripts/run_tests.sh integration  # integration tests
#   ./scripts/run_tests.sh system       # system tests
#   ./scripts/run_tests.sh scenarios    # procurement scenario tests
#   ./scripts/run_tests.sh load         # load/stress tests (slower)
#   ./scripts/run_tests.sh quick        # fastest subset (retry + memory)

set -e
cd "$(dirname "$0")/.."

SUITE="${1:-all}"
PYTHON="${PYTHON:-python3}"
START=$(date +%s)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SKEIN Test Runner"
echo "  Suite: $SUITE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

case "$SUITE" in
  all)
    $PYTHON -m unittest discover -s tests -p "test_*.py" -v
    ;;
  unit)
    $PYTHON -m unittest discover -s tests/unit -p "test_*.py" -v
    ;;
  integration)
    $PYTHON -m unittest tests.integration.test_framework_integration -v
    ;;
  system)
    $PYTHON -m unittest tests.system.test_multi_agent_system -v
    ;;
  scenarios)
    $PYTHON -m unittest tests.scenarios.test_procurement_scenarios -v
    ;;
  load)
    echo "Running load tests (may take 30-60s)..."
    $PYTHON -m unittest tests.load.test_stress_load -v
    ;;
  quick)
    $PYTHON -m unittest tests.unit.test_retry_circuit tests.unit.test_memory -v
    ;;
  *)
    echo "Unknown suite: $SUITE"
    echo "Available: all unit integration system scenarios load quick"
    exit 1
    ;;
esac

END=$(date +%s)
ELAPSED=$((END - START))
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done in ${ELAPSED}s"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
