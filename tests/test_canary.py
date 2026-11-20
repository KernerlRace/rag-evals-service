from ops.canary_check import VariantStats, decide

STABLE = VariantStats(requests=300, error_rate=0.0, p95_seconds=4.0, abstain_rate=0.10)


def test_waits_for_enough_canary_traffic():
    verdict, _ = decide(STABLE, VariantStats(10, 0.0, 4.0, 0.1))
    assert verdict == "WAIT"


def test_promotes_when_canary_is_no_worse():
    verdict, _ = decide(STABLE, VariantStats(100, 0.0, 5.0, 0.12))  # +1 s p95 is within 1.25x + 0.5 s
    assert verdict == "PROMOTE"


def test_rolls_back_on_errors_latency_or_abstentions():
    assert decide(STABLE, VariantStats(100, 0.05, 4.0, 0.1))[0] == "ROLLBACK"
    assert decide(STABLE, VariantStats(100, 0.0, 6.0, 0.1))[0] == "ROLLBACK"
    verdict, reasons = decide(STABLE, VariantStats(100, 0.0, 4.0, 0.30))
    assert verdict == "ROLLBACK" and "abstain" in reasons[0]
