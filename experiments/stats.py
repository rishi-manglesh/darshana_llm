"""Statistical Significance Testing for Win Rate Experiments

Provides bootstrap confidence intervals, effect sizes, and pairwise
comparison tables for evaluating darshana training experiments.
"""

import math
import random as pyrandom
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def bootstrap_ci(wins, total, n_boot=10000, alpha=0.05, seed=42):
    """Bootstrap 95% CI for win rate.

    Args:
        wins: number of wins
        total: total comparisons
        n_boot: number of bootstrap samples
        alpha: significance level (default 0.05 for 95% CI)
        seed: random seed

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0, 0.0

    point = wins / total

    if HAS_NUMPY:
        rng = np.random.RandomState(seed)
        samples = rng.binomial(total, point, n_boot) / total
        lower = float(np.percentile(samples, 100 * alpha / 2))
        upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    else:
        pyrandom.seed(seed)
        samples = []
        for _ in range(n_boot):
            s = sum(1 for _ in range(total) if pyrandom.random() < point)
            samples.append(s / total)
        samples.sort()
        lower = samples[int(n_boot * alpha / 2)]
        upper = samples[int(n_boot * (1 - alpha / 2))]

    return point, lower, upper


def cohens_h(p1, p2):
    """Cohen's h effect size for comparing two proportions.

    Args:
        p1: first proportion (e.g., experimental win rate)
        p2: second proportion (e.g., baseline/chance = 0.5)

    Returns:
        float: Cohen's h (small=0.2, medium=0.5, large=0.8)
    """
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def effect_size_label(h):
    """Interpret Cohen's h effect size."""
    h_abs = abs(h)
    if h_abs < 0.2:
        return "negligible"
    elif h_abs < 0.5:
        return "small"
    elif h_abs < 0.8:
        return "medium"
    else:
        return "large"


def format_win_rate(wins, total, n_boot=10000):
    """Format win rate with 95% CI.

    Returns:
        str like "67% [58-75%]"
    """
    if total == 0:
        return "N/A"

    point, lower, upper = bootstrap_ci(wins, total, n_boot=n_boot)
    return f"{point*100:.0f}% [{lower*100:.0f}-{upper*100:.0f}%]"


def pairwise_comparison_table(configs_results, baseline_rate=0.5):
    """Generate a formatted pairwise comparison table.

    Args:
        configs_results: dict mapping config_name -> {"wins": N, "total": N}
        baseline_rate: expected win rate under null hypothesis

    Returns:
        str: formatted table
    """
    lines = []
    lines.append(f"{'Config':<25} {'Win Rate':>15} {'95% CI':>15} {'Effect':>10} {'Size':>12}")
    lines.append("-" * 80)

    for config, stats in sorted(configs_results.items()):
        wins = stats.get("wins", 0)
        total = stats.get("total", 0)
        if total == 0:
            lines.append(f"{config:<25} {'N/A':>15}")
            continue

        point, lower, upper = bootstrap_ci(wins, total)
        h = cohens_h(point, baseline_rate)
        label = effect_size_label(h)

        lines.append(
            f"{config:<25} {point*100:>6.0f}%"
            f"       [{lower*100:.0f}-{upper*100:.0f}%]"
            f"     {h:>+.3f}    {label:>10}"
        )

    return "\n".join(lines)


def is_significant(wins, total, threshold=0.5, alpha=0.05):
    """Test if win rate is significantly above threshold.

    Args:
        wins: number of wins
        total: total comparisons
        threshold: null hypothesis win rate (default 0.5)
        alpha: significance level

    Returns:
        bool: True if lower CI bound > threshold
    """
    if total == 0:
        return False
    _, lower, _ = bootstrap_ci(wins, total, alpha=alpha)
    return lower > threshold


def compute_verdict(configs_results, darshana_config, western_config, random_config):
    """Determine experiment verdict based on win rates.

    Args:
        configs_results: dict of config -> {"wins": N, "total": N}
        darshana_config: name of darshana config
        western_config: name of Western control config
        random_config: name of random control config

    Returns:
        (verdict_str, explanation_str)
    """
    def wr(config):
        s = configs_results.get(config, {})
        t = s.get("total", 0)
        return s.get("wins", 0) / t if t > 0 else 0

    def sig(config):
        s = configs_results.get(config, {})
        return is_significant(s.get("wins", 0), s.get("total", 0))

    d_wr = wr(darshana_config)
    w_wr = wr(western_config)
    r_wr = wr(random_config)
    d_sig = sig(darshana_config)

    if d_wr > w_wr and d_wr > r_wr and d_sig:
        if d_wr - w_wr > 0.05:
            return "PROVEN", (
                f"{darshana_config} ({d_wr*100:.0f}%) > "
                f"{western_config} ({w_wr*100:.0f}%) > "
                f"{random_config} ({r_wr*100:.0f}%)"
            )
        else:
            return "PARTIALLY PROVEN", (
                f"{darshana_config} ({d_wr*100:.0f}%) ~ "
                f"{western_config} ({w_wr*100:.0f}%) > "
                f"{random_config} ({r_wr*100:.0f}%): "
                f"ordering helps but darshana not uniquely special"
            )
    elif (d_wr > r_wr or w_wr > r_wr) and (d_sig or sig(western_config)):
        return "PARTIALLY PROVEN", (
            f"Ordering helps (> random) but darshana not best: "
            f"{darshana_config}={d_wr*100:.0f}%, "
            f"{western_config}={w_wr*100:.0f}%, "
            f"{random_config}={r_wr*100:.0f}%"
        )
    else:
        return "DISPROVEN", (
            f"No significant ordering effect: "
            f"{darshana_config}={d_wr*100:.0f}%, "
            f"{western_config}={w_wr*100:.0f}%, "
            f"{random_config}={r_wr*100:.0f}%"
        )
