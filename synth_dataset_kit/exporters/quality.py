from __future__ import annotations

import json
import logging
from pathlib import Path

from synth_dataset_kit.models import QualityReport

logger = logging.getLogger(__name__)


def export_quality_report_html(report: QualityReport, path: str) -> str:
    """Generate an HTML quality report."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Score distribution for chart
    buckets = sorted(report.score_distribution.items())
    [b[0] for b in buckets]
    chart_values = [b[1] for b in buckets]

    # Topic coverage
    topics = list(report.topic_coverage.items())[:15]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quality Report — {report.dataset_name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 2rem; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; color: #fff; }}
  .subtitle {{ color: #888; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #1a1a1a; border-radius: 12px; padding: 1.5rem; border: 1px solid #2a2a2a; }}
  .card .label {{ color: #888; font-size: 0.85rem; margin-bottom: 0.5rem; }}
  .card .value {{ font-size: 1.8rem; font-weight: 700; }}
  .card .value.green {{ color: #4ade80; }}
  .card .value.yellow {{ color: #fbbf24; }}
  .card .value.red {{ color: #f87171; }}
  .card .value.blue {{ color: #60a5fa; }}
  .section {{ background: #1a1a1a; border-radius: 12px; padding: 1.5rem; border: 1px solid #2a2a2a; margin-bottom: 1.5rem; }}
  .section h2 {{ font-size: 1.1rem; margin-bottom: 1rem; color: #fff; }}
  .bar-chart {{ display: flex; flex-direction: column; gap: 0.5rem; }}
  .bar-row {{ display: flex; align-items: center; gap: 0.5rem; }}
  .bar-label {{ width: 120px; font-size: 0.8rem; color: #aaa; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .bar-track {{ flex: 1; height: 24px; background: #2a2a2a; border-radius: 4px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; display: flex; align-items: center; padding-left: 8px; font-size: 0.75rem; color: #fff; }}
  .bar-fill.score {{ background: linear-gradient(90deg, #4ade80, #22c55e); }}
  .bar-fill.topic {{ background: linear-gradient(90deg, #60a5fa, #3b82f6); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin: 2px; }}
  .badge.warn {{ background: #fbbf2433; color: #fbbf24; }}
  .badge.ok {{ background: #4ade8033; color: #4ade80; }}
  .footer {{ text-align: center; color: #555; font-size: 0.8rem; margin-top: 2rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>📊 Quality Report</h1>
  <p class="subtitle">{report.dataset_name} — {report.total_examples} examples</p>

  <div class="grid">
    <div class="card">
      <div class="label">Total Examples</div>
      <div class="value blue">{report.total_examples}</div>
    </div>
    <div class="card">
      <div class="label">Passed (≥{7.0})</div>
      <div class="value green">{report.passed_examples}</div>
    </div>
    <div class="card">
      <div class="label">Failed</div>
      <div class="value {"red" if report.failed_examples > 0 else "green"}">{report.failed_examples}</div>
    </div>
    <div class="card">
      <div class="label">Avg Quality Score</div>
      <div class="value {"green" if report.avg_quality_score >= 7 else "yellow" if report.avg_quality_score >= 5 else "red"}">{report.avg_quality_score:.1f}</div>
    </div>
    <div class="card">
      <div class="label">Diversity Score</div>
      <div class="value {"green" if report.diversity_score >= 0.7 else "yellow"}">{report.diversity_score:.2f}</div>
    </div>
    <div class="card">
      <div class="label">Self-BLEU Proxy</div>
      <div class="value {"green" if report.self_bleu_proxy <= 0.3 else "yellow"}">{report.self_bleu_proxy:.2f}</div>
    </div>
    <div class="card">
      <div class="label">Lexical Diversity</div>
      <div class="value {"green" if report.lexical_diversity >= 0.2 else "yellow"}">{report.lexical_diversity:.2f}</div>
    </div>
    <div class="card">
      <div class="label">Contamination Hits</div>
      <div class="value {"red" if report.contamination_hits > 0 else "green"}">{report.contamination_hits}</div>
    </div>
    <div class="card">
      <div class="label">Near Duplicates</div>
      <div class="value {"yellow" if report.near_duplicate_examples > 0 else "green"}">{report.near_duplicate_examples}</div>
    </div>
    <div class="card">
      <div class="label">Embedding Diversity</div>
      <div class="value {"green" if (report.embedding_diversity_score or 0) >= 0.65 else "yellow"}">{report.embedding_diversity_score if report.embedding_diversity_score is not None else "n/a"}</div>
    </div>
  </div>

  <div class="grid" style="grid-template-columns: 1fr 1fr;">
    <div class="card">
      <div class="label">Avg User Message</div>
      <div class="value blue">{report.avg_user_length:.0f} <span style="font-size:0.8rem;color:#888">words</span></div>
    </div>
    <div class="card">
      <div class="label">Avg Assistant Response</div>
      <div class="value blue">{report.avg_assistant_length:.0f} <span style="font-size:0.8rem;color:#888">words</span></div>
    </div>
  </div>

  <div class="section">
    <h2>Score Distribution</h2>
    <div class="bar-chart">
"""

    max_count = max(chart_values) if chart_values else 1
    for label, value in buckets:
        pct = (value / max_count) * 100
        html += f"""      <div class="bar-row">
        <div class="bar-label">{label}</div>
        <div class="bar-track"><div class="bar-fill score" style="width:{pct}%">{value}</div></div>
      </div>
"""

    html += """    </div>
  </div>

  <div class="section">
    <h2>Topic Coverage</h2>
    <div class="bar-chart">
"""

    max_topic = max((t[1] for t in topics), default=1)
    for topic_name, count in topics:
        pct = (count / max_topic) * 100
        display_name = topic_name[:40] + "..." if len(topic_name) > 40 else topic_name
        html += f"""      <div class="bar-row">
        <div class="bar-label" title="{topic_name}">{display_name}</div>
        <div class="bar-track"><div class="bar-fill topic" style="width:{pct}%">{count}</div></div>
      </div>
"""

    html += """    </div>
  </div>
"""

    if report.difficulty_distribution:
        html += """  <div class="section">
    <h2>Difficulty Distribution</h2>
    <div class="bar-chart">
"""
        max_diff = max(report.difficulty_distribution.values())
        for difficulty_name, count in report.difficulty_distribution.items():
            pct = (count / max_diff) * 100
            html += f"""      <div class="bar-row">
        <div class="bar-label">{difficulty_name}</div>
        <div class="bar-track"><div class="bar-fill score" style="width:{pct}%">{count}</div></div>
      </div>
"""
        html += """    </div>
  </div>
"""

    if report.issue_counts:
        html += """  <div class="section">
    <h2>Rule-Based Issues</h2>
    <div class="bar-chart">
"""
        max_issue_count = max(report.issue_counts.values())
        for issue_name, count in report.issue_counts.items():
            pct = (count / max_issue_count) * 100
            html += f"""      <div class="bar-row">
        <div class="bar-label">{issue_name}</div>
        <div class="bar-track"><div class="bar-fill topic" style="width:{pct}%">{count}</div></div>
      </div>
"""
        html += """    </div>
  </div>
"""

    if report.topic_heatmap:
        html += """  <div class="section">
    <h2>Topic Heatmap</h2>
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <thead>
        <tr>
          <th style="text-align:left;padding:8px;border-bottom:1px solid #2a2a2a;">Topic</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Easy</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Medium</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Hard</th>
        </tr>
      </thead>
      <tbody>
"""
        for topic_name, difficulty_counts in list(report.topic_heatmap.items())[:15]:
            html += f"""        <tr>
          <td style="padding:8px;border-bottom:1px solid #222;">{topic_name}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{difficulty_counts.get("easy", 0)}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{difficulty_counts.get("medium", 0)}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{difficulty_counts.get("hard", 0)}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
  </div>
"""

    if report.seed_cluster_distribution or report.generated_cluster_distribution:
        html += """  <div class="section">
    <h2>Seed vs Generated Distribution</h2>
    <div style="margin-bottom:12px;color:#aaa;">Divergence: """
        html += f"""{report.distribution_divergence:.4f} · Match Score: {report.distribution_match_score:.2f}/100 · Semantic Coverage: {report.semantic_coverage_score:.2%} · Graph Coverage: {report.graph_coverage_score:.2%}</div>
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <thead>
        <tr>
          <th style="text-align:left;padding:8px;border-bottom:1px solid #2a2a2a;">Cluster</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Planned</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Generated</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Gap</th>
        </tr>
      </thead>
      <tbody>
"""
        cluster_ids = sorted(
            set(report.seed_cluster_distribution) | set(report.generated_cluster_distribution)
        )
        for cluster_id in cluster_ids[:20]:
            planned = report.seed_cluster_distribution.get(cluster_id, 0)
            generated = report.generated_cluster_distribution.get(cluster_id, 0)
            gap = report.underrepresented_clusters.get(cluster_id, 0)
            html += f"""        <tr>
          <td style="padding:8px;border-bottom:1px solid #222;">{cluster_id}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{planned}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{generated}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{gap}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
  </div>
"""
    if report.graph_frontier_clusters:
        html += """  <div class="section">
    <h2>Graph Frontier</h2>
    <div style="color:#aaa;">"""
        html += ", ".join(report.graph_frontier_clusters[:8])
        html += """</div>
  </div>
"""
    if (
        report.semantic_cluster_target_distribution
        or report.semantic_cluster_generated_distribution
    ):
        html += """  <div class="section">
    <h2>Semantic Coverage</h2>
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <thead>
        <tr>
          <th style="text-align:left;padding:8px;border-bottom:1px solid #2a2a2a;">Semantic Cluster</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Target</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Generated</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Gap</th>
        </tr>
      </thead>
      <tbody>
"""
        semantic_cluster_ids = sorted(
            set(report.semantic_cluster_target_distribution)
            | set(report.semantic_cluster_generated_distribution)
        )
        for cluster_id in semantic_cluster_ids[:20]:
            target = report.semantic_cluster_target_distribution.get(cluster_id, 0)
            generated = report.semantic_cluster_generated_distribution.get(cluster_id, 0)
            gap = report.semantic_coverage_gaps.get(cluster_id, 0)
            html += f"""        <tr>
          <td style="padding:8px;border-bottom:1px solid #222;">cluster_{cluster_id}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{target}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{generated}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{gap}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
  </div>
"""

    if report.rebalancing_history:
        html += """  <div class="section">
    <h2>Rebalancing History</h2>
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <thead>
        <tr>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Round</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Requested</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Accepted</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Rejected</th>
          <th style="text-align:right;padding:8px;border-bottom:1px solid #2a2a2a;">Divergence</th>
        </tr>
      </thead>
      <tbody>
"""
        for round_info in report.rebalancing_history[:20]:
            html += f"""        <tr>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{round_info.get("round", 0)}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{round_info.get("requested", 0)}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{round_info.get("accepted_total", 0)}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{round_info.get("rejected_batch", 0)}</td>
          <td style="padding:8px;text-align:right;border-bottom:1px solid #222;">{round_info.get("distribution_divergence", 0.0)}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
  </div>
"""

    if report.contaminated_benchmarks:
        html += """  <div class="section">
    <h2>⚠️ Contamination Detected</h2>
    <p style="margin-bottom:0.5rem;">The following benchmarks had potential overlap:</p>
"""
        for bench in report.contaminated_benchmarks:
            html += f'    <span class="badge warn">{bench}</span>\n'
        html += "  </div>\n"
    else:
        html += """  <div class="section">
    <h2>✅ No Contamination Detected</h2>
    <p>Dataset passed decontamination checks against all configured benchmarks.</p>
  </div>
"""

    if report.contamination_verdicts:
        html += """  <div class="section">
    <h2>Contamination Verdicts</h2>
"""
        for verdict, count in sorted(report.contamination_verdicts.items()):
            html += f'    <span class="badge {"warn" if verdict != "clean" else "ok"}">{verdict}: {count}</span>\n'
        html += "  </div>\n"

    if report.contamination_methods:
        html += """  <div class="section">
    <h2>Contamination Methods</h2>
    <div class="bar-chart">
"""
        max_method_count = max(report.contamination_methods.values())
        for method_name, count in sorted(
            report.contamination_methods.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            pct = (count / max_method_count) * 100
            html += f"""      <div class="bar-row">
        <div class="bar-label">{method_name}</div>
        <div class="bar-track"><div class="bar-fill topic" style="width:{pct}%">{count}</div></div>
      </div>
"""
        html += """    </div>
  </div>
"""

    if report.contamination_method_benchmarks:
        html += """  <div class="section">
    <h2>Contamination Evidence By Benchmark</h2>
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <thead>
        <tr>
          <th style="text-align:left;padding:8px;border-bottom:1px solid #2a2a2a;">Benchmark</th>
          <th style="text-align:left;padding:8px;border-bottom:1px solid #2a2a2a;">Methods</th>
        </tr>
      </thead>
      <tbody>
"""
        for bench_name, method_counts in sorted(report.contamination_method_benchmarks.items()):
            method_text = ", ".join(
                f"{method}: {count}"
                for method, count in sorted(
                    method_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            )
            html += f"""        <tr>
          <td style="padding:8px;border-bottom:1px solid #222;">{bench_name}</td>
          <td style="padding:8px;border-bottom:1px solid #222;">{method_text}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
  </div>
"""

    if report.contamination_evidence_samples:
        html += """  <div class="section">
    <h2>Contamination Evidence Samples</h2>
"""
        for sample in report.contamination_evidence_samples[:10]:
            matched_text = str(sample.get("matched_text", ""))[:180]
            confidence = sample.get("confidence")
            confidence_text = f" ({confidence})" if confidence is not None else ""
            html += (
                f'    <div style="padding:10px 0;border-bottom:1px solid #222;">'
                f"<strong>{sample.get('benchmark', 'unknown')}</strong> via "
                f'<span class="badge warn">{sample.get("method", "unknown")}{confidence_text}</span>'
                f'<div style="color:#aaa;margin-top:6px;">{matched_text}</div>'
                f"</div>\n"
            )
        html += "  </div>\n"

    if report.benchmark_sources:
        html += """  <div class="section">
    <h2>Benchmark Sources</h2>
"""
        for bench_name, source in report.benchmark_sources.items():
            sample_count = report.benchmark_sample_counts.get(bench_name, 0)
            html += f'    <span class="badge {"ok" if source == "datasets" else "warn"}">{bench_name}: {source} ({sample_count})</span>\n'
        html += "  </div>\n"

    if report.benchmark_load_errors:
        html += """  <div class="section">
    <h2>Benchmark Load Errors</h2>
"""
        for bench_name, error in report.benchmark_load_errors.items():
            html += f'    <span class="badge warn">{bench_name}: {error}</span>\n'
        html += "  </div>\n"

    html += """
  <div class="footer">
    Generated by synth-dataset-kit v0.1.0
  </div>
</div>
</body>
</html>"""

    with open(p, "w") as f:
        f.write(html)

    logger.info(f"Quality report saved to {p}")
    return str(p)


def export_quality_report_json(report: QualityReport, path: str) -> str:
    """Save the quality report as machine-readable JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
    logger.info(f"Quality report JSON saved to {p}")
    return str(p)
