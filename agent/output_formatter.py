import pathlib

from tools.schemas import (
    AgentTrace,
    DatasetProfile,
    FeatureCandidate,
    FormattedOutput,
)


class OutputFormatter:
    def format(self, trace: AgentTrace, profile: DatasetProfile) -> FormattedOutput:
        dataset_name = pathlib.Path(profile.target_col).name  # use target_col as label fallback
        # Build kept_features from iteration records that were kept
        kept_features: list[FeatureCandidate] = []
        for rec in trace.iterations:
            if rec.decision == "kept":
                shap_val = 0.0
                for entry in rec.shap_summary.ranked_features:
                    if entry.feature_name == rec.feature_name:
                        shap_val = entry.mean_abs_shap
                        break
                kept_features.append(
                    FeatureCandidate(
                        name=rec.feature_name,
                        transformation_code=rec.transformation_code,
                        hypothesis=rec.hypothesis,
                        mean_abs_shap=shap_val,
                        auc_delta=rec.auc_delta,
                        decision="kept",
                    )
                )

        auc_lift = trace.final_auc - trace.baseline_auc
        n_iter = len(trace.iterations)
        k = len(kept_features)

        lines = [
            f"Agent ran {n_iter} iterations on {profile.target_col}. "
            f"Baseline AUC: {trace.baseline_auc:.4f}. "
            f"Final AUC: {trace.final_auc:.4f}. "
            f"Lift: {auc_lift:.4f}.",
            f"{k} features kept:",
        ]
        for i, feat in enumerate(kept_features, start=1):
            lines.append(
                f"{i}. {feat.name}: {feat.hypothesis} "
                f"(SHAP contribution: {feat.mean_abs_shap:.4f})"
            )

        report_text = "\n".join(lines)

        return FormattedOutput(
            baseline_auc=trace.baseline_auc,
            final_auc=trace.final_auc,
            auc_lift=auc_lift,
            kept_features=kept_features,
            report_text=report_text,
        )
