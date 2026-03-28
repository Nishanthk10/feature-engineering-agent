from tools.schemas import EvaluationResult, FeatureShapEntry, ShapSummary


class ShapTool:
    def format_for_llm(self, eval_result: EvaluationResult) -> ShapSummary:
        sorted_features = sorted(
            eval_result.shap_values.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )

        ranked_features = [
            FeatureShapEntry(feature_name=name, mean_abs_shap=value, rank=rank)
            for rank, (name, value) in enumerate(sorted_features, start=1)
        ]

        top_3 = ranked_features[:3]
        parts = [f"({e.rank}) {e.feature_name} (shap={e.mean_abs_shap:.3f})" for e in top_3]
        top_3_summary = "Top features by importance: " + ", ".join(parts)

        return ShapSummary(ranked_features=ranked_features, top_3_summary=top_3_summary)
