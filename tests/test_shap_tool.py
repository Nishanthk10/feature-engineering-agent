from tools.schemas import EvaluationResult
from tools.shap_tool import ShapTool


def make_eval_result(shap_values: dict[str, float]) -> EvaluationResult:
    return EvaluationResult(
        auc=0.75,
        f1=0.70,
        shap_values=shap_values,
        feature_names=list(shap_values.keys()),
    )


class TestShapToolRanking:
    def test_ranked_features_sorted_descending(self):
        result = make_eval_result({"age": 0.05, "income": 0.12, "balance": 0.08})
        summary = ShapTool().format_for_llm(result)
        values = [e.mean_abs_shap for e in summary.ranked_features]
        assert values == sorted(values, reverse=True)

    def test_rank_field_is_1_indexed_and_sequential(self):
        result = make_eval_result({"age": 0.05, "income": 0.12, "balance": 0.08})
        summary = ShapTool().format_for_llm(result)
        ranks = [e.rank for e in summary.ranked_features]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_rank_1_is_highest_shap(self):
        result = make_eval_result({"age": 0.05, "income": 0.12, "balance": 0.08})
        summary = ShapTool().format_for_llm(result)
        assert summary.ranked_features[0].feature_name == "income"
        assert summary.ranked_features[0].rank == 1


class TestShapToolTop3Summary:
    def test_top_3_summary_contains_top_3_feature_names(self):
        result = make_eval_result({
            "age": 0.05,
            "income": 0.12,
            "balance": 0.08,
            "tenure": 0.03,
        })
        summary = ShapTool().format_for_llm(result)
        assert "income" in summary.top_3_summary
        assert "balance" in summary.top_3_summary
        assert "age" in summary.top_3_summary

    def test_top_3_summary_excludes_rank_4_feature(self):
        result = make_eval_result({
            "age": 0.05,
            "income": 0.12,
            "balance": 0.08,
            "tenure": 0.03,
        })
        summary = ShapTool().format_for_llm(result)
        assert "tenure" not in summary.top_3_summary

    def test_top_3_summary_starts_with_expected_prefix(self):
        result = make_eval_result({"a": 0.1, "b": 0.2, "c": 0.3})
        summary = ShapTool().format_for_llm(result)
        assert summary.top_3_summary.startswith("Top features by importance:")


class TestShapToolSingleFeature:
    def test_single_feature_returns_one_entry(self):
        result = make_eval_result({"only_feat": 0.09})
        summary = ShapTool().format_for_llm(result)
        assert len(summary.ranked_features) == 1

    def test_single_feature_rank_is_1(self):
        result = make_eval_result({"only_feat": 0.09})
        summary = ShapTool().format_for_llm(result)
        assert summary.ranked_features[0].rank == 1

    def test_single_feature_top_3_summary_contains_feature(self):
        result = make_eval_result({"only_feat": 0.09})
        summary = ShapTool().format_for_llm(result)
        assert "only_feat" in summary.top_3_summary


class TestShapToolEdgeCases:
    def test_two_features_top_3_summary_lists_both(self):
        result = make_eval_result({"feat_a": 0.10, "feat_b": 0.05})
        summary = ShapTool().format_for_llm(result)
        assert "feat_a" in summary.top_3_summary
        assert "feat_b" in summary.top_3_summary

    def test_two_features_does_not_raise(self):
        result = make_eval_result({"feat_a": 0.10, "feat_b": 0.05})
        ShapTool().format_for_llm(result)  # must not raise

    def test_tied_shap_values_do_not_raise(self):
        result = make_eval_result({"feat_a": 0.07, "feat_b": 0.07, "feat_c": 0.07})
        ShapTool().format_for_llm(result)  # must not raise

    def test_tied_shap_values_all_features_ranked(self):
        result = make_eval_result({"feat_a": 0.07, "feat_b": 0.07, "feat_c": 0.07})
        summary = ShapTool().format_for_llm(result)
        assert len(summary.ranked_features) == 3

    def test_all_zero_shap_values_assigns_ranks(self):
        result = make_eval_result({"feat_a": 0.0, "feat_b": 0.0, "feat_c": 0.0})
        summary = ShapTool().format_for_llm(result)
        ranks = sorted(e.rank for e in summary.ranked_features)
        assert ranks == [1, 2, 3]

    def test_ranked_features_length_equals_input_feature_count(self):
        shap_values = {f"feat_{i}": float(i) / 10 for i in range(6)}
        result = make_eval_result(shap_values)
        summary = ShapTool().format_for_llm(result)
        assert len(summary.ranked_features) == 6
