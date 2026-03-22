import copy
import json
import pathlib

from agent.data_loader import DatasetLoader
from agent.llm_reasoner import LLMReasoner
from tools.evaluate import EvaluateTool
from tools.execute import ExecuteTool
from tools.profile import ProfileTool
from tools.schemas import (
    AgentTrace,
    IterationRecord,
    ShapSummary,
)
from tools.shap_tool import ShapTool

OUTPUTS_DIR = pathlib.Path("outputs")
EARLY_STOP_DELTA = 0.001
EARLY_STOP_CONSECUTIVE = 2


def _write_trace(entries: list[dict]) -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    tmp = OUTPUTS_DIR / "trace.tmp.json"
    tmp.write_text(json.dumps(entries, indent=2, default=str))
    tmp.replace(OUTPUTS_DIR / "trace.json")


def _empty_shap_summary() -> ShapSummary:
    return ShapSummary(ranked_features=[], top_3_summary="No SHAP data available.")


class AgentLoop:
    def run(
        self,
        dataset_path: str,
        target_col: str,
        max_iter: int = 5,
    ) -> AgentTrace:
        # 1. Load dataset
        _, working_df = DatasetLoader().load(dataset_path, target_col)

        # 2. Baseline evaluation
        baseline_result = EvaluateTool().evaluate(working_df, target_col)
        baseline_auc = baseline_result.auc
        current_auc = baseline_auc
        current_shap = ShapTool().format_for_llm(baseline_result)
        profile = ProfileTool().profile(working_df, target_col)

        # 3. Write baseline entry before iteration 1 (INV-05)
        baseline_entry = {
            "iteration": 0,
            "status": "baseline",
            "auc": baseline_auc,
            "f1": baseline_result.f1,
            "features_used": baseline_result.feature_names,
        }
        trace_entries: list[dict] = [baseline_entry]
        _write_trace(trace_entries)

        print(f"Baseline AUC: {baseline_auc:.4f}")

        reasoner = LLMReasoner()
        iteration_records: list[IterationRecord] = []
        small_delta_count = 0

        # 4. Iteration loop — INV-04: hard cap at 10 regardless of max_iter
        effective_max = min(max_iter, 10)
        for i in range(1, effective_max + 1):
            auc_before = current_auc

            # a. LLM reason
            reasoning = reasoner.reason(
                profile=profile,
                shap_summary=current_shap,
                iteration_history=iteration_records,
                current_features=[c for c in working_df.columns if c != target_col],
            )

            # b. Execute transformation in sandbox
            exec_result = ExecuteTool().execute(working_df, reasoning.transformation_code)

            # c. Execute failed → log error record and continue
            if not exec_result.success:
                record = IterationRecord(
                    iteration=i,
                    hypothesis=reasoning.hypothesis,
                    feature_name=reasoning.feature_name,
                    transformation_code=reasoning.transformation_code,
                    auc_before=auc_before,
                    auc_after=auc_before,
                    auc_delta=0.0,
                    shap_summary=_empty_shap_summary(),
                    decision="error",
                    error_message=exec_result.error_message,
                    status="failed",
                )
                iteration_records.append(record)
                trace_entries.append(record.model_dump())
                _write_trace(trace_entries)
                print(f"Iteration {i}: execute error — {exec_result.error_message}")
                small_delta_count += 1
                if small_delta_count >= EARLY_STOP_CONSECUTIVE:
                    print(f"Early stop: {EARLY_STOP_CONSECUTIVE} consecutive low-delta iterations")
                    break
                continue

            # d. Leakage check stub — always returns False (no leakage)
            # TODO: replace with real leakage detector in Task 3.1
            is_leaking = False

            if is_leaking:
                record = IterationRecord(
                    iteration=i,
                    hypothesis=reasoning.hypothesis,
                    feature_name=reasoning.feature_name,
                    transformation_code=reasoning.transformation_code,
                    auc_before=auc_before,
                    auc_after=auc_before,
                    auc_delta=0.0,
                    shap_summary=_empty_shap_summary(),
                    decision="error",
                    error_message="Leakage detected — feature encodes target.",
                    status="failed",
                )
                iteration_records.append(record)
                trace_entries.append(record.model_dump())
                _write_trace(trace_entries)
                small_delta_count += 1
                if small_delta_count >= EARLY_STOP_CONSECUTIVE:
                    print(f"Early stop: {EARLY_STOP_CONSECUTIVE} consecutive low-delta iterations")
                    break
                continue

            # e. Evaluate with new feature
            candidate_df = exec_result.output_df
            new_eval = EvaluateTool().evaluate(candidate_df, target_col)
            auc_after = new_eval.auc
            auc_delta = auc_after - auc_before

            # f. SHAP summary for next iteration's LLM context
            new_shap = ShapTool().format_for_llm(new_eval)

            # g. Decide keep / discard
            decision = "kept" if auc_delta > 0 else "discarded"

            # h. Update working df if kept
            if decision == "kept":
                working_df = copy.deepcopy(candidate_df)
                current_auc = auc_after
                current_shap = new_shap
                profile = ProfileTool().profile(working_df, target_col)

            # i. Write IterationRecord atomically
            record = IterationRecord(
                iteration=i,
                hypothesis=reasoning.hypothesis,
                feature_name=reasoning.feature_name,
                transformation_code=reasoning.transformation_code,
                auc_before=auc_before,
                auc_after=auc_after,
                auc_delta=auc_delta,
                shap_summary=new_shap,
                decision=decision,
                error_message=None,
                status="completed",
            )
            iteration_records.append(record)
            trace_entries.append(record.model_dump())
            _write_trace(trace_entries)

            print(f"Iteration {i}: AUC {auc_after:.4f} (delta {auc_delta:+.4f}) — {decision}")

            # j. Early stop check
            if abs(auc_delta) < EARLY_STOP_DELTA:
                small_delta_count += 1
            else:
                small_delta_count = 0

            if small_delta_count >= EARLY_STOP_CONSECUTIVE:
                print(f"Early stop: {EARLY_STOP_CONSECUTIVE} consecutive iterations with |delta| < {EARLY_STOP_DELTA}")
                break

        final_features = [c for c in working_df.columns if c != target_col]

        return AgentTrace(
            baseline_auc=baseline_auc,
            iterations=iteration_records,
            final_feature_set=final_features,
            final_auc=current_auc,
        )
