import copy
import json
import pathlib
from datetime import datetime

try:
    import mlflow
except Exception:
    mlflow = None  # type: ignore

from agent.data_loader import DatasetLoader
from agent.leakage_detector import LeakageDetector
from agent.llm_reasoner import LLMReasoner
from tools.evaluate import EvaluateTool
from tools.execute import ExecuteTool
from tools.profile import ProfileTool
from tools.schemas import (
    AgentTrace,
    IterationRecord,
    ShapSummary,
    TaskType,
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
        task_type: str | None = None,
    ) -> AgentTrace:
        # 1. Load dataset
        loader = DatasetLoader()
        _, working_df = loader.load(dataset_path, target_col)
        try:
            detected = loader.detect_task_type(
                dataset_path, target_col,
                task_type=TaskType(task_type) if task_type is not None else None,
            )
            effective_task_type = detected if isinstance(detected, TaskType) else TaskType.classification
        except Exception:
            effective_task_type = TaskType.classification

        # MLflow step 1: start parent run
        try:
            mlflow.set_tracking_uri("./mlruns")
            mlflow.set_experiment("feature-agent")
            parent_run = mlflow.start_run(
                run_name=f"agent-{dataset_path}-{datetime.now().isoformat()}"
            )
        except Exception as e:
            print(f"[MLflow warning]: {e}")
            parent_run = None

        # 2. Baseline evaluation
        baseline_result = EvaluateTool().evaluate(working_df, target_col, effective_task_type)
        baseline_metric = baseline_result.primary_metric
        current_metric = baseline_metric
        current_shap = ShapTool().format_for_llm(baseline_result)
        profile = ProfileTool().profile(working_df, target_col)

        # MLflow step 2: log baseline
        try:
            if parent_run:
                with mlflow.start_run(run_name="baseline", nested=True):
                    mlflow.log_metric("baseline_auc_or_rmse", baseline_result.primary_metric)
                    mlflow.log_param("task_type", effective_task_type.value)
                    mlflow.log_param("feature_count", len(profile.feature_cols))
        except Exception as e:
            print(f"[MLflow warning]: {e}")

        # 3. Write baseline entry before iteration 1 (INV-05)
        baseline_entry = {
            "iteration": 0,
            "status": "baseline",
            "auc": baseline_metric,
            "f1": baseline_result.secondary_metric,
            "primary_metric": baseline_metric,
            "secondary_metric": baseline_result.secondary_metric,
            "task_type": effective_task_type.value,
            "features_used": baseline_result.feature_names,
        }
        trace_entries: list[dict] = [baseline_entry]
        _write_trace(trace_entries)

        print(f"Baseline {effective_task_type.value} metric: {baseline_metric:.4f}")

        reasoner = LLMReasoner()
        iteration_records: list[IterationRecord] = []
        small_delta_count = 0

        # 4. Iteration loop — INV-04: hard cap at 10 regardless of max_iter
        effective_max = min(max_iter, 10)
        for i in range(1, effective_max + 1):
            metric_before = current_metric

            # a. LLM reason
            reasoning = reasoner.reason(
                profile=profile,
                shap_summary=current_shap,
                iteration_history=iteration_records,
                current_features=[c for c in working_df.columns if c != target_col],
                task_type=effective_task_type,
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
                    auc_before=metric_before,
                    auc_after=metric_before,
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
                try:
                    if parent_run:
                        with mlflow.start_run(run_name=f"iteration-{i}", nested=True):
                            mlflow.log_metric("metric_before", record.auc_before)
                            mlflow.log_metric("metric_after", record.auc_after)
                            mlflow.log_metric("metric_delta", record.auc_delta)
                            mlflow.log_param("hypothesis", record.hypothesis[:250])
                            mlflow.log_param("feature_name", record.feature_name)
                            mlflow.log_param("decision", record.decision)
                            mlflow.log_text(record.transformation_code, f"code_iter_{i}.py")
                except Exception as e:
                    print(f"[MLflow warning]: {e}")
                small_delta_count += 1
                if small_delta_count >= EARLY_STOP_CONSECUTIVE:
                    print(f"Early stop: {EARLY_STOP_CONSECUTIVE} consecutive low-delta iterations")
                    break
                continue

            # d. Leakage check
            new_col_series = exec_result.output_df[reasoning.feature_name]
            target_series = exec_result.output_df[target_col]
            leak = LeakageDetector().is_leaking(
                new_col_series, target_series, reasoning.feature_name, target_col,
                task_type=effective_task_type,
            )

            if leak.is_leaking:
                record = IterationRecord(
                    iteration=i,
                    hypothesis=reasoning.hypothesis,
                    feature_name=reasoning.feature_name,
                    transformation_code=reasoning.transformation_code,
                    auc_before=metric_before,
                    auc_after=metric_before,
                    auc_delta=0.0,
                    shap_summary=_empty_shap_summary(),
                    decision="discarded",
                    error_message=leak.reason,
                    status="failed",
                )
                iteration_records.append(record)
                trace_entries.append(record.model_dump())
                _write_trace(trace_entries)
                print(f"Iteration {i}: leakage detected — {leak.reason}")
                try:
                    if parent_run:
                        with mlflow.start_run(run_name=f"iteration-{i}", nested=True):
                            mlflow.log_metric("metric_before", record.auc_before)
                            mlflow.log_metric("metric_after", record.auc_after)
                            mlflow.log_metric("metric_delta", record.auc_delta)
                            mlflow.log_param("hypothesis", record.hypothesis[:250])
                            mlflow.log_param("feature_name", record.feature_name)
                            mlflow.log_param("decision", record.decision)
                            mlflow.log_text(record.transformation_code, f"code_iter_{i}.py")
                except Exception as e:
                    print(f"[MLflow warning]: {e}")
                small_delta_count += 1
                if small_delta_count >= EARLY_STOP_CONSECUTIVE:
                    print(f"Early stop: {EARLY_STOP_CONSECUTIVE} consecutive low-delta iterations")
                    break
                continue

            # e. Evaluate with new feature
            candidate_df = exec_result.output_df
            new_eval = EvaluateTool().evaluate(candidate_df, target_col, effective_task_type)
            metric_after = new_eval.primary_metric
            metric_delta = metric_after - metric_before

            # f. SHAP summary for next iteration's LLM context
            new_shap = ShapTool().format_for_llm(new_eval)

            # g. Decide keep / discard
            # For regression: lower RMSE is better → keep if delta < 0
            # For classification: higher AUC is better → keep if delta > 0
            if effective_task_type == TaskType.regression:
                decision = "kept" if metric_delta < 0 else "discarded"
            else:
                decision = "kept" if metric_delta > 0 else "discarded"

            # h. Update working df if kept
            if decision == "kept":
                working_df = copy.deepcopy(candidate_df)
                current_metric = metric_after
                current_shap = new_shap
                profile = ProfileTool().profile(working_df, target_col)

            # i. Write IterationRecord atomically
            record = IterationRecord(
                iteration=i,
                hypothesis=reasoning.hypothesis,
                feature_name=reasoning.feature_name,
                transformation_code=reasoning.transformation_code,
                auc_before=metric_before,
                auc_after=metric_after,
                auc_delta=metric_delta,
                shap_summary=new_shap,
                decision=decision,
                error_message=None,
                status="completed",
            )
            iteration_records.append(record)
            trace_entries.append(record.model_dump())
            _write_trace(trace_entries)

            print(f"Iteration {i}: metric {metric_after:.4f} (delta {metric_delta:+.4f}) — {decision}")

            # MLflow step 3: log iteration
            try:
                if parent_run:
                    with mlflow.start_run(run_name=f"iteration-{i}", nested=True):
                        mlflow.log_metric("metric_before", record.auc_before)
                        mlflow.log_metric("metric_after", record.auc_after)
                        mlflow.log_metric("metric_delta", record.auc_delta)
                        mlflow.log_param("hypothesis", record.hypothesis[:250])
                        mlflow.log_param("feature_name", record.feature_name)
                        mlflow.log_param("decision", record.decision)
                        mlflow.log_text(record.transformation_code, f"code_iter_{i}.py")
            except Exception as e:
                print(f"[MLflow warning]: {e}")

            # j. Early stop check
            if abs(metric_delta) < EARLY_STOP_DELTA:
                small_delta_count += 1
            else:
                small_delta_count = 0

            if small_delta_count >= EARLY_STOP_CONSECUTIVE:
                print(f"Early stop: {EARLY_STOP_CONSECUTIVE} consecutive iterations with |delta| < {EARLY_STOP_DELTA}")
                break

        final_features = [c for c in working_df.columns if c != target_col]

        trace = AgentTrace(
            baseline_metric=baseline_metric,
            iterations=iteration_records,
            final_feature_set=final_features,
            final_metric=current_metric,
            task_type=effective_task_type,
        )

        # MLflow step 4: log final metrics and end run
        try:
            if parent_run:
                mlflow.log_metric("final_metric", trace.final_metric)
                mlflow.log_metric("total_lift", trace.final_metric - trace.baseline_metric)
                mlflow.log_metric("iterations_run", len(trace.iterations))
                mlflow.end_run()
        except Exception as e:
            print(f"[MLflow warning]: {e}")

        return trace
