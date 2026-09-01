from __future__ import annotations

from pathlib import Path

from staged_attribution.config import ExperimentConfig
from staged_attribution.run_cell import build_cell_dag
from staged_attribution.run_experiment import build_jobs
from staged_attribution.run_stratified_reevaluation import build_reevaluation_dag


CONFIG_PATH = Path("stage_1/configs/staged_attribution/experiment.json")


def test_integrated_training_and_corrected_evaluation_contract() -> None:
    config = ExperimentConfig.load(CONFIG_PATH)
    controlled = config.budgets["controlled"]

    assert controlled.stage2_train_trajectories == 64_000
    assert controlled.stage2_subset_seed == 20260826
    assert controlled.stage2_batch_size == 8
    assert controlled.stage2_gradient_accumulation == 2
    assert controlled.stage2_effective_batch_size == 16
    assert set(controlled.stage2_subset_manifests) == {"porto", "chengdu"}
    assert controlled.stage2_subset_manifests["porto"].endswith(
        "porto_stage2_train_64000_seed20260826.json"
    )
    assert controlled.stage2_subset_manifests["chengdu"].endswith(
        "chengdu_stage2_train_64000_seed20260826.json"
    )
    assert config.provenance["stage2_resource_option"] == "B"
    decision = config.provenance["stage2_resource_decision"]
    assert decision["stage2_subset_index_sha256"] == (
        "8673fd0992d5f41af0ba0d810db6770edc8e4f1624d6a76cb984480fa5524bbd"
    )
    assert decision["porto_manifest_sha256"] == (
        "026720480bfed13f9f98aa4f34611a3a0ce06b98af03e712ca60ef3f13153942"
    )
    assert decision["chengdu_manifest_sha256"] == (
        "ba7d822c6ec0dbfe46f88d5653b127017f62918b92c81090993a2e8251acf8e4"
    )
    assert len(build_jobs(config, "controlled")) == 60

    adaptive = config.cell("e3_q2")
    assert adaptive.views == frozenset({"gps", "road"})
    assert adaptive.q_weights == config.q_variants["q2"]

    training_dag = build_cell_dag("e3_q2")
    assert [node.name for node in training_dag] == [
        "stage1",
        "tokens",
        "stage2_data",
        "stage2",
        "stage3",
        "g_attribution",
        "decode",
        "stage4",
    ]
    assert "stage2_plan.json" in training_dag[3].required_artifacts
    assert set(training_dag[-1].required_artifacts) == {
        "stage4_metrics.json",
        "g00_contributions_manifest.json",
        "g10_contributions_manifest.json",
        "g01_contributions_manifest.json",
        "g11_contributions_manifest.json",
    }

    reevaluation_dag = build_reevaluation_dag()
    assert [node.name for node in reevaluation_dag] == [
        "source_reference",
        "stage3",
        "g_attribution",
        "decode",
        "stage4",
    ]
    assert set(reevaluation_dag[-1].required_artifacts) == {
        "stage4_metrics.json",
        "g00_contributions_manifest.json",
        "g10_contributions_manifest.json",
        "g01_contributions_manifest.json",
        "g11_contributions_manifest.json",
    }
