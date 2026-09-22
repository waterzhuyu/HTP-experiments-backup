import copy
import json
import pytest
from codebook_ablation.domain_summary import summarize,load_verified,SETTINGS,SEEDS,DOMAIN_METRICS,SUPPORT_SHA
from hierarchical_rqvae.preflight import sha256


def reports():
    return [{'setting':setting,'seed':seed,'status':'pair_completed','population':1024,
        'support_sha256':SUPPORT_SHA,'bootstrap_replicates':20000,'bootstrap_seed':271828,
        'source_runs':{cell:{'training_revision':'same'} for cell in ('B2','H1')},
        'comparisons':[{'metric':metric,'baseline_regret':.2,'candidate_regret':.3,
            'difference_regret':.1,'ci95':[.05,.15],'p_two_sided_centered_bootstrap':1.}
            for metric in DOMAIN_METRICS]} for setting in SETTINGS for seed in SEEDS]


def test_global_family_not_per_pair():
    data=reports();data[0]['comparisons'][0]['p_two_sided_centered_bootstrap']=.001
    result=summarize(data)
    assert result['status']=='secondary_analysis_ready'
    assert len(result['comparisons'])==120
    assert result['comparisons'][0]['holm_p_global_120']==pytest.approx(.12)
    assert result['descriptive_equal_seed_means'][0]['mean_difference_regret']==pytest.approx(.1)


def test_partial_data_does_not_issue_adjusted_results():
    result=summarize(reports()[:-1])
    assert result['status']=='incomplete' and result['verified_pairs']==14
    assert 'comparisons' not in result


def test_duplicate_pair_or_metric_and_mixed_revision_rejected():
    data=reports()
    with pytest.raises(ValueError,match='duplicate pair'):summarize(data+[data[0]])
    bad=copy.deepcopy(data);bad[0]['comparisons'][0]['metric']=DOMAIN_METRICS[1]
    with pytest.raises(ValueError,match='duplicate metric'):summarize(bad)
    bad=copy.deepcopy(data);bad[0]['source_runs']['B2']['training_revision']='different'
    with pytest.raises(ValueError,match='mixed training'):summarize(bad)


def test_tampered_report_rejected_before_source_loading(tmp_path):
    report=tmp_path/'report.json';report.write_text('{}')
    receipt={'status':'completed','files':{'report.json':sha256(report)}}
    (tmp_path/'artifact_manifest.json').write_text(json.dumps(receipt))
    report.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='artifact hash mismatch'):load_verified(report)


def test_old_or_mixed_bootstrap_protocol_rejected():
    data=reports()
    data[0]['bootstrap_replicates']=2000
    with pytest.raises(ValueError,match='protocol mismatch'):summarize(data)


def test_global_family_has_sufficient_p_resolution():
    data=reports()
    data[0]['comparisons'][0]['p_two_sided_centered_bootstrap']=1/20001
    result=summarize(data)
    assert result['comparisons'][0]['holm_p_global_120']==pytest.approx(120/20001)
    assert result['minimum_first_step_holm_p'] < .05
