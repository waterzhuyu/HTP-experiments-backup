# GPU watch-and-launch runbook

## Purpose

`staged_attribution.watch_gpu_and_launch` is a server-side watcher for the formal
GPS/Road/Grid staged-attribution queue. It polls the four GPUs from the execution
host and delegates to the existing append-only parallel runner as soon as at least
one GPU satisfies both gates:

- free memory is at least 16,000 MiB;
- utilization is at most 20 percent.

The default polling interval is 15 seconds. GPU ownership is never changed and no
other user's process is signalled.

## Mandatory launch contract

The watcher refuses to run unless all of the following remain true:

1. execution branch and HEAD equal the command-line contract;
2. the worktree is clean;
3. the formal config SHA256 equals the command-line contract;
4. `provenance.stage2_resource_option` explicitly records A, B, or C;
5. `candidate_only` is not true;
6. Stage2 population, microbatch, accumulation, and manifests match that option;
7. the artifact root basename equals the experiment ID and remains outside the Git
   worktree;
8. no same-user formal runner or cell process is already active.

The resource options are frozen as follows:

- A: full Stage2 population, microbatch 8, accumulation 2;
- B: 64,000 trajectories per city, microbatch 8, accumulation 2, with explicit
  immutable Porto and Chengdu subset manifests;
- C: full Stage2 population, microbatch 16, accumulation 1.

The contract is revalidated before every GPU query. A drift aborts the watcher
before it creates another attempt. A non-blocking file lock prevents two watchers
from launching the same queue. Existing completed attempts are skipped; failed
attempts are retained and the next numbered directory is used. Before delegation,
the watcher also inspects only the current user's process list so an orphaned cell
cannot be duplicated; it never signals another user's process.

## Validate, then arm

After the user selects A, B, or C and the one formal config is committed, first run
the command with `--validate-only`. Use the committed HEAD and config SHA256 from
that exact revision. Validation must not create an attempt directory.

Then run the same command without `--validate-only` under a persistent process
manager. Required arguments are:

```text
--worktree /workspace/guozuyu/HTP-staged-attribution-integrated
--config /workspace/guozuyu/HTP-staged-attribution-integrated/stage_1/configs/staged_attribution/experiment.json
--artifact-root /home/guozuyu/HTP-staged-attribution-artifacts/staged_attribution/gps_road_grid_staged_attribution_v1_20260826
--expected-branch exp/staged-attribution-gps-road-grid-integrated
--expected-head <committed-head>
--expected-config-sha256 <formal-config-sha256>
--resource-option <A-or-B-or-C>
--budget controlled
--maximum-parallel 4
--poll-seconds 15
--minimum-free-mib 16000
--maximum-utilization 20
