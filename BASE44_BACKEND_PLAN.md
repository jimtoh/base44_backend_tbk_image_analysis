# Base44 Analysis Backend Plan

## Purpose and current status

This repository contains the authoritative Python image-processing pipeline and a
single-process FastAPI service in `image_analysis/base44_service.py`. The service
wraps the original algorithms; it does not translate them into Base44 functions,
which would create a different matcher. It uses a filesystem-backed job queue and
vault store and is designed for one service instance with persistent storage.
Multiple replicas require a shared durable queue/store and are not implemented.

Keep `image_analysis/painting_crop_v8.py` and
`image_analysis/golden_match_v5.py` as the algorithm sources of truth. Service work
should wrap their callable functions without changing crop logic, feature
extraction, thresholds, or scoring. The current callable boundaries are:

- `crop_painting(input_path, output_path, fmt, predictor, manual_corners=None)`
- `build_predictor(model_path)`
- `build_vault(golden_dir, vault_dir, progress=True)`
- `load_vault(vault_dir)`
- `search_shot(data, indexes, candidate_path)`

The cropper accepts manual corners in input-image pixels, ordered top-left,
top-right, bottom-right, bottom-left. It validates the quadrilateral and skips SAM
when manual corners are supplied. `search()` reloads the vault on every call, so a
service should cache the loaded vault and indexes rather than call it per job.

## Service API contract

All routes below use JSON except image artifacts returned as files or URLs. Require
`Authorization: Bearer <service-token>` on every route except
`GET /health/live`. Compare tokens in constant time. Never expose the token to a
browser or include it in logs.

### `GET /health/live` and `GET /health/ready`

`/health/live` reports whether the process is running. `/health/ready` reports
readiness only when the SAM checkpoint and a complete vault revision are available.
Return HTTP 503 when not ready. The ready response includes service, crop, matcher,
model/checksum, config, and active vault revision identifiers; do not return
filesystem paths or credentials.

### `PUT /v1/vault`

Synchronize a complete desired vault manifest. Require a caller-supplied revision
and a list of stable reference IDs, image URLs, and SHA-256 hashes:

```json
{
  "revision": "vault-2026-10-25-a1b2c3",
  "references": [
    {
      "id": "reference-123",
      "image_url": "https://temporary-private-image-url",
      "sha256": "64-character-lowercase-hex-digest"
    }
  ]
}
```

Download and verify images, then build the index in a new revision directory. Make
the new revision active only after the build succeeds; keep the prior revision
available for already accepted jobs. An identical manifest/hash set must not cause
a rebuild. Persist source images, generated index files, and the ID-to-image mapping
on durable storage. Map matcher paths back to stable reference IDs in API results.

Only allow HTTPS URLs from configured storage hosts. Reject redirects to other
hosts, private/loopback/link-local IPs, oversized responses, unsupported image
formats, and hash mismatches. Prefer a storage-native authenticated download or
direct upload if Base44 can provide one; do not accept arbitrary URLs.

### `POST /v1/jobs`

Accept 1-10 client photos and an optional vault revision. Persist the request and
return HTTP 202 with a job ID; run analysis in a background worker. A photo has a
stable ID, a temporary private image URL, and optionally manually adjusted corners:

```json
{
  "vault_revision": "vault-2026-10-25-a1b2c3",
  "photos": [
    {
      "id": "submission-photo-1",
      "image_url": "https://temporary-private-image-url",
      "corners": [[120, 80], [980, 100], [950, 1260], [90, 1220]]
    }
  ]
}
```

`corners` is optional and must be in the uploaded image's orientation-corrected,
full-resolution pixel coordinates in TL, TR, BR, BL order. Preserve the exact
coordinates Base44 submitted. If present, pass them to `crop_painting` as
`manual_corners`; do not run SAM for that photo. Reject malformed, out-of-bounds,
or non-convex quadrilaterals as input errors.

Download and validate images before processing. Apply request, file-count, pixel
dimension, and byte-size limits. Orient images consistently before corner handling;
the coordinate space in this contract is the oriented image presented to the crop
step. Capture the submitted hash and algorithm/config versions with each job.

### `GET /v1/jobs/{job_id}`

Return HTTP 200 for any known job, including a failed job. Use `queued`, `running`,
`succeeded`, or `failed` for `status`; include a progress message and structured
error for failures. Return 404 for an unknown job. A job remains pinned to the
vault revision named at submission (or the active revision when omitted), even if
the active vault later changes.

Example completed response:

```json
{
  "job_id": "job-uuid",
  "status": "succeeded",
  "vault_revision": "vault-2026-10-25-a1b2c3",
  "versions": {
    "crop": "painting_crop_v8",
    "matcher": "golden_match_v5",
    "model": "sam_vit_b_01ec64.pth",
    "config": "sha256:..."
  },
  "result": {
    "status": "match",
    "matched_reference_id": "reference-123",
    "photos": [
      {
        "photo_id": "submission-photo-1",
        "crop": {
          "method": "manual",
          "corners": [[120, 80], [980, 100], [950, 1260], [90, 1220]],
          "output_size": {"width": 860, "height": 1180},
          "preview_url": "https://analysis.example.com/v1/jobs/job-uuid/artifacts/submission-photo-1.jpg"
        },
        "match": {
          "matched_reference_id": "reference-123",
          "inliers": 82,
          "runner_up_inliers": 27,
          "channel_inliers": {
            "grey": 82,
            "red": 30,
            "green": 25,
            "blue": 28,
            "chroma": 19
          },
          "template_score_percent": 74.2
        }
      }
    ]
  }
}
```

The schema above is illustrative: implement field names consistently and document
any necessary differences before Base44 integration. `result.status` is `match`,
`no_match`, or `needs_corners`. A crop/read/processing failure must never be
translated to `no_match`. For `needs_corners`, return the photo ID and a stable
error code so Base44 can ask for corner adjustment and resubmit. Template score is
display-only similarity, not confidence or an authenticity verdict.

Preview routes are protected by the same bearer token. Configure `BASE44_PUBLIC_URL`
to return absolute URLs; without it, the API returns a service-relative artifact
path. Base44 must fetch previews from a trusted server-side context, not expose the
service token to a browser.

The top-level match must preserve the matcher rule: a golden wins only if it passes
the existing per-shot thresholds and reaches consensus on at least half of the
submitted shots. Preserve each shot's top reference, inlier count, runner-up count,
per-channel counts, and template score. Keep the reference ID separate from its
private storage path.

## Runtime and persistence requirements

- Load the SAM predictor once per worker process; use CPU by default unless a
  tested deployment explicitly provides compatible GPU support.
- Load the active vault's NPZ and FLANN indexes once and cache them. Swap revisions
  atomically and retain old revisions while jobs reference them.
- The current implementation persists job state to files and recovers queued/running
  jobs after a process restart. Run exactly one service process against each data
  directory. For multiple replicas, replace the local queue and file store with a
  shared durable queue and job/result store; do not mount one local queue directory
  into several concurrently running service instances.
- Store uploaded/downloaded source images and crop artifacts in private durable
  storage. Return short-lived artifact URLs; never expose local paths.
- Configure concurrency to fit model memory and CPU limits. Avoid parallel SAM
  inference in one model instance unless the chosen runtime is verified thread-safe.
- Pin Python and dependency versions, including the Segment Anything source
  revision. Record and verify the SAM checkpoint SHA-256 before startup.
- Package the service in a Docker image and document persistent volume mounts,
  required secrets, health probes, resource limits, and a deployment target capable
  of running PyTorch, OpenCV, and SAM.

## Acceptance checks before Base44 integration

1. API manual corners reach `crop_painting` unchanged and bypass SAM. Compare crop
  pixels with a direct `crop_painting` call on fixed fixtures before claiming
  equivalence.
2. Matching a fixed crop against a fixed vault returns the same result as
   `search_shot` for the same files and existing thresholds.
3. A multi-photo job returns the same consensus match as `golden_match_v5.search`.
4. Missing/invalid image, crop failure, unavailable vault, and worker exception
   produce explicit failures, never a successful no-match response.
5. Unauthorized requests, disallowed download hosts, redirect escapes, byte/pixel
   limit violations, and image hash mismatches are rejected.
6. Restarting the service recovers accepted jobs and the active vault; updating the
  vault does not alter results for jobs pinned to the prior revision.

The local CLI remains the behavior reference while implementing and verifying the
service. Do not claim API/CLI equivalence until these checks pass on fixed fixtures.