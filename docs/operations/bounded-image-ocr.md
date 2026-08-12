# Bounded image OCR

Paid image OCR is available only through the explicit, manifest-driven command:

```bash
gbrain image-ocr-run ./image-ocr-manifest.jsonl \
  --max-images 100 \
  --max-usd 1 \
  --reserve-usd-per-call 0.01 \
  --yes
```

`GBRAIN_EMBEDDING_IMAGE_OCR=true` is not authorization to call an OCR provider.
Routine import and sync paths continue importing images and multimodal embeddings,
but their importer contains no paid OCR provider call. The only OCR provider
boundary is private to this confirmed command. There is no env-only override.

## Manifest contract

The input is deterministic JSONL with exactly four fields per line:

```json
{"source_id":"photos","slug":"images/example.png","file_path":"/absolute/source/root/images/example.png","sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}
```

Before any provider access, the command validates the complete manifest: unique
entries, registered source and local root, canonical source-relative slug, absolute
contained non-symlink file, supported image extension, and exact SHA-256. The
registered root itself may not be a symlink; preflight freezes its canonical
realpath and device/inode identity plus the image file's device/inode identity.
Any mismatch rejects the whole run before the first reservation or provider call.

The actual magic, structure, single-frame status, dimensions, and decoded-pixel
bound are validated for PNG, JPEG, GIF, WebP, HEIC/HEIF, and AVIF. Zero-byte,
malformed, animated/multiframe, zero-dimension, over-8,000-pixel-dimension, and
over-25,000,000-decoded-pixel inputs fail before reservation. Strict container
inspection bounds dimensions and frame count before any decode allocation. PNG,
HEIC/HEIF, and AVIF then use their installed decoders; HEIC frame handles and
AVIF `ispe` metadata are checked before an RGBA decode can be requested. On
macOS, JPEG/GIF/WebP are fully decoded through ImageIO using `/usr/bin/sips`
without a shell, inside a private temporary directory with no-follow file opens,
strict process/output checks, and guaranteed cleanup. The resulting PNG is
structure-checked and fully decoded again, and its dimensions/frame count must
match the bounded source metadata. JPEG/GIF/WebP, HEIC/HEIF, and AVIF provider
payloads are therefore re-encoded to validated PNG. If the required decoder is
unavailable on a platform, the paid lane rejects the codec rather than falling
back to metadata-only acceptance.

After the durable reservation, the command repeats byte, SHA, file identity,
full compressed decode, PNG re-encoding, format, frame, dimension, pixel, token,
source-root, and source-scoped page-state validation from a no-follow file
descriptor. It builds the wire body from that validated snapshot. There is no
await point between the final synchronous file/body check and transport dispatch.
Any drift consumes the conservative reservation but makes zero transport
attempts. A named cross-process image-import
fence is held from final revalidation through provider access and OCR persistence;
routine image imports acquire the same fence. Already-imported exact hashes make
zero provider calls. Files larger than 20 MB, or whose exact serialized UTF-8
Anthropic JSON request body exceeds 10 MiB, are rejected during full-manifest
preflight and checked again at the private provider boundary.

## Budget and failure semantics

All three caps and `--yes` are mandatory. Code policy refuses values above 1,000
images or $10, and refuses a reserve below $0.01 per call. Changing those outer
ceilings requires an intentional code review.

The only paid transport is the unexported `src/core/image-ocr-provider.ts` module.
It constructs one exact JSON request for
`https://api.anthropic.com/v1/messages`, pins model
`claude-haiku-4-5-20251001`, `max_tokens: 1024`, one fixed short OCR prompt,
and performs one `fetch` attempt with no retry loop. Both
`ANTHROPIC_BASE_URL` and a configured Anthropic provider base URL are rejected;
the lane never inherits expansion-model or gateway endpoint configuration.
Anthropic documents Haiku 4.5 standard-resolution vision as scaling within a
1,568-pixel long edge and about 1.15 megapixels, at approximately one visual
token per 750 scaled pixels, with an 8,000-pixel request dimension limit. The
implementation rounds scaled dimensions and tokens upward, calculates every
entry's visual-token/cost bound, and enforces the 25-megapixel local decode
ceiling before allocation. The global reservation proof retains the stricter
1,568-visual-token ceiling, adds 500 input tokens for fixed prompt/request
overhead, and derives the named model's $1/MTok input and $5/MTok output prices
from the canonical pricing table: `(2,068 × $1/MTok) + (1,024 × $5/MTok) =
$0.007188`, leaving $0.002812 inside each $0.01 reservation. A manifest entry is
rejected if its computed worst case exceeds the requested per-call reserve. The
pricing and image-rule snapshot is pinned to this named model; a pricing, model,
or provider image-rule change requires policy/test review. Provider credits or a
workspace billing limit are an outer circuit breaker, not a substitute for the
local call/spend controls.
Source: https://platform.claude.com/docs/en/build-with-claude/vision

The ledger lives under `~/.gbrain/ocr-budget/`. Its fixed UTC-date lock is an
atomic directory containing a 256-bit random owner-token record with PID, parent
PID, process start, hostname, and acquisition metadata. Open directory/file
device-inode identities and the token are verified before every reservation,
audit transition, and release. Release removes only that token-named owner file
and never recursively removes a replacement lock. The shared image-import fence
is acquired first. After any fence wait, the command samples
a fresh clock value, acquires that UTC date's exclusive ledger lock, and durably
reserves the call and dollars there. It then performs the complete post-reservation
revalidation described above and durably records `transport_attempted` immediately
before invoking transport. A fence wait crossing midnight therefore charges the
call to its actual attempt date; independent 1,000-image and $10 ceilings apply
to every UTC day. Later runs may tighten a date's configured limits but can never
loosen them. A crash
after reservation over-counts safely. An existing, stale, or ambiguous fence or
ledger lock is never broken automatically; the run fails closed for operator
review and requires manual recovery after confirming no holder remains. Ledger
schema 2 gives every reservation a random ID, its exact reserved USD and
request-specific maximum input tokens, and a durable lifecycle state: `reserved`,
`transport_attempted`, `receipt_validated`, `invalid_provider_observation`,
`persisted`, or `failed`, with explicit pending/failed/ambiguous/invalid/
over-limit/persisted outcome, failure stage, receipt or sanitized invalid
observation, usage/cost, and persistence fields. A crash leaves its last
conservative state in place and never refunds or reopens budget.

Provider success is narrower than HTTP success. The response must be a Messages
`message` for exactly `claude-haiku-4-5-20251001`, have a nonempty request ID,
exactly one nonempty text block, `stop_reason: end_turn`, and finite nonnegative
integer input, cache-creation, cache-read, and output usage fields. Input usage
may not exceed that image's validated visual tokens plus the pinned 500-token
nonvisual allowance; output usage may not exceed the request's 1,024-token cap;
and canonical observed USD may not exceed that attempt's exact durable
reservation. `max_tokens`, missing/invalid usage, model/type drift, multiple
outputs, empty text, and any envelope violation are rejected without persistence.
This fixed request does not enable prompt caching; until cache-tier pricing is
deliberately added to this pinned contract, nonzero cache usage is treated as
unreconcilable and fails closed. An otherwise well-formed response that exceeds
the input/output/cost envelope becomes a sanitized durable
`invalid_provider_observation`: request ID, pinned model, stop reason, usage,
computed USD, and the explicit over-limit reason are retained, but OCR text and
the response body are not. Reports include that observed usage/cost while valid
receipt and persisted-import counters remain zero, and the run stops immediately.
Admission continues using the conservative reservation.

At either cap, the next entry stops before provider or database access. HTTP
redirects are rejected rather than followed, preserving the canonical endpoint
and one-request contract. Provider,
decode, embedding, or import failure stops the lane immediately. The reservation
remains charged, no filename OCR fallback is written, and no sync bookmark is
read or changed. Raw argv is validated before global parsing: only `--brain`,
`--help`, `--max-images`, `--max-usd`, `--reserve-usd-per-call`, `--yes`, and
one manifest positional are legal. Global output/timeout/source flags and every
unknown flag are rejected rather than stripped or ignored.

Every terminal path—including early flag rejection, thin-client rejection,
engine connection failure, manifest/lock rejection, provider failure, and ledger
close failure—writes exactly one machine-readable JSON report to stdout. Reports
separate reservations, provider attempts, valid provider receipts, observed
input/cache/output tokens and USD, persisted imports, and failures. Reports never
include OCR text, credentials, or provider response bodies, and preserve
completed/reserved counters when later cleanup fails; unreadable locked-ledger
values are represented as `null`.

Immediately before transport, the command also freezes the target page's exact
write token: existence plus row ID, content generation, update timestamp, hash,
and deletion state. Persistence enforces that token inside the same database
transaction. An existing row is locked only if every token field still matches;
an absent row is claimed with a conflict-free conditional insert. If a generic
writer changed or created the target during provider latency, the paid receipt
remains conservatively accounted but OCR text does not overwrite that writer.

The bounded OCR import stores OCR text without invoking multimodal embedding, so
the command's dollar ledger covers its only provider request. Embedding remains a
separate operation with its own provider and budget controls.

## Exact-hash donor adoption (zero provider)

Images whose source-local target does not exist can reuse trustworthy derived
OCR/vector data from an active image page with the exact same physical SHA-256:

```bash
gbrain image-donor-adopt ./image-donor-manifest.jsonl \
  --max-images 100 \
  --yes
```

This is a separate storage-only command. It does not configure the AI gateway,
construct an OCR request, read or reserve the OCR budget, call a model, or fall
through to `image-ocr-run`. Its terminal JSON reports all provider, model,
gateway, budget-reservation, and budget-USD counters as zero. It accepts only
`--brain`, `--help`, `--max-images`, `--yes`, and one manifest positional;
`--max-images` is mandatory and limited to 1 through 1,000. The entire manifest
is preflighted before the first row transaction, including entries beyond the
cap. A missing, malformed, or pre-v125 schema-version marker fails closed with
`schema_incompatible`; this command never runs migrations.

The donor command reuses the same exact-four-key JSONL, canonical registered
root, containment, non-symlink, stable device/inode/size/mtime/ctime/SHA, image
format, full-decode, frame, dimension, and pixel checks described above. It does
not build the paid provider wire body. The registered source root and the exact
file are checked again inside the shared image-import fence at both transaction
boundaries.

A qualifying donor is selected deterministically by page ID then chunk ID and
is rebound under lock before copying. The page must be active,
`page_kind='image'`, have the exact physical hash, and have exactly one chunk.
That chunk must be index 0, `chunk_source='image_asset'`, `modality='image'`,
contain a non-null text embedding with a nonblank model, and have text exactly
equal to the page's `compiled_truth`. OCR text must be at least 120 characters,
nonblank, and contain a letter or number. A non-null embedding signature must
equal the chunk model plus the actual text-vector dimensions. Honest legacy
NULLs for token count, embedded timestamp, signature, image vector, and
multimodal vector remain NULL. Contextual vectors are reusable only when
`contextual_retrieval_mode` is NULL/`none` and `corpus_generation` is NULL;
title- or synopsis-conditioned vectors are rejected because they encode donor
ownership context.

Each attempted row runs in one transaction inside the fence. Any active or
soft-deleted `(source_id, slug)` target is a conflict unless it is the exact,
unchanged result of the same manifest adoption. A successful adoption creates
fresh target page/chunk IDs and target-owned source/path/file metadata, copies
only OCR and vector-derived values, creates no links, and stores versioned donor
IDs plus image/OCR, donor-state, and text/image/multimodal-vector digests in
namespaced target frontmatter. Exact replay checks those original digests, so a
later coordinated re-embedding of donor and target is drift rather than a new
idempotency baseline. The donor is never updated. Postconditions re-read the
donor, target graph, and file row before commit.

`files.storage_path` is globally unique, so the command handles it without the
normal source-scoped upsert: absence creates a target-owned file row; an existing
same-hash row is preserved byte-for-byte and reported as
`preserved_existing`; an existing different-hash row aborts the row transaction.
The deterministic report schema remains in manifest order, includes no wall
clock, and binds run-specific database identities/digests plus the manifest hash,
aggregate counts, first stable failure code, donor/file dispositions, and a
full digest-only CAS receipt for every adopted or idempotent row. It never emits
OCR text, vectors, credentials, provider configuration, or raw exception text.

The receipt-backed rollback helper is deliberately compare-and-delete rather
than a general migration framework. It refuses any target/chunk/file drift,
including a new file-migration-ledger child, deletes a file only when that
receipt created it, then deletes the target page (the chunk cascades), and never
changes the donor or a pre-existing file row.
Database sequences and the page-generation clock are monotonic and are not
claimed to return to their prior values. There is no public bulk rollback CLI,
scheduler integration, source rewrite, link reconciliation, or legacy-row
retirement in this v0 path.

## Scope boundary

This lane processes only manifest entries. It does not run sync, full scans,
retry-failed, extraction, Dream, jobs, or all-source discovery beyond validating
the source IDs named in the manifest.

Use paid `image-ocr-run` only when no qualifying exact-hash donor exists. Donor
adoption never serves as authorization for paid OCR and cannot transition into
the paid lane on failure.
