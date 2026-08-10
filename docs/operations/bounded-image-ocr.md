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
contained non-symlink file, supported image extension, and exact SHA-256. Any
mismatch rejects the whole run before the first reservation or provider call.
The registered root, canonical relative path, current source-scoped page state,
and the hash of the bytes actually sent are checked again immediately before
provider dispatch. A named cross-process image-import fence is held from that
final revalidation through provider access and OCR persistence; routine image
imports acquire the same fence, so they cannot race a paid result into the
normal same-hash skip path. Already-imported exact hashes make zero provider
calls. Files larger than 20 MB, or whose exact serialized UTF-8 Anthropic JSON
request body exceeds 10 MiB, are rejected during full-manifest preflight before
reservation and checked again at the private provider boundary.

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
Anthropic documents Haiku 4.5 standard-resolution vision at no more than 1,568
visual input tokens. The coded cost proof adds a conservative 500 input tokens
for the fixed prompt/request overhead and uses the current official prices of
$1/MTok input and $5/MTok output: `(2,068 × $1/MTok) + (1,024 × $5/MTok) =
$0.007188`, leaving $0.002812 inside each $0.01 reservation. The proof is pinned
to this named model's current official price contract; a pricing or model change
requires policy/test review. Provider credits or a workspace billing limit are
an outer circuit breaker, not a substitute for the local call/spend controls.
Source: https://platform.claude.com/docs/en/build-with-claude/vision

The ledger lives under `~/.gbrain/ocr-budget/`. The image-import fence is
acquired first. After any wait and final revalidation, immediately before every
provider attempt, the command samples a fresh clock value, acquires that UTC
date's exclusive ledger lock, and durably reserves the call and dollars there.
A fence wait crossing midnight therefore charges the call to its actual attempt
date; independent 1,000-image and $10 ceilings apply to every UTC day. Later
runs may tighten a date's configured limits but can never loosen them. A crash
after reservation over-counts safely. An existing, stale, or ambiguous fence or
ledger lock is never broken automatically; the run fails closed for operator
review and requires manual recovery after confirming no holder remains.

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
never include provider response bodies and preserve completed/reserved counters
when later cleanup fails; unreadable locked-ledger values are represented as
`null`.

The bounded OCR import stores OCR text without invoking multimodal embedding, so
the command's dollar ledger covers its only provider request. Embedding remains a
separate operation with its own provider and budget controls.

## Scope boundary

This lane processes only manifest entries. It does not run sync, full scans,
retry-failed, extraction, Dream, jobs, or all-source discovery beyond validating
the source IDs named in the manifest.

Exact-hash donor adoption is intentionally out of scope. Donor reuse is zero-model
work and belongs in a separate migration. Prepare bounded-run manifests only for
genuinely new images that have no exact-hash donor.
