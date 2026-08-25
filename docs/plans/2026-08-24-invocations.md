# Plan: Runtime invocations

**Status:** Proposed
**Date:** 2026-08-24
**Cross-repository order:** cllama, then my-cli, then Clawdapus, then optional MGL adapter

## Outcome

Make cllama the single runtime authority for identity-bound LLM access,
context assembly, skills, tools, memory, model policy, budget, policy hooks, and
audit. A human launcher and a container controller use the same versioned
Invocation contract. A new invocation works without a materialized context
directory.

This is a redesign, not an extension of context-directory IPC. The existing
directory loader remains temporarily as a compatibility adapter.

## Operator and harness contract

The complete system must satisfy seven cross-repository stories:

1. `my ai --role <role> --purpose <purpose> <harness>` starts Claude Code or
   Codex through cllama with a new, short-lived invocation and revokes it on
   exit.
2. `claw up` starts every declared member with the same invocation semantics
   and no provider credentials or context-directory dependency in its
   container.
3. `claw spawn` and `claw retire` add and remove a member without rebuilding or
   restarting existing members.
4. A member can ask the trusted Clawdapus controller to create a child only
   within its delegation ceiling.
5. Existing members observe joins and departures on their next turn through a
   live `pod-members` feed, not rebuilt environment or peer files.
6. Context and memory are isolated by stable subject, role, purpose/view, and
   invocation provenance.
7. The same launch contract works against a local sidecar or centrally managed
   HTTPS proxy and fails closed when authorization, proxy reachability, or
   harness compatibility fails.

cllama owns the final effective request for all seven stories. Launchers supply
authenticated, normalized inputs; they do not render the provider-visible
system prompt.

## 1. New and changed surfaces

### Public v1 contract

Add a stdlib-only public wire package and client:

- `pkg/invocation/`: versioned request, response, identity, capability, and
  error types;
- `pkg/invocationclient/`: create, inspect, list, revoke, and resolve-error
  helpers;
- `pkg/invocation/testdata/conformance/v1/`: canonical valid and invalid JSON
  requests, redacted responses, and expected effective-request snapshots.

`CreateInvocationRequest` carries:

- `version`;
- stable `subject {kind, id}` where v1 kinds are `person` and `member`; it is
  required in the stored Invocation but may be omitted from a central-human
  create request when the authorizer supplies it;
- `organization`, `role`, and optional `purpose`;
- optional `parent_invocation_id`;
- `harness {name, protocol}` where v1 protocols are `openai-chat`,
  `openai-responses`, and `anthropic-messages`;
- optional runtime identity: pod, service, ordinal, labels, handles, and image
  digest;
- ordered context inputs by kind: organization contract, role, purpose/task,
  and invocation-local modules;
- skill index entries and bounded instructional bodies;
- managed tool specifications;
- feeds and context blocks;
- one memory relationship plus named `view`;
- inline compiled rules plus digest;
- model policy, budget, channel allowlist, and delegation ceiling;
- trusted-issuer-only `policy_exempt` for the policy evaluator's own LLM
  invocation, preserving ADR-025 recursion prevention;
- requested expiry.

`CreateInvocationResponse` returns only the server-assigned invocation ID,
one-time raw bearer, expiry, and projection digest. It never echoes provider or
control credentials. The stored `Invocation` is immutable.

Purpose is context, not authority. It may narrow a role but never adds tools,
models, memory scope, budget, channels, or delegation. Role remains a loadout;
the authenticated issuer authorization decides whether a subject may assume
it.

### Control API in the existing cllama process

Add these routes to `cmd/cllama/main.go` through
`internal/control/handler.go`:

- `POST /control/v1/invocations` — authorize, validate, project, persist, and
  return a bearer once;
- `GET /control/v1/invocations` — operator list with redacted metadata and
  state;
- `GET /control/v1/invocations/{id}` — inspect the immutable projection and
  provenance without any secret;
- `DELETE /control/v1/invocations/{id}` — revoke idempotently.

This is another surface on the existing binary, not another service.

The reference binary supports two creation authorities behind one
`InvocationAuthorizer` interface:

1. a trusted issuer credential, configured for a local launcher, one central
   human subject, or a pod controller, that may create only its configured
   organizations, subjects, pods, and roles; and
2. a parent invocation bearer that may create a child only when every resolved
   capability is within its explicit delegation ceiling.

The authorizer returns the authorized, normalized Invocation inputs; the
control handler never persists the raw request first. A locally trusted my-cli
issuer or pod-scoped Clawdapus controller may supply complete inputs within its
configured organization/pod/subject/role scope. A central human authenticator
may instead accept only organization, role, purpose, and harness selection and
resolve the server-owned loadout itself. Thus `mayAssume` is not merely a role
boolean followed by acceptance of caller-authored tools or context.

The reference central-human mechanism is a random per-person issuer token sent
only over HTTPS. The server stores its digest and maps it to one immutable
person subject plus organization and role bounds; the client cannot choose a
different subject. my-cli stores the raw issuer token through an OS
keychain/1Password-style `auth_ref`, never in the organization manifest or
harness environment. Revocation takes effect on the next control request.

An ordinary data-plane bearer cannot create an unrelated invocation. A central
deployment may replace the reference token check with OIDC, mTLS, or another
human authenticator while retaining server-side `mayAssume(subject, role)`
through the same interface. V1 does not require launcher public-key
distribution, workload PKI, OIDC, or a generic ABAC service.

### Invocation store

Add `internal/invocation/{store,token,directory_store}.go`:

- `Create`, `Resolve`, `List`, `Get`, and `Revoke`;
- random bearer secret returned once, SHA-256 digest stored;
- one atomically replaced JSON record per invocation;
- restart recovery, deterministic expiry, and idempotent revocation;
- in-memory implementation for tests.

Filesystem storage is private cllama persistence. It is never mounted into a
workload and is not a second context transport. A later storage backend can
implement the same small interface without changing callers.

### Data-plane resolution and identity

Change bearer parsing to `<invocation-id>:<secret>`. Resolve one immutable
`InvocationIdentity` at the start of a request and carry it through provider
routing, feeds, memory, policy, tools, budgets, snapshots, history, and audit.

The identity separates:

- stable subject ID, used for governed memory scope and compatible member
  identity;
- invocation ID, used for authentication, cache, continuity, budget, history,
  snapshots, and audit isolation;
- role and purpose, used for loadout and view selection.

For member subjects, compatibility `X-Claw-ID` remains `subject.id`. Add
`X-Claw-Invocation-ID`; propagate role and purpose only to explicitly trusted
services that need them. Never log the bearer secret.

### Final context assembly

cllama, not a launcher or harness adapter, assembles the effective request in
one fixed order:

1. organization contract;
2. role modules;
3. purpose/task modules;
4. invocation metadata and skill index;
5. bounded dynamic memory, context blocks, feeds, channel deltas, and time at
   the existing late-context seam.

Stable modules remain in the provider-cacheable prefix. Volatile context
remains late. OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages
paths use the same normalized projection and record module provenance and
digests in context snapshots.

Instructional skills are presented as a granted index plus the existing
managed `load_skill` operation. Ungranted names are indistinguishable from
missing names. Executable operations remain managed tools. Files/assets needed
by a workload are launcher materialization concerns and are never ambient
proxy authority.

### Existing capability planes

- Feeds are stored as typed invocation entries; `fetchFeeds` no longer reads a
  manifest from `ContextDir`. Cache keys begin with invocation ID.
- `pod-members` is an ordinary authenticated feed subscription with zero cache
  TTL. cllama fetches it on every requesting turn and does not become the
  membership database; Clawdapus owns the current-state payload and its
  conformance fixture.
- Memory recall/retain adds invocation ID, stable subject, role, purpose/view,
  and pod provenance. One configured memory relationship remains the rule.
- Rules use one inline `RulesManifest {version, rules, artifact_sha256}`. Each
  entry keeps the existing `{id, mode, text, source, content_sha256}` shape. A
  path is accepted only through the legacy adapter.
- Managed tools, model policy/failover, budget, channels, response gates, and
  policy-evaluator hooks receive `InvocationIdentity` rather than a bare agent
  string.
- Audit and session history add invocation ID, subject ID, role, purpose, and
  projection digest.

## 2. Delete or demote

- Demote required `AGENTS.md`, `CLAWDAPUS.md`, `metadata.json`, and sibling
  capability files under `CLAW_CONTEXT_ROOT` to `agentctx.LegacyAdapter`.
- Remove plaintext bearer tokens from every new persistent record.
- Remove `ContextDir` from the dynamic request path, including feed and rules
  loading.
- Demote process-global `CLAW_POD` identity to legacy mode; pod is per
  invocation.
- Stop overloading `agentID` as authentication ID, memory principal, feed/tool
  caller, continuity key, and audit key.
- Move clawdash/claw-api consumers from context-directory scanning to the
  redacted list/inspect API.
- Do not add context generations or symlinks, a second projection compiler,
  a configuration service, CSR/PKI, peer key exchange, an event bus, a generic
  pipeline DAG, or a separate authorization service.
- Do not mediate ChatGPT or Claude subscription OAuth tokens in v1. Documented
  harness gateway/custom-provider modes receive only the cllama invocation
  bearer; cllama uses supported provider/API credentials. A vendor-supported
  subscription adapter can be proposed later behind an optional credential
  interface.

## 3. Invariants to preserve

- Credential starvation: workloads never receive provider credentials.
- One immutable resolution per request; revoked or expired invocations fail
  before provider, feed, memory, policy, or tool calls.
- Unknown/malformed credentials return authentication failure. A recognized
  revoked invocation returns HTTP 403 with intervention
  `invocation_revoked`; expiry is equally explicit.
- Existing provider identity, model policy/failover, budget, managed-tool
  continuity, streaming, response gates, original/effective auditing, and
  OpenAI/Anthropic ingress parity remain intact.
- Prompt-cache stability remains intact: static prefix first, volatile context
  late, no per-turn timestamps in the stable prefix.
- Parent creation is subset-only across role grants, tools, model policy,
  memory scope/view, budget, channels, TTL, and further delegation.
- `policy_exempt` can be set only by a trusted issuer, is inherited as false by
  ordinary children, and bypasses policy callbacks only; credential
  starvation, model policy, budget, audit, and all other enforcement remain.
- The legacy directory adapter and an equivalent Invocation must produce
  semantically equivalent effective upstream requests during migration.

## 4. Unit and integration verification

Add focused tests at the ownership boundary:

- store: entropy, digest-only persistence, atomic recovery, expiry,
  revocation, duplicate-ID rejection, concurrent resolution, and no secret in
  list/get/log output;
- control: issuer authentication, subject/role denial, schema/version limits,
  TTL limits, central server-owned loadout resolution, rejection of
  caller-authored capabilities on that path, parent subset success, and
  escalation denial before record creation;
- scoped issuer: digest-only token storage, immutable subject binding,
  cross-subject/cross-role denial, HTTPS requirement for remote use, rotation,
  revocation, and absence from harness environment and audit logs;
- data plane: invalid, expired, and revoked bearers make zero downstream calls;
- projection: exact stable module order and late-context placement for OpenAI
  Chat Completions, OpenAI Responses, and Anthropic Messages;
- isolation: concurrent invocations for one subject with different roles and
  views never share feed cache, context, tools, continuity, snapshots, budget,
  or history;
- propagation: memory, feeds, tools, policy, audit, and compatibility headers
  receive the intended stable and ephemeral identities;
- membership feed: zero TTL forces a fetch on the next turn and the feed cache
  never shares entries across Invocations;
- legacy parity: typed semantic comparison of effective requests, not byte
  equality with old generated files;
- client/fixture conformance: both valid and intentionally invalid v1 fixtures
  are consumed by the public client without repository-private types.

All ordinary tests continue to pass with `go test ./...`.

## 5. Required delivery spikes

Hermetic spikes are mandatory CI gates and use the real cllama binary plus fake
HTTP dependencies. Live provider drills are additional release evidence and
never substitute for deterministic acceptance coverage.

### `TestSpikeDynamicInvocationBinary`

Start a real cllama binary, fake OpenAI Chat/Responses and Anthropic providers, fake
memory, fake feed, and fake managed-tool endpoints. Create one invocation,
exercise all three ingress protocols, and assert exact stable and late context,
allowed skill/tool/model behavior, memory view/provenance, audit fields, and
provider credential substitution. Revoke it and prove the same bearer returns
403 `invocation_revoked` with zero downstream calls. Inspect store and logs to
prove the raw bearer is absent.

### `TestSpikeConcurrentSameSubjectIsolation`

Create two simultaneous invocations for one person with different roles,
purposes, tools, feeds, memory views, and budgets. Assert disjoint effective
requests, tool grants, feed cache entries, memory calls, histories, snapshots,
and budget effects.

### `TestSpikeInvocationRestartExpiryAndInspection`

Create, list, and inspect an invocation without secret disclosure. Restart the
real process against the same store, prove the bearer and projection still
work, advance to expiry, and prove deterministic fail-closed behavior.

### `TestSpikeParentDelegationCeiling`

Use a parent bearer to create a permitted narrower child. Attempt to widen each
authority dimension independently and prove every attempt fails before either
an invocation record or provider call exists.

### `TestSpikeLegacyInvocationSemanticParity`

Load a representative legacy directory and an equivalent v1 invocation, run
both provider formats, and compare normalized provider-visible requests and
capability decisions. This protects migration without preserving the old file
layout as the new design.

### `TestSpikeScopedCentralIssuer`

Run cllama behind a TLS test server with two per-person issuer tokens and
server-owned role loadouts. Prove each token is bound to its subject, permitted
role selection succeeds, cross-subject/cross-role and caller-authored
capability fields fail before persistence, revocation takes effect on the next
create, and neither raw issuer token reaches a workload, list/get response,
store record, or log.

Add CI commands for both the ordinary suite and the hermetic spike target. The
cllama release is blocked unless all six pass.

## 6. Implementation and release sequence

1. Land the public v1 types, redacted error model, conformance fixtures, and
   stdlib client.
2. Land the in-memory and atomic-directory stores, token hashing, control
   authorizer, and create/list/get/delete endpoints.
3. Resolve `InvocationIdentity` in the proxy and convert stable context
   assembly for all three ingress formats.
4. Convert feeds, memory, tools, policy, budgets, snapshots, history, and audit
   one plane at a time; keep all existing tests green at each step.
5. Put the old `agentctx` tree behind the legacy adapter and add semantic parity
   coverage.
6. Land and enable all hermetic binary spikes in CI. Publish a tagged cllama
   release only after the v1 contract is frozen by those fixtures.
7. my-cli consumes the tag for human harness launch/revoke. Clawdapus consumes
   the tag after my-cli, uses two-phase proxy bootstrap, and bumps its submodule
   only to the released tag.
8. Remove the legacy directory adapter only after both clients and Claude Code,
   Codex, OpenClaw, Hermes, Nanobot, and Picoclaw pass the cross-repository
   delivery matrix.

## Deferred decisions

- OIDC, mTLS, or another production human authenticator may replace the
  reference scoped issuer token through `InvocationAuthorizer`; none is needed
  to make the central v1 story functional.
- Storage beyond the reference atomic directory store is an implementation of
  `Store`, not a protocol change.
- Subscription credential mediation requires an explicit vendor-supported
  gateway contract and its own threat model and acceptance spikes.
- Memory compaction algorithms and schedules belong to the selected memory
  service; cllama carries the identity-bound view and provenance.
