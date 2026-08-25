# Plan: Runtime invocations

**Status:** Proposed
**Date:** 2026-08-24
**Cross-repository dependency:** cllama freezes and tags the contract first;
my-cli and Clawdapus can then consume it independently; the optional MGL
adapter follows the frozen rule and policy shapes

## Outcome

Make cllama the single runtime authority for identity-bound LLM access,
context assembly, skills, tools, memory, model policy, budget, policy hooks, and
audit. A human launcher and a container controller use the same versioned
Invocation contract. A new invocation works without a materialized context
directory.

This is a redesign, not an extension of context-directory IPC. The existing
directory loader remains temporarily as a compatibility adapter.

## Operator and harness contract

The complete system must satisfy eight cross-repository stories:

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
8. A human-operated harness and an organization-managed claw selecting the
   same declared role resolve the same role-entry and contract digests,
   instructional skill closure, managed business-tool schemas, memory policy,
   and rules. Subject-specific runtime envelopes and external principals remain
   explicit and do not change the role package.

cllama owns the final effective request for all eight stories. Launchers supply
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
- for bundle publication, ordered organization/effective-role contract inputs;
  for selection create, only bounded purpose/task and invocation-local late
  inputs;
- skill index entries with optional skill/tool requirements and bounded
  instructional bodies;
- managed tool specifications with deployment-owned, identity-aware principal
  binding references;
- for bundle publication, feeds and context blocks; a selection create cannot
  supply an endpoint, memory relationship, or tool target, and a controller may
  request only the reserved `pod-members` subscription resolved from its
  current bundle grant;
- one memory relationship plus named `view`;
- inline compiled rules plus digest;
- model policy, budget, channel allowlist, and delegation ceiling;
- `policy_exempt` for the policy evaluator's own LLM Invocation only when the
  current bundle's controller grant explicitly authorizes it, preserving
  ADR-025 recursion prevention;
- requested expiry.

`CreateInvocationResponse` returns only the server-assigned invocation ID,
one-time raw bearer, expiry, and projection digest. It never echoes provider or
control credentials. The stored `Invocation` is immutable.

Purpose is context, not authority. It may narrow a role but never adds tools,
models, memory scope, budget, channels, or delegation. Role remains a loadout;
the authenticated issuer authorization decides whether a subject may assume
it.

For organization-managed creates, the request is a selection plus bounded late
inputs: organization, effective role, purpose, harness, requested expiry,
subject/runtime identity where the issuer is allowed to assert it, and
invocation-local session or work references. It cannot carry organization or
role modules, skills, tools, rules, memory policy, models, budgets, channels,
or delegation. Those resolve from the current bundle. A Flux task identifier,
session startup note, or label may appear only in the non-authoritative late
layer and never changes the stable role entry.

### Control API in the existing cllama process

Add these routes to `cmd/cllama/main.go` through
`internal/control/handler.go`:

- `PUT /control/v1/loadouts/{organization}` — admin-only atomic replacement
  of one organization's complete membership and role-loadout bundle;
- `GET /control/v1/loadouts[/{organization}]` — admin-only redacted bundle
  list/inspection with digests;
- `DELETE /control/v1/loadouts/{organization}` — admin-only removal that
  disables future central-human creates for the organization;
- `POST /control/v1/issuers` — admin-only creation of a
  person/organization-scoped launcher issuer or an
  organization/controller-scoped pod issuer, returning its raw token once;
- `GET /control/v1/issuers[/{id}]` — admin-only redacted issuer inspection;
- `DELETE /control/v1/issuers/{id}` — admin-only idempotent revocation;
- `POST /control/v1/invocations` — authorize, validate, project, persist, and
  return a bearer once;
- `GET /control/v1/invocations` — credential-scoped list with redacted metadata
  and state: bootstrap admin sees all, a controller issuer sees only its
  organization and bundle-declared member namespace, and a personal issuer
  sees only its bound subject;
- `GET /control/v1/invocations/{id}` — inspect the immutable projection and
  provenance without any secret, with the same credential scope as list;
- `DELETE /control/v1/invocations/{id}` — revoke idempotently;
- `POST /control/v1/invocations/revoke` — admin-only bulk revocation filtered
  by organization plus one or more of subject, role-entry digest, or bundle
  digest, for urgent membership/contract/grant changes; returns redacted IDs
  and count.

This is another surface on the existing binary, not another service.

The reference binary supports two creation paths behind one
`InvocationAuthorizer` interface:

1. a trusted issuer credential, configured for a local launcher, one central
   human subject, or a pod controller, that may create only its configured
   organizations, subjects, pods, and roles; and
2. a parent Invocation bearer, presented alongside an otherwise authorized
   launcher or controller issuer, that may create a child only when every
   resolved capability is within its explicit delegation ceiling.

Issuer scopes are explicit rather than an ambient superuser flag. A
loadout/issuer admin credential cannot create an Invocation or call the data
plane; a central personal issuer cannot submit capability inputs; and a local
launcher or pod-controller issuer cannot mutate bundles. Admin scope is always
a distinct credential.

The authorizer returns the authorized, normalized Invocation inputs; the
control handler never persists the raw request first. Every
organization-managed my-cli or Clawdapus create resolves the server-owned role
entry from the current organization bundle. A central human authenticator
supplies its bound person subject and is rechecked against `members`. A
pod-controller issuer supplies its bound controller ID and may assert only a
member subject in the corresponding bundle-declared namespace and one of that
controller grant's current roles.
Either may add only the bounded selection and late inputs described above. A
controller may enable the reserved `pod-members` feed, but cllama resolves its
URL and credential alias from that controller's current bundle entry; no
selection request can name or override a feed, memory, or tool endpoint.
Thus `mayAssume` is not merely a role boolean followed by acceptance of
caller-authored tools or context.

A standalone pod or development launcher may publish an atomic bundle with
explicit `pod-local` or `local` provenance and then use the same selection
path. A pod-local bundle contains explicit flat single-source role entries;
there is no inheritance, MGL contract composition, or inferred dependency
closure. V1 has no separate complete-input create authority. For a member
child, parent creation requires both the controller issuer and the parent
Invocation bearer. It resolves the selected child role from the same current
bundle, checks that role against the controller's current `controllers` grant,
starts from the resolved parent Invocation, and applies subset checks. The
controller credential proves pod scope; the parent bearer proves the
delegation ceiling.

### Server-owned organization loadouts

Central mode uses one atomic, versioned bundle per organization rather than
piecemeal role records. The deployment's bootstrap admin credential publishes
the complete bundle through `PUT /control/v1/loadouts/{organization}`. It contains the
person-subject-to-role `members` map, a `controllers` map from stable controller
ID to member subject namespace, allowed roles, an optional declared
`pod_members_feed {base_url, auth_ref}`, and the narrowly constrained evaluator
exemption flag, organization context inputs,
and every named,
already-flattened effective role's contract modules, skills, tools, feeds,
memory view, rules, model policy, budget, channels, `max_invocation_ttl`,
delegation ceiling, and
allowed harness protocols. Each role entry carries its organization-contract,
effective-role-contract, source-role, and skill-body digests. It contains no
credential.
Tool/provider entries may name deployment-owned logical credential references;
the server validates their organization scope and resolvability before commit
and never accepts a raw secret in the bundle. The reference resolver maps
configured `server://<organization>/<name>` aliases to process environment or
deployment secret-store values; other resolvers implement the same narrow
interface.
The same rule applies to controller feed entries: central `base_url` must be an
administrator-published HTTPS origin containing scheme, host, and optional port
only—no userinfo, path, query, or fragment—and `auth_ref` a resolvable
organization-scoped server alias. cllama appends only the fixed
`/feeds/pod-members` path, disables redirects, pins the request to that origin,
and sends the resolved ADR-013 feed credential; the
raw credential never enters the bundle, Invocation, controller request, or
workload. Standalone mode may use its private network origin and local alias.
The handler validates the entire bundle before atomically replacing the current
digest, so a caller can never observe mixed organization/role generations and
removed roles or members do not remain as stale grants.

The bundle is also the sole role-authorization source for
organization-managed pods. Admin issuer creation binds a controller token to
an organization and controller ID only after that ID exists in the current
bundle; every create rechecks the current `controllers` entry. Removing or
narrowing the entry affects future creates without rotating the token. The pod
cannot self-select its namespace or role scope.

An internal policy evaluator uses the same controller issuer kind rather than
a third authentication path. Its current bundle entry has an isolated
`policy-evaluator/` member namespace, allows only the internal
`policy-evaluator` role, and explicitly sets `policy_exempt: true`. That flag is
valid only for this exact role shape. Issuer administration cannot add or
override it; removing or narrowing the bundle entry disables future evaluator
Invocation creation.

Role composition is an organization-control-plane concern. cllama does not
evaluate role inheritance, attributes, labels, conditions, or whether a parent
revision is current in the canonical control plane. It accepts only declared
flat role entries and verifies canonical digests, unique IDs, bounded bodies,
complete skill/tool dependency closure, credential-reference shape, the
presence and shape of composite parent-revision digests, and the other v1
limits. Unknown dependencies, a dependency cycle encoded in the index, or an
internally inconsistent source digest fails bundle publication atomically.
my-cli alone raises `role_revision_stale` by comparing those recorded parent
digests with current canonical revisions before publication.

List and get return redacted metadata, entry digests, and provenance. A
central-human create resolves the token-bound subject against `members`,
selects the requested role from the current bundle, records both the bundle
digest and role-entry digest in the immutable Invocation, and then applies the
optional narrowing purpose. Missing organization/role state fails closed as
`loadout_missing`; a missing subject/role grant is `role_not_authorized`.
Per-person issuers cannot publish, inspect, or delete loadouts. Deleting or
replacing a bundle affects future creates only; already-issued Invocations
remain immutable and are governed by normal expiry/revocation.

The reference central-human mechanism is a random per-person issuer token sent
only over HTTPS. The server stores its digest and maps it to one immutable
person subject plus organization. Issuer creation accepts only the exact
subject ID already present in the current bundle; this credential binding is
not a person directory. Current role authorization comes from the published
bundle, and the client cannot choose a different subject. my-cli
stores the raw issuer token through an OS
keychain/1Password-style `auth_ref`, never in the organization manifest or
harness environment. A deployment supplies one bootstrap admin credential via
its secret mechanism; that credential provisions, inspects, rotates, and
revokes personal issuers through the routes above and is never valid on the
data plane. Revocation takes effect on the next control request. This makes the
reference central flow operable without adding an identity service; OIDC or
mTLS may later replace the issuer store.

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

Add the same small in-memory and atomic-directory implementations for the
organization-loadout store. Bundle replacement is validate-then-rename, stores
only canonical bytes and digests, and is private control-plane state. It is not
a workload mount or a new service.

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

cllama, not a launcher or harness adapter, assembles the stable
provider-cacheable prefix in one fixed order:

1. organization contract;
2. flattened effective-role contract modules;
3. role skill index.

The late seam then appends, in order:

4. bounded purpose/task text and Invocation metadata;
5. bounded session/Flux-task startup context; and
6. dynamic memory, context blocks, feeds, channel deltas, and time.

Each role entry grants exactly one named memory view. Purpose is bounded late
query/narrowing metadata within that view plus optional caller-supplied text;
it cannot select another view, and there are no purpose modules looked up from
the bundle. Labels are propagated
only as late metadata to already-granted feeds, memory, routing, and audit; they
do not select modules or grants. OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages
paths use the same normalized projection and record module provenance and
digests in context snapshots.

Instructional skills are presented as a granted index plus the existing
managed `load_skill` operation. Ungranted names are indistinguishable from
missing names. Index `requires` metadata is informational at request time; the
bundle publisher computes the closure and cllama validates it before accepting
the bundle. Executable operations remain managed tools. Files/assets needed by
a workload are launcher materialization concerns and are never ambient proxy
authority.

There is no runtime role preprocessor or context DAG. The organization control
plane publishes only declared effective roles. Stable projection/cache keys
include bundle, role-entry, organization-contract, effective-role-contract,
and harness/protocol digests. Purpose, Flux task/session startup context,
labels, dynamic memory, feeds, and time remain late so a new assignment does
not invalidate the stable role prefix.

### Existing capability planes

- Feeds are stored as typed invocation entries; `fetchFeeds` no longer reads a
  manifest from `ContextDir`. Cache keys begin with invocation ID.
- `pod-members` is an ordinary authenticated feed subscription with zero cache
  TTL. For a controller-created Invocation cllama resolves the route only from
  that controller's current bundle grant and fetches it on every requesting
  turn. It does not become the membership database; Clawdapus owns the
  current-state payload and its conformance fixture. Caller-supplied URLs are
  rejected.
- Memory recall/retain adds invocation ID, stable subject, role, purpose/view,
  and pod provenance. One configured memory relationship remains the rule.
- Rules use one inline `RulesManifest {version, rules, artifact_sha256}`. Each
  entry keeps the existing `{id, mode, text, source, content_sha256}` shape. A
  path is accepted only through the legacy adapter.
- Managed tools, model policy/failover, budget, channels, response gates, and
  policy-evaluator hooks receive `InvocationIdentity` rather than a bare agent
  string.
- Provider and managed-tool credential references resolve only through the
  deployment-owned proxy/tool-broker boundary. Unknown, cross-organization, or
  raw credential values fail bundle publication or selection resolution before
  an Invocation is stored.
- A managed tool may bind to a deployment-owned standing role-service alias or
  to a short-lived subject grant/session supplied by a trusted local broker or
  Flux Gate runtime integration. At execution time the injected
  `ToolPrincipalResolver.Resolve(InvocationIdentity, tool_id, operation,
  task_ref)` seam may obtain a short-lived Flux Gate binding only for the
  already-granted tool and operation; it returns the credential directly to the
  tool executor, cannot widen the loadout, and stores no grant state. A person
  on a local sidecar may use their own
  OS-keychain/1Password reference. A member cannot request a person binding,
  central cllama stores no standing per-person portal credential, and no secret
  or browser session reaches the harness.
- Audit and session history add invocation ID, subject ID, role, purpose, and
  projection digest.

### Durable work and external effects

cllama neither orchestrates business work nor pauses a mediated tool loop for
human approval. Flux owns durable tasks, blockers, wakes, checkpoints, and
fingerprinted approval requests. A controller-side Flux adapter may claim a
wake and then create a new selection-based Invocation with the Flux task
reference in the late purpose/task layer; a human explicitly invokes their
launcher for a queued Flux task. cllama has no wake receiver.

Flux does not perform privileged effects. Flux Gate owns just-in-time access
grants and their authoritative ledger; accounting, mail, source-control,
payment, and other domain services own their effects. The effect-owning service
defines the canonical proposal schema and digest; Flux stores that opaque ID
and digest. The effect owner or its isolated executor issues a signed,
audience-bound, expiring, single-use release token only after reading the Flux
approval record with its own credential and verifying that the approved opaque
proposal digest equals its canonical digest. The service
validates issuer, signature, audience, digest, expiry, and nonce and consumes
the token atomically. Flux Gate ceremonies apply to JIT access grants, not
effect-release signing. To cllama these are ordinary managed tools. cllama
authenticates the Invocation,
enforces the granted operation and identity binding, records mediation, and
does not become the approval or effect system of record.

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
- Do not add free-form role union, attribute-driven authority, label-conditioned
  stable modules, a second `access[]` capability plane, a durable-work ledger,
  or approval pause/resume to cllama.
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
- `policy_exempt` can be set only by a controller issuer whose current bundle
  grant has the exact internal evaluator shape above, is inherited as false by
  ordinary children, and bypasses policy callbacks only; credential
  starvation, model policy, budget, audit, and all other enforcement remain.
- A policy evaluator uses a separately published internal
  `policy-evaluator` role entry with no business tools or memory, an explicit
  model/budget, and a controller issuer bound to that current bundle entry; it
  cannot synthesize a caller-authored loadout or gain exemption from issuer
  administration alone.
- Central loadout bundles contain only credential aliases; raw provider/tool
  secrets remain in deployment configuration and never enter a bundle,
  Invocation, launcher response, or workload.
- cllama's bundle store is a runtime publication, not the canonical directive,
  role, membership, or work store. `ContractRevision` (one scope), bundle (one
  organization), and Invocation projection (one launch) are digest-linked and
  do not overwrite one another.
- The legacy directory adapter and an equivalent Invocation must produce
  semantically equivalent effective upstream requests during migration.

## 4. Unit and integration verification

Add focused tests at the ownership boundary:

- store: entropy, digest-only persistence, atomic recovery, expiry,
  revocation, duplicate-ID rejection, concurrent resolution, and no secret in
  list/get/log output;
- control: issuer authentication, bootstrap-admin and admin-only person and
  controller issuer create/redacted-read/revoke, current-bundle existence and
  namespace/role-scope checks, one-time raw token return, admin-only
  atomic bundle put/list/get/delete, complete-bundle validation,
  stale-role/member removal, subject/role denial,
  `loadout_missing`, schema/version and TTL limits, central server-owned
  loadout resolution, recorded bundle/role digests, rejection of
  caller-authored capabilities on that path, raw/unknown/cross-organization
  credential-reference rejection, parent subset success, and escalation denial
  before record creation;
- control separation: admin credentials cannot create; launcher/controller
  issuers cannot publish; urgent filtered bulk revoke affects exactly the
  selected organization/subject/bundle/role-entry digest;
- evaluator authorization: only the current exact `policy-evaluator`
  controller grant permits `policy_exempt`; admin flags, personal issuers,
  ordinary controller entries, and child delegation cannot enable it;
- role-entry validation: exact contract/source digests, complete nested skill
  and tool closure, missing/cyclic dependency rejection, positive bounded
  `max_invocation_ttl`, requested-expiry denial above that ceiling, and
  rejection of any stable module or grant on an organization-managed selection
  create;
- scoped issuer: digest-only token storage, immutable subject binding,
  cross-subject/cross-role denial, HTTPS requirement for remote use, rotation,
  revocation, absence from harness environment and audit logs, and list/get
  isolation across subjects, controller namespaces, pods, and organizations;
- child create: parent bearer plus launcher/controller issuer are both
  required; a member child role must fit both the parent delegation ceiling and
  the controller's current bundle grant;
- data plane: invalid, expired, and revoked bearers make zero downstream calls;
- projection: exact stable module order and late-context placement for OpenAI
  Chat Completions, OpenAI Responses, and Anthropic Messages;
- isolation: concurrent invocations for one subject with different roles and
  views never share feed cache, context, tools, continuity, snapshots, budget,
  or history;
- propagation: memory, feeds, tools, policy, audit, and compatibility headers
  receive the intended stable and ephemeral identities;
- membership feed: zero TTL forces a fetch on the next turn and the feed cache
  never shares entries across Invocations; central mode accepts only the
  current controller grant's HTTPS origin and credential alias, and rejects
  caller-supplied, cross-controller, or non-feed endpoints before fetch; URL
  userinfo/path/query/fragment and redirects fail with zero follow-on request;
- tool principal resolution: a short-lived Flux Gate grant resolves only after
  Invocation/tool/operation/task binding, reaches only the tool executor, and
  cannot widen the role or persist in cllama;
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

### `TestSpikeRoleOnboarding`

Use the canonical billing bundle fixture owned by
`pkg/invocation/testdata/conformance/v1/`. Publish it with a declared
`billing-reconciliation` role and controller grant, organization and role
contract revisions, nested skill dependencies, fake
accounting and browser-broker tools, Flux/Flux Gate tools, and a
release-token-validating fake send service. Create one person and one member
Invocation by selection. Assert identical stable role/contract/skill/tool
digests, distinct runtime envelopes and principal bindings, no browser/session
secret exposure, missing, mismatched, forged, expired, replayed, and
wrong-audience release-token denial, and resume in a new Invocation from a Flux
task reference without stable-prefix churn. From central cllama, fetch the
member's authenticated `pod-members` endpoint through only the
bundle-declared controller route and reject an attacker-supplied URL. A
redirecting feed returns a fetch failure and receives no follow-on request; URL
userinfo, path, query, and fragment fail bundle publication.

### `TestSpikeScopedCentralIssuer`

Run cllama behind a TLS test server with a bootstrap admin credential. Use it to
create two per-person issuer records and one controller issuer, all bound to
subjects/controller IDs already present in the current bundle, and prove raw
tokens are returned once and redacted thereafter. Prove the admin can
atomically publish and replace a complete
organization bundle; personal issuers cannot publish or inspect it; an absent
bundle or role returns `loadout_missing`; each personal token is bound to its
subject; the published `members` map governs permitted roles; removed
members/roles lose future-create authority; caller-authored capability fields
and raw/unknown/cross-organization credential references fail before
persistence; a valid `server://` alias resolves only at request execution; the
Invocation records the selected bundle and role digests; issuer revocation
takes effect on the next create; and no raw admin/personal issuer or
provider/tool token reaches a workload, list/get response, store record, or
log. Restart the real process after publication and prove the same canonical
bundle digest still governs new creates.
Prove controller namespace/role denial, removal from the current bundle,
attempted self-scoping, and use of any issuer for bundle publication all fail
closed. Prove a personal issuer cannot list/get another subject, a controller
cannot list/get outside its organization or member namespace, and only the
bootstrap admin can inspect across those scopes. Subject-filtered bulk revoke
affects exactly that person's live Invocations.

The bootstrap admin is the deployment's single admin-surface credential class:
it may publish bundles and administer issuers but is never valid for Invocation
creation or the data plane. A separately scoped controller entry for the
internal evaluator proves that only the bundle-authorized evaluator can create
`policy_exempt` Invocations.

Add CI commands for both the ordinary suite and the hermetic spike target. The
cllama release is blocked unless all seven pass.

## 6. Implementation and release sequence

1. Land the public v1 types, issuer and organization-loadout bundle types,
   selection/late-input boundary, contract/source digests, skill requirements,
   identity-aware tool binding, redacted error
   model, conformance fixtures, and stdlib client.
2. Land the in-memory and atomic-directory Invocation/loadout/issuer stores,
   token hashing, control authorizer, admin-scoped issuer and bundle endpoints,
   and Invocation create/list/get/delete endpoints.
3. Resolve `InvocationIdentity` in the proxy and convert stable context
   assembly for all three ingress formats.
4. Convert feeds, memory, tools, policy, budgets, snapshots, history, and audit
   one plane at a time; keep all existing tests green at each step.
5. Put the old `agentctx` tree behind the legacy adapter and add semantic parity
   coverage.
6. Land and enable all hermetic binary spikes in CI. Publish a tagged cllama
   release only after the v1 contract is frozen by those fixtures.
7. my-cli consumes the tag for human harness launch/revoke. Independently,
   Clawdapus consumes the same tag, uses two-phase proxy bootstrap, and bumps
   its submodule only to the released tag.
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
