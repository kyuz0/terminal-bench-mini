# Suite manifests

Each JSON manifest defines one immutable benchmark suite. A source records its
upstream dataset, version, repository revision, and the relative directory that
contains its tasks. Each task records its upstream name, optional upstream
registry digest, and a mandatory digest of the complete local task directory.

`content_sha256` uses `terminal-bench-local-task-tree-v1`: files are ordered by
their relative POSIX paths, and SHA-256 covers the algorithm marker plus each
length-prefixed path, its executable flag, and the SHA-256 of its bytes. Other
file metadata and checkout location are excluded. Symlinks are rejected. The
digest therefore detects any change to instructions, environments, solutions,
verifiers, or supporting assets without making otherwise identical checkouts
host-dependent.

The suite manifest hash covers its ID, version, upstream provenance, tiers, and
all per-task identities. Human-facing text and local task-root paths do not
affect it. Results from manifest-backed runs are stored in a namespace containing
the suite ID and manifest hash, so tasks from different suite definitions cannot
be merged accidentally.
