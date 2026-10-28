# Contribution conventions

Maintainer attribution for this checkout: **lrtang**. Author and committer names for the initial local history are both `lrtang`. Repository-local Git configuration controls these names and leaves global settings unchanged.

Do not append automated-tool author names, co-author trailers, sign-offs or generated-by banners. This concerns commit/byline attribution. Preserve scholarly citations, actual human contributions and any disclosures required by the course. Do not rewrite another contributor's history or claim work was unaided when that is untrue.

GitHub account association and the commit display name are separate fields. Repository authentication uses the owner account; a noreply commit email keeps the personal email out of history.

Groupmates use their own real Git identities. The `lrtang` setting applies to maintainer-authored work in this checkout and does not relabel other people's commits. Use a feature branch and Pull Request, sync before integration, and preserve other contributors' work. The process is documented in [COLLABORATION.md](COLLABORATION.md).

Before a commit, run the standard-library repository check and the tests affected by the change. Keep experimental definitions fixed before viewing new target results. Report negative seeds and limitations. Keep every new experiment in a fresh output directory.

Local checkpoint archives and course attachments are deliberately absent. Do not add them, credentials, a complete environment folder or another course's material to a commit. Member names, submission status and presentation slots require verification rather than inference.
