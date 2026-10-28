# Project working instructions

- Maintainer attribution is `lrtang`. Do not add automated-tool identities to commit author/committer fields, co-author trailers, bylines or generated-by notices.
- Preserve factual citations, required course disclosures and genuine human contributor attribution. Do not rewrite history to conceal another person's contribution.
- Groupmates keep their own commit identities. Fetch/sync remote changes before integration, work on a feature branch and preserve other contributors' edits; do not force-push shared history.
- After runtime source/config changes, rebuild scripts/build_team_notebook.py. Run its --check mode before committing a launcher. Shared notebook access and GPU/Drive permissions are separate.
- Read README.md and BUILD.md before changing the experimental protocol. Keep the original archived results and provenance; create a new run directory for new work.
- Update the relevant README when adding files. Use relative links. Keep raw data, weights, credentials, course attachments and runtime caches outside Git.
- Provide README.zh-CN.md alongside each section README.md. Keep CLI examples and extension limitations aligned with real code; rebuild/check both language launchers after relevant changes.
- Use scripts/check_repository.py and the relevant numerical tests. Report what actually ran and distinguish packaging checks from GPU results.
- Do not publish the private repository, submit coursework or contact others without authorization.
