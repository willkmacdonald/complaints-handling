Finalize the current sprint after all PRs are complete.

## Context

This is the **Medical Device Complaint Handling System** project. Read `current-sprint.md` to identify the sprint name and completed work.

## Steps

1. **Run Security Scan**
   - Review the codebase for security issues, focusing on:
     - Hardcoded credentials or API keys
     - SQL injection vulnerabilities
     - Input validation gaps
     - PHI/PII data handling
   - Fix any CRITICAL or HIGH severity issues
   - If security issues found, commit fixes with "fix: address security findings"

2. **Run Tests**
   - Run `pytest` to ensure all tests pass
   - Fix any failing tests before proceeding

3. **Update README.md**
   - Review changes made during the sprint
   - Update the "Completed" section under "Project Status" with new PRs
   - Update "Project Structure" if new directories were added
   - Keep updates concise and relevant

4. **Update implementation-plan.md**
   - Mark completed deliverables with `[x]` checkmarks
   - Update phase status if appropriate (e.g., "IN PROGRESS" → "COMPLETE")
   - Add any new items discovered during implementation

5. **Update current-sprint.md**
   - Mark all completed PRs with checkmarks
   - Update "Sprint Exit Criteria" section

6. **Commit Documentation**
   - Stage all documentation changes
   - Commit with message: "docs: finalize sprint - {sprint_name}"
   - Extract sprint name from the `# Current Sprint:` header in current-sprint.md

7. **Push to Remote**
   - Push changes to origin/main

## Guard Rails

Do NOT:
- Skip the security scan (it must run even if you think code is safe)
- Skip running tests
- Create new feature files (finalization is documentation only)
- Modify source code except for security fixes or test fixes
- Add new dependencies
- Push to any branch other than main

## Output Markers (for automation)

When complete, output one of:
- "Sprint finalization complete: {sprint_name}"
- "Sprint finalization failed: {reason}"
