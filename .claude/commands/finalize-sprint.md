Finalize the current sprint after all PRs are complete.

## Steps

1. **Run Security Scan**
   - Launch the security-scanner agent on the codebase
   - Review findings and fix any CRITICAL or HIGH severity issues
   - If security issues found, commit fixes with "fix: address security findings"

2. **Update README.md**
   - Review changes made during the sprint
   - Update README if any new features, commands, or usage patterns were added
   - Keep updates concise and relevant

3. **Update implementation-plan.md**
   - Mark completed items from this sprint as done
   - Add any new items discovered during implementation
   - Archive the sprint section if appropriate

4. **Commit Documentation**
   - Commit with message: "docs: finalize sprint - {sprint_name}"
   - Include sprint name from current-sprint.md YAML frontmatter

5. **Push to Remote**
   - Push changes to origin/main

## Guard Rails

Do NOT:
- Skip the security scan (it must run even if you think code is safe)
- Create new feature files (finalization is documentation only)
- Modify source code except for security fixes
- Add new dependencies
- Use deepcontext or Ref (not needed for documentation updates)
- Push to any branch other than main

## Output Markers (for automation)

When complete, output one of:
- "Sprint finalization complete: {sprint_name}"
- "Sprint finalization failed: {reason}"
