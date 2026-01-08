Run the pr-reviewer agent on the most recent changes (use git diff main).

## Steps

1. Run pr-reviewer agent on git diff
2. List CRITICAL and IMPORTANT issues only (skip minor/style)
3. Apply fixes for CRITICAL and IMPORTANT issues
4. Commit with message "fix: address pr-reviewer findings"

## Output Markers (Required for Automation)

After completing the review, output these markers:

**Issue counts:**
```
CRITICAL issues: {count}
IMPORTANT issues: {count}
```

**Summary:**
```
Review complete: {count} issues fixed
```

If no issues were found:
```
CRITICAL issues: 0
IMPORTANT issues: 0
Review complete: 0 issues fixed
```

## Restrictions

Do NOT:
- Update README.md
- Update implementation-plan.md
- Use deepcontext (not needed for fixes)
