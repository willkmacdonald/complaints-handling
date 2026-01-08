Implement the PR described by the user.

IMPORTANT: Do NOT automatically use Ref, deepcontext, microsoft-learn or exa.
Only use these tools if:
- Ref: You need docs for a library you haven't used before in this project
- deepcontext: You're touching code in an unfamiliar part of the codebase
- exa: You need to look up something not in the codebase or library docs
- microsoft-learn: You need to verify the latest microsoft best practices (e.g., Microsoft Agent Framework, Foundry IQ)

## Automation Support

When running in automated mode, check the YAML frontmatter in current-sprint.md:
- `sprint.current_pr` indicates which PR to implement
- If set, implement that specific PR from the sprint file

## Steps

1. Read current-sprint.md to understand context
2. Check YAML frontmatter for `sprint.current_pr` (if present, implement that PR)
3. Read the key files listed in current-sprint.md
4. Implement the feature/fix
5. Run tests if applicable
6. Commit the changes with appropriate message

## Output Markers (Required for Automation)

At the end of implementation, output ONE of these markers:

**On success:**
```
PR {n} implementation complete
```

**On failure:**
```
PR {n} implementation failed: {reason}
```

Where `{n}` is the PR number and `{reason}` describes why it failed.

## Restrictions

Do NOT update README.md or implementation-plan.md (that's a weekly task).
