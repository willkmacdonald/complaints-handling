# Current Sprint: Phase 1 - Single Channel MVP (Structured Forms)

## Sprint Goal

Build a functional MVP that processes online form submissions, extracts complaint data, suggests IMDRF codes using an LLM, and provides a basic review interface with audit logging.

**Success Gate:** >80% of suggested codes accepted by human reviewer without modification

---

## Key Files Reference

```
src/
  models/           # Complaint data models (from Phase 0)
  imdrf/            # IMDRF code reference database (from Phase 0)
  evaluation/       # Evaluation framework (from Phase 0)
  testing/          # Test case loader (from Phase 0)
data/
  test_cases/form/  # 7 synthetic form test cases (from Phase 0)
  imdrf/            # IMDRF code reference data (from Phase 0)
```

---

## PR-1: Form Ingestion Pipeline

### Description
Build the pipeline to ingest online form submissions and convert them to the canonical ComplaintRecord format.

### Scope
- [ ] Define `FormSubmission` Pydantic model representing raw form data:
  - Submitter information (name, email, phone, relationship to patient)
  - Device information (name, manufacturer, model, serial, lot)
  - Event details (date, description, patient outcome, device outcome)
  - Attachments metadata
  - Submission timestamp
- [ ] Create `parse_form_submission(raw_data: dict) -> FormSubmission` function
- [ ] Create `form_to_complaint(form: FormSubmission) -> ComplaintRecord` conversion function
- [ ] Handle missing/optional fields gracefully
- [ ] Flag incomplete submissions that need follow-up
- [ ] Unit tests with form test cases

### Files to Create
```
src/
  intake/
    __init__.py
    forms.py         # FormSubmission model, parsing, conversion
tests/
  test_form_intake.py
```

### Acceptance Criteria
- Successfully parses all 7 form test cases
- Correctly maps form fields to ComplaintRecord
- Handles missing optional fields without errors
- Flags records with missing required fields

### Estimated Size: Small (1-2 days)

---

## PR-2: LLM Integration Foundation

### Description
Set up the Azure OpenAI integration and create the base infrastructure for LLM-powered features.

### Scope
- [ ] Create LLM client wrapper with:
  - Azure OpenAI configuration from environment
  - Retry logic with exponential backoff
  - Token usage tracking
  - Error handling and logging
- [ ] Define base prompt templates structure
- [ ] Create response parsing utilities
- [ ] Add structured output validation (ensure LLM returns valid data)
- [ ] Integration tests with mocked responses
- [ ] Update `.env.example` with Azure OpenAI variables

### Files to Create
```
src/
  llm/
    __init__.py
    client.py        # Azure OpenAI client wrapper
    prompts.py       # Prompt template management
    parsing.py       # Response parsing utilities
tests/
  test_llm_client.py
```

### Acceptance Criteria
- Client successfully connects to Azure OpenAI
- Retry logic handles transient failures
- Token usage is tracked and logged
- Invalid responses are caught and handled

### Estimated Size: Medium (2-3 days)

---

## PR-3: IMDRF Code Suggestion Service

### Description
Implement the core IMDRF code suggestion engine using LLM with few-shot prompting.

### Scope
- [ ] Design prompt template for IMDRF coding:
  - System prompt explaining IMDRF coding task
  - Include code hierarchy context
  - Few-shot examples (2-3 per code category)
  - Structured output format (JSON with codes, confidence, reasoning)
- [ ] Create `suggest_codes(complaint: ComplaintRecord) -> list[CodingSuggestion]` function
- [ ] Implement confidence scoring (0.0-1.0)
- [ ] Extract source text citations for each suggestion
- [ ] Validate suggested codes against IMDRF reference
- [ ] Handle edge cases (no clear codes, multiple interpretations)
- [ ] Unit tests with mocked LLM responses
- [ ] Integration tests with real LLM on test cases

### Files to Create
```
src/
  coding/
    __init__.py
    service.py       # Code suggestion service
    prompts.py       # IMDRF coding prompts
tests/
  test_coding_service.py
```

### Acceptance Criteria
- Returns valid IMDRF codes only (validated against reference)
- Confidence scores correlate with actual accuracy
- Source text citations point to relevant complaint text
- Handles ambiguous cases with multiple suggestions
- >70% accuracy on form test cases (Phase 1 baseline)

### Estimated Size: Large (3-4 days)

---

## PR-4: MDR Determination Service

### Description
Implement the MDR (Medical Device Report) determination logic to flag complaints that require FDA reporting.

### Scope
- [ ] Implement rules-based MDR criteria detection:
  - Death mentioned or implied
  - Serious injury (hospitalization, surgery, permanent damage)
  - Malfunction that could cause death/serious injury if recurs
- [ ] Create `determine_mdr(complaint: ComplaintRecord) -> MDRDetermination` function
- [ ] Add LLM-assisted severity assessment for edge cases
- [ ] Implement conservative defaults (flag uncertain cases)
- [ ] Extract MDR criteria evidence from narrative
- [ ] Unit tests covering all severity levels in test cases

### Files to Create
```
src/
  routing/
    __init__.py
    mdr.py           # MDR determination service
tests/
  test_mdr_service.py
```

### Acceptance Criteria
- 100% sensitivity (no false negatives on MDR-required cases)
- Clear reasoning for each determination
- Uncertain cases default to "requires review"
- All test cases correctly classified

### Estimated Size: Medium (2-3 days)

---

## PR-5: Audit Logging Foundation

### Description
Implement the audit logging infrastructure required for regulatory compliance.

### Scope
- [ ] Define audit event models:
  - `AuditEvent` base model (timestamp, user, action, resource)
  - `ComplaintCreatedEvent`
  - `CodingSuggestedEvent`
  - `CodingReviewedEvent` (for future use)
  - `MDRDeterminedEvent`
- [ ] Create audit logger interface:
  - `log_event(event: AuditEvent) -> None`
  - `get_events(resource_id: str) -> list[AuditEvent]`
- [ ] Implement JSON file-based storage (simple, sufficient for MVP)
- [ ] Ensure append-only behavior (no deletions/modifications)
- [ ] Add timestamps with timezone (UTC)
- [ ] Unit tests for audit logging

### Files to Create
```
src/
  audit/
    __init__.py
    models.py        # Audit event models
    logger.py        # Audit logger implementation
tests/
  test_audit.py
```

### Acceptance Criteria
- All events have complete timestamps and metadata
- Events are append-only (cannot be modified/deleted)
- Events can be retrieved by resource ID
- JSON storage is human-readable for debugging

### Estimated Size: Small (1-2 days)

---

## PR-6: Basic Review CLI

### Description
Build a simple CLI tool for reviewing AI suggestions, allowing human approval/rejection of IMDRF codes.

### Scope
- [ ] Create Typer CLI application with commands:
  - `review list` - Show pending complaints for review
  - `review show <id>` - Display complaint with AI suggestions
  - `review approve <id>` - Approve AI-suggested codes
  - `review modify <id>` - Modify codes before approval
  - `review reject <id>` - Reject and re-queue for manual coding
- [ ] Display complaint details in Rich formatted tables
- [ ] Show AI suggestions with confidence and reasoning
- [ ] Highlight source text for each suggestion
- [ ] Capture review decisions in audit log
- [ ] Integration tests for CLI commands

### Files to Create
```
src/
  cli/
    __init__.py
    review.py        # Review CLI commands
    display.py       # Rich display utilities
tests/
  test_review_cli.py
```

### Acceptance Criteria
- CLI displays complaints and suggestions clearly
- Human can approve, modify, or reject suggestions
- All decisions are captured in audit log
- Works with all form test cases

### Estimated Size: Medium (2-3 days)

---

## PR-7: End-to-End Pipeline

### Description
Wire together all components into a complete form processing pipeline and validate against test cases.

### Scope
- [ ] Create `process_form(raw_data: dict) -> ProcessingResult` orchestrator:
  1. Parse form submission
  2. Convert to ComplaintRecord
  3. Suggest IMDRF codes
  4. Determine MDR status
  5. Log all events to audit
  6. Return complete result
- [ ] Define `ProcessingResult` model with all outputs
- [ ] Add CLI command: `process form <file>` - Process a form submission
- [ ] Run evaluation on all form test cases
- [ ] Generate accuracy report
- [ ] Document end-to-end workflow

### Files to Create
```
src/
  pipeline/
    __init__.py
    forms.py         # Form processing pipeline
    models.py        # ProcessingResult model
tests/
  test_pipeline.py
  test_e2e_forms.py  # End-to-end tests with real LLM
```

### Acceptance Criteria
- Pipeline processes all 7 form test cases successfully
- Evaluation shows >70% coding accuracy (baseline for Phase 1)
- All pipeline steps are logged to audit
- CLI provides clear feedback on processing status

### Estimated Size: Medium (2-3 days)

---

## Sprint Summary

| PR | Title | Size | Dependencies | Priority |
|----|-------|------|--------------|----------|
| PR-1 | Form Ingestion Pipeline | S | None | P0 - Do First |
| PR-2 | LLM Integration Foundation | M | None | P0 |
| PR-5 | Audit Logging Foundation | S | None | P0 |
| PR-3 | IMDRF Code Suggestion Service | L | PR-2 | P1 |
| PR-4 | MDR Determination Service | M | PR-2 | P1 |
| PR-6 | Basic Review CLI | M | PR-1, PR-3, PR-5 | P1 |
| PR-7 | End-to-End Pipeline | M | All above | P2 - Do Last |

### Recommended Order

```
Week 1:
  PR-1 (Forms) ─────────────────────────────────────┐
  PR-2 (LLM Client) ──┬──> PR-3 (Coding Service) ──>├──> PR-6 (Review CLI)
                      └──> PR-4 (MDR Service) ──────┤
  PR-5 (Audit) ─────────────────────────────────────┘

Week 2:
  PR-6 (Review CLI) ──> PR-7 (E2E Pipeline) ──> Evaluation & Refinement
```

### Sprint Exit Criteria

- [ ] All 7 PRs merged
- [ ] Form processing pipeline functional end-to-end
- [ ] >70% coding accuracy on form test cases (baseline)
- [ ] 100% MDR sensitivity (no false negatives)
- [ ] All actions logged to audit trail
- [ ] CLI allows human review of suggestions
- [ ] Ready to begin Phase 2 (Enhance Coding Quality)

---

## Technical Notes

### Azure OpenAI Configuration

Required environment variables (add to `.env`):
```
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=<deployment-name>  # e.g., gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### Prompt Engineering Guidelines

1. **System prompt**: Define the IMDRF coding expert role and constraints
2. **Context**: Provide relevant IMDRF code hierarchy for the complaint type
3. **Few-shot examples**: 2-3 examples per major code category
4. **Output format**: Structured JSON with code, confidence, source_text, reasoning
5. **Constraints**: Only return codes from the valid IMDRF reference

### Testing Strategy

- **Unit tests**: Mock LLM responses for deterministic testing
- **Integration tests**: Real LLM calls on subset of test cases
- **Evaluation**: Run full test suite and generate accuracy metrics

---

## Notes

- Phase 0 foundation is complete: models, IMDRF codes, test cases, evaluation framework
- Start with forms because they're most structured (lowest extraction difficulty)
- Focus on coding accuracy first; UI polish comes in later phases
- Conservative MDR determination is critical (100% sensitivity required)
- Audit logging must be in place from the start for compliance
