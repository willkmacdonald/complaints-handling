# Current Sprint: Phase 2 - Enhance Coding Quality

## Sprint Goal

Improve IMDRF code suggestion accuracy from ~70% baseline to >80% acceptance rate through prompt engineering (few-shot examples + chain-of-thought) and data-driven confidence calibration.

**Success Gate:** >80% F1 score on test cases

---

## Design Principles

See `.claude/CLAUDE.md` for full details. Key principle for Phase 2:

**Cloud Migration Readiness**: Use storage abstractions so file-based implementations can swap to Azure Blob/Cosmos later. Keep business logic in services, not CLI.

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

## PR-1: Form Ingestion Pipeline ✅

### Description
Build the pipeline to ingest online form submissions and convert them to the canonical ComplaintRecord format.

### Scope
- [x] Define `FormSubmission` Pydantic model representing raw form data:
  - Submitter information (name, email, phone, relationship to patient)
  - Device information (name, manufacturer, model, serial, lot)
  - Event details (date, description, patient outcome, device outcome)
  - Attachments metadata
  - Submission timestamp
- [x] Create `parse_form_submission(raw_data: dict) -> FormSubmission` function
- [x] Create `form_to_complaint(form: FormSubmission) -> ComplaintRecord` conversion function
- [x] Handle missing/optional fields gracefully
- [x] Flag incomplete submissions that need follow-up
- [x] Unit tests with form test cases (26 tests passing)

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

## PR-2: LLM Integration Foundation ✅

### Description
Set up the Azure OpenAI integration and create the base infrastructure for LLM-powered features.

### Scope
- [x] Create LLM client wrapper with:
  - Azure OpenAI configuration from environment
  - Retry logic with exponential backoff
  - Token usage tracking
  - Error handling and logging
- [x] Define base prompt templates structure
- [x] Create response parsing utilities
- [x] Add structured output validation (ensure LLM returns valid data)
- [x] Integration tests with mocked responses
- [x] Update `.env.example` with Azure OpenAI variables

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

## PR-3: IMDRF Code Suggestion Service ✅

### Description
Implement the core IMDRF code suggestion engine using LLM with few-shot prompting.

### Scope
- [x] Design prompt template for IMDRF coding:
  - System prompt explaining IMDRF coding task
  - Include code hierarchy context
  - Few-shot examples (2-3 per code category)
  - Structured output format (JSON with codes, confidence, reasoning)
- [x] Create `suggest_codes(complaint: ComplaintRecord) -> list[CodingSuggestion]` function
- [x] Implement confidence scoring (0.0-1.0)
- [x] Extract source text citations for each suggestion
- [x] Validate suggested codes against IMDRF reference
- [x] Handle edge cases (no clear codes, multiple interpretations)
- [x] Unit tests with mocked LLM responses
- [x] Integration tests with real LLM on test cases

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

## PR-4: MDR Determination Service ✅

### Description
Implement the MDR (Medical Device Report) determination logic to flag complaints that require FDA reporting.

### Scope
- [x] Implement rules-based MDR criteria detection:
  - Death mentioned or implied
  - Serious injury (hospitalization, surgery, permanent damage)
  - Malfunction that could cause death/serious injury if recurs
- [x] Create `determine_mdr(complaint: ComplaintRecord) -> MDRDetermination` function
- [x] Add LLM-assisted severity assessment for edge cases
- [x] Implement conservative defaults (flag uncertain cases)
- [x] Extract MDR criteria evidence from narrative
- [x] Unit tests covering all severity levels in test cases

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

## PR-5: Audit Logging Foundation ✅

### Description
Implement the audit logging infrastructure required for regulatory compliance.

### Scope
- [x] Define audit event models:
  - `AuditEvent` base model (timestamp, user, action, resource)
  - `ComplaintCreatedEvent`
  - `CodingSuggestedEvent`
  - `CodingReviewedEvent` (for future use)
  - `MDRDeterminedEvent`
- [x] Create audit logger interface:
  - `log_event(event: AuditEvent) -> None`
  - `get_events(resource_id: str) -> list[AuditEvent]`
- [x] Implement JSON file-based storage (simple, sufficient for MVP)
- [x] Ensure append-only behavior (no deletions/modifications)
- [x] Add timestamps with timezone (UTC)
- [x] Unit tests for audit logging

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

## PR-6: Basic Review CLI ✅

### Description
Build a simple CLI tool for reviewing AI suggestions, allowing human approval/rejection of IMDRF codes.

### Scope
- [x] Create Typer CLI application with commands:
  - `review list` - Show pending complaints for review
  - `review show <id>` - Display complaint with AI suggestions
  - `review approve <id>` - Approve AI-suggested codes
  - `review modify <id>` - Modify codes before approval
  - `review reject <id>` - Reject and re-queue for manual coding
- [x] Display complaint details in Rich formatted tables
- [x] Show AI suggestions with confidence and reasoning
- [x] Highlight source text for each suggestion
- [x] Capture review decisions in audit log
- [x] Integration tests for CLI commands

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

## PR-7: End-to-End Pipeline ✅

### Description
Wire together all components into a complete form processing pipeline and validate against test cases.

### Scope
- [x] Create `process_form(raw_data: dict) -> ProcessingResult` orchestrator:
  1. Parse form submission
  2. Convert to ComplaintRecord
  3. Suggest IMDRF codes
  4. Determine MDR status
  5. Log all events to audit
  6. Return complete result
- [x] Define `ProcessingResult` model with all outputs
- [x] Add CLI command: `process form <file>` - Process a form submission
- [x] Run evaluation on all form test cases
- [x] Generate accuracy report
- [x] Document end-to-end workflow

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

| PR | Title | Size | Dependencies | Status |
|----|-------|------|--------------|--------|
| PR-1 | Form Ingestion Pipeline | S | None | ✅ Complete |
| PR-2 | LLM Integration Foundation | M | None | ✅ Complete |
| PR-5 | Audit Logging Foundation | S | None | ✅ Complete |
| PR-3 | IMDRF Code Suggestion Service | L | PR-2 | ✅ Complete |
| PR-4 | MDR Determination Service | M | PR-2 | ✅ Complete |
| PR-6 | Basic Review CLI | M | PR-1, PR-3, PR-5 | ✅ Complete |
| PR-7 | End-to-End Pipeline | M | All above | ✅ Complete |

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

- [x] All 7 PRs merged
- [x] Form processing pipeline functional end-to-end
- [x] >70% coding accuracy on form test cases (baseline)
- [x] 100% MDR sensitivity (no false negatives)
- [x] All actions logged to audit trail
- [x] CLI allows human review of suggestions
- [x] Ready to begin Phase 2 (Enhance Coding Quality)

**Phase 1 Status: COMPLETE** - 334 tests passing, 79% code coverage

---

# Phase 2 PRs

## PR-8: Evaluation Pipeline Runner ✅

### Description
Create a CLI-driven evaluation pipeline that runs the coding service against all test cases with live LLM and generates accuracy reports.

### Scope
- [x] Create `EvaluationRunMetadata` model (run_id, timestamp, strategy, model, filters, token stats)
- [x] Create `TestCaseEvaluationResult` model (per-test-case results with predicted codes, metrics)
- [x] Create `EvaluationRunResult` model (aggregates metadata + all results + report)
- [x] Implement `EvaluationRunner` class that runs coding service on test cases
- [x] Create storage interface and file-based implementation for evaluation runs
- [x] Add CLI commands:
  - `evaluate run --strategy <strategy>` - Run evaluation
  - `evaluate report <run_id>` - Generate report for a run
  - `evaluate history` - List past evaluation runs

### Files Created/Modified
```
src/evaluation/
  models.py         # EvaluationRunMetadata, TestCaseEvaluationResult, EvaluationRunResult
  runner.py         # run_evaluation(), _evaluate_test_case()
  storage.py        # EvaluationStorage protocol, FileEvaluationStorage
src/cli/
  evaluate.py       # run, report, history commands
  __init__.py       # Registered evaluate subcommand
```

### Acceptance Criteria
- [x] Can run evaluation against all 24 test cases with live LLM
- [x] Results persisted via storage interface (file-based for now)
- [x] Report shows F1/precision/recall by difficulty, device type, channel
- [x] CLI provides clear progress feedback during runs
- [x] 52 evaluation tests passing

### Estimated Size: Medium

---

## PR-9: Few-Shot + Chain-of-Thought Prompting ✅

### Description
Enhance the IMDRF coding prompt with few-shot examples and structured chain-of-thought reasoning.

### Scope
- [x] Create `IMDRF_CODING_TEMPLATE_V2` with chain-of-thought structure:
  - Step 1: Device Classification
  - Step 2: Identify Device Problems
  - Step 3: Identify Patient Problems
  - Step 4: Code Selection
- [x] Curate 5 few-shot examples from existing test cases
- [x] Create `data/prompts/few_shot_examples.json` with full reasoning chains
- [x] Add `load_few_shot_examples()` function
- [x] Update `CodingService` with `strategy` parameter
- [x] `PromptStrategy` enum moved to `src/models/enums.py`: ZERO_SHOT, FEW_SHOT, CHAIN_OF_THOUGHT, FEW_SHOT_COT

### Files Created/Modified
```
src/llm/
  prompts.py            # Added IMDRF_CODING_TEMPLATE_V2, FewShotExample model,
                        # load_few_shot_examples(), format_few_shot_examples(),
                        # format_few_shot_as_messages()
src/coding/
  service.py            # Added strategy parameter, _build_prompt_messages()
src/models/
  enums.py              # Added PromptStrategy enum
src/evaluation/
  models.py             # Re-exports PromptStrategy for backward compatibility
  runner.py             # Passes strategy to CodingService
data/prompts/
  few_shot_examples.json  # 5 curated examples with full CoT reasoning
tests/
  test_coding_service.py  # Added TestPromptStrategies, TestFewShotExamples
```

### Acceptance Criteria
- [x] New prompt template includes 4-step reasoning structure
- [x] 5 diverse examples cover different device types and coding patterns
- [x] CodingService can toggle between zero-shot, few-shot, CoT, and combined modes
- [x] 370 tests passing, 79% coverage

### Estimated Size: Medium

---

## PR-10: Confidence Calibration System ✅

### Description
Analyze confidence-accuracy correlation and find optimal confidence thresholds.

### Scope
- [x] Create calibration data models (SuggestionOutcome, CalibrationBin, CalibrationAnalysis)
- [x] Implement `calculate_calibration_error()` - compute ECE/MCE metrics
- [x] Implement `find_optimal_threshold()` - find threshold maximizing F1
- [x] Create `CalibrationConfig` for storing optimal thresholds
- [x] Add CLI command: `evaluate calibration <run_id>`
- [ ] Update `CodingService` to load and use calibrated thresholds (deferred to PR-12)

### Files Created/Modified
```
src/evaluation/
  calibration.py        # NEW: Calibration functions and models
src/cli/
  evaluate.py           # Add calibration subcommand
tests/
  test_calibration.py   # NEW: 21 tests for calibration
```

### Acceptance Criteria
- [x] ECE and MCE metrics calculated correctly
- [x] Optimal threshold found that maximizes F1 score
- [ ] Calibration config integrates with CodingService (PR-12)

### Estimated Size: Small

---

## PR-11: Ablation Testing Framework ✅

### Description
Framework for comparing prompt strategies and measuring statistical significance.

### Scope
- [x] Create comparison models (StrategyComparison, AblationReport)
- [x] Implement `run_ablation_test()` - run all strategies on same test cases
- [x] Implement `compare_strategies()` - statistical comparison of two runs
- [x] Add paired t-test for significance
- [x] Add CLI commands: `evaluate ablation`, `evaluate compare`, `evaluate ablation-report`, `evaluate ablation-history`
- [x] Generate cost-effectiveness analysis (F1 per 1K tokens)

### Files Created/Modified
```
src/evaluation/
  ablation.py           # NEW: Ablation functions and models
src/cli/
  evaluate.py           # Add ablation, compare subcommands
pyproject.toml          # Add scipy as optional [stats] dependency
tests/
  test_ablation.py      # NEW: 16 tests for ablation
```

### Acceptance Criteria
- [x] Can run ablation test across all 4 prompt strategies
- [x] Statistical significance calculated for each comparison (requires scipy)
- [x] Cost-effectiveness metric (F1/1K tokens) helps choose practical strategy

### Estimated Size: Medium

---

## PR-12: Integrate & Validate Improvements

### Description
Run final evaluation, select best strategy, and validate >80% target.

### Scope
- [ ] Run ablation test with all strategies
- [ ] Analyze results and select optimal strategy
- [ ] Run confidence calibration on optimal strategy
- [ ] Update default CodingService configuration
- [ ] Document final accuracy metrics

### Files to Modify
```
src/coding/service.py       # Update defaults
current-sprint.md           # Document results
```

### Success Gate
- **Primary**: >80% F1 score on test cases
- **Secondary**: Confidence calibration ECE < 10%
- **Tertiary**: Token cost increase < 2x baseline

### Estimated Size: Small

---

## Phase 2 Sprint Summary

| PR | Title | Size | Dependencies | Status |
|----|-------|------|--------------|--------|
| PR-8 | Evaluation Pipeline Runner | M | None | ✅ Complete |
| PR-9 | Few-Shot + CoT Prompting | M | None | ✅ Complete |
| PR-10 | Confidence Calibration | S | PR-8 | ✅ Complete |
| PR-11 | Ablation Testing | M | PR-8, PR-9 | ✅ Complete |
| PR-12 | Integrate & Validate | S | All above | Pending |

### Execution Order

```
PR-8 (Evaluation Runner) ──┬──> PR-10 (Calibration) ──┐
                           │                          ├──> PR-12 (Integrate)
PR-9 (Prompting) ──────────┴──> PR-11 (Ablation) ─────┘
```

PR-8 and PR-9 can proceed in parallel.

---

# Phase 1 Reference (Complete)

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
