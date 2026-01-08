# Current Sprint: Phase 0 - Foundation

## Sprint Goal

Establish the foundational reference data and test infrastructure needed to develop and validate the AI-powered complaint handling system.

**Sprint Duration:** 2 weeks

---

## PR-1: IMDRF Code Reference Database

### Description
Build a structured, queryable database of IMDRF codes that will serve as the source of truth for code validation and LLM prompt construction.

### Scope
- [ ] Download and parse IMDRF Annex A (Device Problem Codes)
- [ ] Download and parse IMDRF Annex C (Patient Problem Codes)
- [ ] Create JSON schema for hierarchical code structure
- [ ] Build code reference files with:
  - Code ID
  - Code name
  - Parent code (for hierarchy)
  - Full path (e.g., "Material Problem > Material Integrity > Crack")
  - Description/definition
  - Examples (where available)
- [ ] Create Python utility functions:
  - `get_code_by_id(code_id: str) -> IMDRFCode`
  - `get_children(code_id: str) -> list[IMDRFCode]`
  - `get_ancestors(code_id: str) -> list[IMDRFCode]`
  - `search_codes(query: str) -> list[IMDRFCode]`
  - `validate_code(code_id: str) -> bool`
- [ ] Unit tests for utility functions

### Files to Create
```
src/
  imdrf/
    __init__.py
    models.py          # Pydantic models for IMDRF codes
    codes.py           # Utility functions
data/
  imdrf/
    annex_a_device_problems.json
    annex_c_patient_problems.json
tests/
  test_imdrf_codes.py
```

### Acceptance Criteria
- All Annex A and Annex C codes parsed and validated
- Hierarchy traversal works correctly
- Search returns relevant codes
- 100% test coverage on utility functions

### Estimated Size: Medium (2-3 days)

---

## PR-2: Complaint Data Models

### Description
Define the canonical data models for complaints that will be used throughout the system, from intake through coding and review.

### Scope
- [ ] Define `ComplaintRecord` Pydantic model with fields:
  - `complaint_id`: Unique identifier
  - `intake_channel`: Enum (FORM, EMAIL, CALL, LETTER, SALES_REP)
  - `received_date`: datetime
  - `device_info`: DeviceInfo model
  - `patient_info`: PatientInfo model (optional, anonymized)
  - `event_info`: EventInfo model
  - `reporter_info`: ReporterInfo model
  - `narrative`: str (raw complaint text)
  - `extracted_fields`: dict (structured extraction results)
  - `status`: Enum (NEW, EXTRACTED, CODED, REVIEWED, CLOSED)
- [ ] Define `DeviceInfo` model:
  - `device_name`: str
  - `manufacturer`: str
  - `model_number`: str (optional)
  - `serial_number`: str (optional)
  - `lot_number`: str (optional)
  - `device_type`: Enum (IMPLANTABLE, DIAGNOSTIC, CONSUMABLE, SAMD, OTHER)
- [ ] Define `EventInfo` model:
  - `event_date`: date (optional)
  - `event_description`: str
  - `patient_outcome`: str (optional)
  - `device_outcome`: str (optional)
- [ ] Define `CodingSuggestion` model:
  - `code_id`: str
  - `code_name`: str
  - `code_type`: Enum (DEVICE_PROBLEM, PATIENT_PROBLEM)
  - `confidence`: float (0.0-1.0)
  - `source_text`: str (text that triggered this suggestion)
  - `reasoning`: str (explanation of why this code applies)
- [ ] Define `CodingDecision` model:
  - `complaint_id`: str
  - `suggested_codes`: list[CodingSuggestion]
  - `approved_codes`: list[str] (after human review)
  - `rejected_codes`: list[str]
  - `added_codes`: list[str] (codes added by reviewer not suggested by AI)
  - `reviewer_id`: str
  - `review_timestamp`: datetime
  - `review_notes`: str (optional)
- [ ] Define `MDRDetermination` model:
  - `requires_mdr`: bool
  - `mdr_criteria_met`: list[str]
  - `confidence`: float
  - `reasoning`: str
- [ ] JSON serialization/deserialization tests

### Files to Create
```
src/
  models/
    __init__.py
    complaint.py       # ComplaintRecord, DeviceInfo, EventInfo, etc.
    coding.py          # CodingSuggestion, CodingDecision
    mdr.py             # MDRDetermination
    enums.py           # All enums (IntakeChannel, DeviceType, etc.)
tests/
  test_models.py
```

### Acceptance Criteria
- All models have complete type hints
- Validation rules enforced (e.g., confidence 0-1)
- JSON round-trip works correctly
- Example instances can be created for all models

### Estimated Size: Small (1-2 days)

---

## PR-3: Synthetic Test Data - Online Forms

### Description
Create realistic synthetic complaint data for the online form intake channel, covering various device types and severity levels.

### Scope
- [ ] Create 7 synthetic form submissions:
  1. **Implantable - Death** (pacemaker failure leading to death)
  2. **Implantable - Serious Injury** (hip implant dislocation)
  3. **Diagnostic - Malfunction with potential harm** (blood glucose meter false low)
  4. **Diagnostic - Malfunction no harm** (imaging artifact, caught before diagnosis)
  5. **Consumable - Quality issue** (catheter packaging damaged)
  6. **SaMD - Software issue** (dosing calculator error)
  7. **User error** (insulin pump programmed incorrectly by patient)

- [ ] Each test case includes:
  - Raw form data (as submitted)
  - Expected extracted `ComplaintRecord`
  - Expected IMDRF codes (with rationale)
  - Expected MDR determination
  - Difficulty rating
  - Edge case notes

- [ ] Create test case loader utility

### Files to Create
```
data/
  test_cases/
    forms/
      form_001_pacemaker_death.json
      form_002_hip_implant_injury.json
      form_003_glucose_meter_malfunction.json
      form_004_imaging_artifact.json
      form_005_catheter_packaging.json
      form_006_dosing_calculator.json
      form_007_insulin_pump_user_error.json
      _manifest.json    # Index of all test cases
src/
  testing/
    __init__.py
    test_case_loader.py
tests/
  test_case_loader_test.py
```

### Acceptance Criteria
- All 7 test cases created with complete ground truth
- Test cases cover all severity levels (death, serious injury, malfunction, user error)
- Test cases cover all device types
- At least 2 test cases require multiple IMDRF codes
- Loader utility can retrieve test cases by ID or filter by attributes

### Estimated Size: Medium (2-3 days)

---

## PR-4: Synthetic Test Data - Emails

### Description
Create realistic synthetic complaint data for the email intake channel, with varying formality levels and sender types.

### Scope
- [ ] Create 8 synthetic email complaints:
  1. **Physician - Formal report** (detailed clinical language, implant issue)
  2. **Physician - Brief notification** (terse, assumes reader knowledge)
  3. **Patient - Detailed complaint** (emotional, lengthy narrative)
  4. **Patient - Angry complaint** (frustrated tone, demands action)
  5. **Patient family member** (reporting on behalf of patient)
  6. **Sales rep - Forwarded customer email** (includes chain)
  7. **Sales rep - Direct report** (structured internal format)
  8. **Nurse/clinical staff** (incident report style)

- [ ] Each email includes:
  - Full email content (headers, body, signature)
  - Attachments metadata (if applicable)
  - Expected extracted `ComplaintRecord`
  - Expected IMDRF codes
  - Expected MDR determination
  - Extraction challenges (what makes this hard)

- [ ] Vary severity levels across emails

### Files to Create
```
data/
  test_cases/
    emails/
      email_001_physician_formal.json
      email_002_physician_brief.json
      email_003_patient_detailed.json
      email_004_patient_angry.json
      email_005_family_member.json
      email_006_sales_rep_forward.json
      email_007_sales_rep_direct.json
      email_008_nurse_incident.json
      _manifest.json
```

### Acceptance Criteria
- All 8 test cases created with complete ground truth
- Emails have realistic structure (headers, signatures, formatting)
- Mix of formal and informal language
- At least one email chain (forwarded/replied)
- Extraction challenges documented for each

### Estimated Size: Medium (2-3 days)

---

## PR-5: Synthetic Test Data - Calls & Documents

### Description
Create synthetic test data for call transcripts and scanned documents (letters).

### Scope

**Call Transcripts (5 cases):**
- [ ] Create 5 synthetic call transcripts:
  1. **Clear, structured caller** (organized, answers questions directly)
  2. **Rambling caller** (goes off-topic, needs redirection)
  3. **Emotional/upset caller** (crying, angry, hard to follow)
  4. **Technical caller** (healthcare professional, uses jargon)
  5. **Language barrier** (simple vocabulary, some confusion)

- [ ] Each transcript includes:
  - Full conversation (agent and caller turns)
  - Call metadata (duration, date, caller ID)
  - Expected extracted `ComplaintRecord`
  - Expected IMDRF codes
  - Extraction challenges

**Scanned Documents (4 cases):**
- [ ] Create 4 synthetic document complaints:
  1. **Typed formal letter** (patient or attorney)
  2. **Typed incident report** (hospital form)
  3. **Handwritten note** (brief, legible)
  4. **Mixed document** (typed with handwritten annotations)

- [ ] Each document includes:
  - Document text content (simulating OCR output)
  - Document metadata
  - Expected extracted `ComplaintRecord`
  - Expected IMDRF codes
  - OCR challenges (if applicable)

### Files to Create
```
data/
  test_cases/
    calls/
      call_001_structured_caller.json
      call_002_rambling_caller.json
      call_003_emotional_caller.json
      call_004_technical_caller.json
      call_005_language_barrier.json
      _manifest.json
    documents/
      doc_001_typed_letter.json
      doc_002_incident_report.json
      doc_003_handwritten_note.json
      doc_004_mixed_annotations.json
      _manifest.json
```

### Acceptance Criteria
- All 9 test cases created with complete ground truth
- Call transcripts have realistic conversation flow
- Documents represent realistic OCR output quality
- Challenges documented for each test case

### Estimated Size: Medium (2-3 days)

---

## PR-6: Test Infrastructure & Evaluation Framework

### Description
Build the infrastructure to load test cases, run evaluations, and measure accuracy metrics.

### Scope
- [ ] Create unified test case loader:
  - Load all test cases from all channels
  - Filter by channel, device type, severity, difficulty
  - Return as `ComplaintRecord` with ground truth
- [ ] Create evaluation framework:
  - `evaluate_extraction(predicted: ComplaintRecord, expected: ComplaintRecord) -> ExtractionMetrics`
  - `evaluate_coding(predicted: list[str], expected: list[str]) -> CodingMetrics`
  - `evaluate_mdr(predicted: MDRDetermination, expected: MDRDetermination) -> MDRMetrics`
- [ ] Define metrics models:
  - `ExtractionMetrics`: field-level accuracy, completeness
  - `CodingMetrics`: precision, recall, F1, exact match
  - `MDRMetrics`: sensitivity (must be 100%), specificity, false positive rate
- [ ] Create summary report generator:
  - Aggregate metrics across test cases
  - Breakdown by channel, device type, severity
  - Export to JSON and markdown

### Files to Create
```
src/
  evaluation/
    __init__.py
    loader.py          # Unified test case loader
    metrics.py         # Metric calculation functions
    models.py          # ExtractionMetrics, CodingMetrics, etc.
    reporter.py        # Report generation
tests/
  test_evaluation.py
```

### Acceptance Criteria
- Can load all test cases from all channels
- Metrics calculations are correct (verified with manual examples)
- Report clearly shows performance breakdown
- Framework ready for Phase 1 development

### Estimated Size: Medium (2-3 days)

---

## PR-7: Development Environment & CI Setup

### Description
Set up the development environment, project structure, and continuous integration pipeline.

### Scope
- [ ] Initialize Python project structure:
  - `pyproject.toml` with dependencies
  - Ruff configuration
  - pytest configuration
  - Pre-commit hooks
- [ ] Create directory structure as defined in PRs above
- [ ] Set up GitHub Actions CI:
  - Lint check (ruff)
  - Type check (mypy)
  - Unit tests (pytest)
  - Coverage reporting
- [ ] Create `.env.example` with required environment variables
- [ ] Create initial README with:
  - Project overview
  - Setup instructions
  - Development workflow

### Files to Create
```
pyproject.toml
.pre-commit-config.yaml
.github/
  workflows/
    ci.yml
.env.example
README.md
src/
  __init__.py
tests/
  __init__.py
  conftest.py
```

### Acceptance Criteria
- `pip install -e .` works
- `ruff check .` passes
- `pytest` runs successfully
- CI pipeline runs on PR
- README has clear setup instructions

### Estimated Size: Small (1 day)

---

## Sprint Summary

| PR | Title | Size | Dependencies | Priority |
|----|-------|------|--------------|----------|
| PR-7 | Development Environment & CI Setup | S | None | P0 - Do First |
| PR-2 | Complaint Data Models | S | PR-7 | P0 |
| PR-1 | IMDRF Code Reference Database | M | PR-7 | P0 |
| PR-3 | Synthetic Test Data - Forms | M | PR-2 | P1 |
| PR-4 | Synthetic Test Data - Emails | M | PR-2 | P1 |
| PR-5 | Synthetic Test Data - Calls & Docs | M | PR-2 | P1 |
| PR-6 | Test Infrastructure & Evaluation | M | PR-1, PR-2 | P1 |

### Recommended Order

```
Week 1:
  PR-7 (Environment) ──┬──> PR-2 (Models) ──┬──> PR-3 (Forms)
                       │                    ├──> PR-4 (Emails)
                       │                    └──> PR-5 (Calls/Docs)
                       │
                       └──> PR-1 (IMDRF Codes) ──┐
                                                 │
Week 2:                                          │
  PR-3, PR-4, PR-5 (parallel) ───────────────────┴──> PR-6 (Evaluation)
```

### Sprint Exit Criteria

- [ ] All 7 PRs merged
- [ ] 24+ synthetic test cases created across all channels
- [ ] IMDRF code database complete and queryable
- [ ] Evaluation framework ready for Phase 1
- [ ] CI pipeline green
- [ ] Ready to begin Phase 1 (Single Channel MVP)

---

## Notes

- Test cases should feel realistic—use medical terminology appropriately
- Ground truth codes may have multiple valid answers; document acceptable alternatives
- Keep PHI completely synthetic (no real patient data patterns)
- IMDRF codes should come from actual IMDRF documentation
