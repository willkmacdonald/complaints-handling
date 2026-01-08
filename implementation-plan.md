# AI-Powered Medical Device Complaint Handling System

## Implementation Plan

### Executive Summary

This document outlines a phased implementation plan for an AI-powered complaint handling system for medical device companies. The system will ingest complaints from multiple channels, extract structured information, suggest IMDRF codes, and support human review—all while maintaining FDA regulatory compliance.

---

## Core Architectural Principle

> **AI as Decision Support, Not Decision Maker**

Given FDA regulations (21 CFR Part 820.198 for Complaint Files, 21 CFR Part 11 for Electronic Records), this system **must** operate as human-in-the-loop. The AI suggests IMDRF codes; humans approve. This isn't a limitation—it's the foundational architecture that enables regulatory compliance.

---

## Regulatory Context

### Applicable Regulations

| Regulation | Relevance |
|------------|-----------|
| **21 CFR Part 820.198** | Complaint file requirements for medical device manufacturers |
| **21 CFR Part 11** | Electronic records and signatures requirements |
| **21 CFR Part 803** | Medical Device Reporting (MDR) requirements |
| **FDA GMLP Guidance** | Good Machine Learning Practice for AI/ML in medical devices |

### Regulatory Requirements for AI Systems

1. **Audit Trails** — Complete record of who did what, when, and why
2. **Electronic Signatures** — For approvals and sign-offs
3. **System Validation** — IQ/OQ/PQ documentation
4. **Access Controls** — Role-based authentication and authorization
5. **Human Oversight** — AI decisions require human review and approval
6. **Explainability** — Must explain why codes were suggested
7. **Confidence Scoring** — System must indicate certainty level

---

## IMDRF Coding Standard

The International Medical Device Regulators Forum (IMDRF) provides standardized terminology:

| Annex | Description | Example |
|-------|-------------|---------|
| **Annex A** | Device Problem Codes | Material integrity, software issue, mechanical failure |
| **Annex B** | Component Codes | Battery, sensor, connector |
| **Annex C** | Patient Problem Codes | Injury, infection, clinical outcomes |
| **Annex D** | Evaluation Codes | Investigation findings |

Codes are hierarchical. Example: `Material Problem > Material Integrity > Crack` maps to `A0601 > A060102 > A06010201`

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INTAKE LAYER                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Online  │ │  Email   │ │   Call   │ │  Letter  │ │ Sales Rep│  │
│  │  Forms   │ │  Parser  │ │   STT    │ │   OCR    │ │  Reports │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
└───────┼────────────┼────────────┼────────────┼────────────┼────────┘
        │            │            │            │            │
        └────────────┴────────────┼────────────┴────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       EXTRACTION LAYER                              │
│  • Extract complaint elements (device, patient, event, dates)       │
│  • Normalize to canonical complaint record                          │
│  • Flag missing required fields                                     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CODING LAYER                                │
│  • Map narrative to IMDRF codes (multi-label)                       │
│  • Confidence scoring for each suggestion                           │
│  • Explainability — cite source text for each code                  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    HUMAN REVIEW LAYER                               │
│  • Present AI suggestions with confidence + explanations            │
│  • Accept / Modify / Reject workflow                                │
│  • Capture decisions for audit trail                                │
│  • Feed corrections back for improvement                            │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ROUTING LAYER                                │
│  • MDR determination (death, serious injury, malfunction)           │
│  • Route to appropriate review queue                                │
│  • Priority/urgency flagging                                        │
│  • QMS integration                                                  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   AUDIT & COMPLIANCE LAYER                          │
│  • Immutable audit log of all actions                               │
│  • Report generation for FDA inspections                            │
│  • Trend analysis and signal detection                              │
│  • Performance monitoring and drift detection                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phased Implementation

### Phase 0: Foundation

**Duration:** Weeks 1-2

**Objectives:**
- Obtain IMDRF code documentation and build structured reference database
- Create synthetic test complaint dataset (25-30 examples)
- Set up development environment
- Define success metrics

**Deliverables:**
- [ ] IMDRF code database with hierarchies (JSON/database)
- [ ] Synthetic test dataset with ground truth codes
- [ ] Development environment setup
- [ ] Success metrics documentation

**Success Gate:** Validated set of test cases with "ground truth" codes covering all channels, device types, and severity levels

---

### Phase 1: Single Channel MVP (Structured Forms)

**Duration:** Weeks 3-6

**Objectives:**
- Process online form submissions (most structured = lowest risk)
- Build extraction from form fields to canonical complaint record
- Implement IMDRF code suggestion using LLM with few-shot prompting
- Build simple review UI showing suggestions with confidence
- Basic audit logging

**Deliverables:**
- [ ] Form ingestion pipeline
- [ ] Extraction service (form → complaint record)
- [ ] IMDRF coding service with LLM
- [ ] Basic review UI
- [ ] Audit logging foundation

**Success Gate:** >80% of suggested codes accepted by human reviewer without modification

---

### Phase 2: Enhance Coding Quality

**Duration:** Weeks 7-10

**Objectives:**
- Add explainability — show which text triggered each code suggestion
- Implement confidence calibration (high confidence = high accuracy)
- Build feedback loop — capture human corrections
- Add multi-code suggestions (device + patient problem combinations)
- Improve prompts based on Phase 1 learnings

**Deliverables:**
- [ ] Explainability feature (text highlighting)
- [ ] Calibrated confidence scores
- [ ] Feedback capture and reporting
- [ ] Multi-code suggestion support
- [ ] Refined LLM prompts

**Success Gate:** Reduce average human review time by 30% while maintaining accuracy

---

### Phase 3: Expand Intake Channels

**Duration:** Weeks 11-16

**Objectives:**

**Email Processing:**
- Parse email structure (sender, date, body, attachments)
- Extract complaint information from unstructured narrative
- Handle various formats (physician reports, patient complaints, sales rep forwards)
- Distinguish complaints from general inquiries

**Document Processing (Letters, Faxes):**
- OCR for scanned documents
- Handle handwritten notes (with human transcription fallback)
- Extract from PDF forms

**Call Center Integration:**
- Integrate with speech-to-text service
- Process call transcripts
- Handle conversational structure

**Deliverables:**
- [ ] Email parsing and extraction service
- [ ] OCR pipeline for documents
- [ ] Speech-to-text integration
- [ ] Unified complaint record from all channels

**Success Gate:** All channels feed into unified complaint record with >75% extraction accuracy

---

### Phase 4: Routing & MDR Determination

**Duration:** Weeks 17-20

**Objectives:**
- Implement rules engine for MDR criteria
- AI-assisted severity assessment
- Smart routing to appropriate review queues
- Priority/urgency flagging
- Integration with downstream QMS

**MDR Criteria (requires reporting):**
- Death
- Serious injury
- Malfunction that could cause death or serious injury if it recurs

**Deliverables:**
- [ ] MDR determination engine (rules + AI-assisted)
- [ ] Severity assessment model
- [ ] Routing rules and queue management
- [ ] QMS integration API

**Success Gate:** 100% of MDR-required complaints correctly flagged (zero false negatives; false positives acceptable)

---

### Phase 5: Validation & Compliance

**Duration:** Weeks 21-26

**Objectives:**

**System Validation (21 CFR Part 11):**
- Installation Qualification (IQ)
- Operational Qualification (OQ)
- Performance Qualification (PQ)
- Complete validation documentation

**AI-Specific Validation:**
- Document training data sources and quality
- Validate performance across device types and complaint types
- Edge case testing with adversarial examples
- Bias testing for systematic errors

**Audit Trail Completeness:**
- Immutable storage verification
- Retention policy implementation
- Access control validation

**Deliverables:**
- [ ] IQ/OQ/PQ documentation
- [ ] AI validation report
- [ ] Audit trail validation
- [ ] Access control documentation
- [ ] Complete validation package

**Success Gate:** Pass internal QA audit simulating FDA inspection

---

### Phase 6: Production & Continuous Improvement

**Duration:** Ongoing

**Objectives:**
- Deploy to production with monitoring
- Performance dashboard
- Drift detection and alerting
- Periodic revalidation
- Change control process for improvements

**Deliverables:**
- [ ] Production deployment
- [ ] Monitoring dashboard
- [ ] Drift detection system
- [ ] Revalidation schedule
- [ ] Change control procedures

**Success Gate:** Stable production operation with defined SLAs

---

## Test Data Strategy

### Synthetic Complaint Dataset Requirements

The test dataset must cover multiple dimensions to ensure comprehensive testing:

#### By Intake Channel
| Channel | Count | Notes |
|---------|-------|-------|
| Online forms | 5-7 | Complete, partial, inconsistent data |
| Emails | 8-10 | From patients, physicians, sales reps; formal and informal |
| Call transcripts | 5-7 | Clear narratives, meandering conversations |
| Letters/documents | 3-5 | Typed, handwritten notes |

#### By Device Type
- Implantable devices (pacemakers, joint replacements)
- Diagnostic equipment (imaging systems, lab analyzers)
- Consumables (catheters, syringes, test strips)
- Software as Medical Device (SaMD)

#### By Severity Level
| Severity | MDR Required | Examples |
|----------|--------------|----------|
| Death | Yes | Patient death associated with device |
| Serious injury | Yes | Hospitalization, permanent damage |
| Malfunction (potential harm) | Yes | Device failure that could cause harm if recurs |
| Malfunction (no harm) | No | Device failed but no patient impact |
| User error | No | Misuse, no device defect |
| Quality concern | No | Cosmetic, packaging, labeling |

#### By IMDRF Code Categories
- **Device problems:** Material, mechanical, software, electrical, packaging
- **Patient problems:** Injury, infection, clinical symptoms
- **Multiple codes:** Combinations requiring multi-label classification

#### Ground Truth for Each Test Case
- Raw input (form data, email text, transcript, document image)
- Expected extracted fields
- Expected IMDRF codes (primary and secondary)
- Expected MDR determination
- Edge cases and known ambiguities
- Difficulty rating (easy/medium/hard)

---

## Key Architectural Decisions

### Decision 1: Human-in-the-Loop is Mandatory

**Rationale:** FDA regulations require human oversight for complaint coding decisions. The AI system is a decision support tool, not an autonomous decision maker.

**Implementation:**
- Every AI suggestion requires human review
- UI designed to minimize burden while ensuring thorough review
- Audit trail captures both AI suggestion and human decision

### Decision 2: LLM-First Approach

**Rationale:** Large Language Models with few-shot prompting provide faster development iteration, better explainability, and sufficient accuracy for this use case.

**Implementation:**
- Use GPT-4 or Claude with constrained output to valid IMDRF codes
- Few-shot examples in prompts for each code category
- Consider fine-tuned models later if cost or accuracy requires it

### Decision 3: Explainability from Day One

**Rationale:** Regulatory environment requires understanding why codes were suggested. Retrofitting explainability is costly and error-prone.

**Implementation:**
- Every code suggestion includes source text citation
- Confidence scores are calibrated and meaningful
- Reasoning chain captured in audit trail

### Decision 4: Conservative MDR Determination

**Rationale:** False negatives (missing an MDR-required complaint) have severe regulatory consequences. False positives (over-reporting) are acceptable.

**Implementation:**
- Rules-based primary determination
- AI-assisted edge case handling
- Default to "requires review" when uncertain
- Human approval required for all MDR decisions

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| LLM hallucination (invalid codes) | High | Medium | Constrained output to valid codes, validation layer |
| Inconsistent coding | Medium | Medium | Temperature=0, clear prompts, human oversight |
| Regulatory uncertainty | High | Low | Conservative approach, strong audit trails |
| PHI data exposure | High | Low | Azure AI with enterprise protection, no training on data |
| Performance degradation over time | Medium | Medium | Drift detection, periodic revalidation |
| User adoption resistance | Medium | Medium | Focus on time savings, intuitive UX |

---

## Technology Stack (Recommended)

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Backend API** | Python/FastAPI | Async, type hints, OpenAPI docs |
| **LLM Integration** | Azure OpenAI | Enterprise security, PHI compliance |
| **Speech-to-Text** | Azure Speech | Integration with Azure ecosystem |
| **OCR** | Azure Document Intelligence | High accuracy, structured extraction |
| **Database** | PostgreSQL | ACID compliance, audit requirements |
| **Audit Log** | Immutable append-only store | Regulatory requirement |
| **Frontend** | React/TypeScript | Modern, maintainable UI |
| **Authentication** | Azure AD | Enterprise SSO, compliance |

---

## Success Metrics

### Accuracy Metrics
- **Code suggestion accuracy:** % of AI suggestions accepted without modification
- **Extraction completeness:** % of required fields correctly extracted
- **MDR sensitivity:** % of MDR-required complaints correctly flagged (target: 100%)

### Efficiency Metrics
- **Review time reduction:** Time saved per complaint vs. manual process
- **Throughput:** Complaints processed per hour
- **First-pass yield:** % of complaints coded correctly on first review

### Compliance Metrics
- **Audit trail completeness:** 100% of actions logged
- **Validation coverage:** All requirements traced to test cases
- **Access control compliance:** No unauthorized access incidents

---

## Immediate Next Steps

1. **Obtain IMDRF documentation** — Build structured code database with hierarchies
2. **Create initial test dataset** — Start with 10-15 diverse synthetic complaints
3. **Set up development environment** — Repository, CI/CD, Azure resources
4. **Prototype extraction + coding** — Single-channel (forms), LLM-based
5. **Build minimal review UI** — Test human-in-the-loop workflow early

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2024-XX-XX | - | Initial draft |

---

## References

- [IMDRF Adverse Event Terminology](https://www.imdrf.org/documents/terminologies-categorized-adverse-event-reporting-aer-terms-terminology-and-codes)
- [21 CFR Part 820 - Quality System Regulation](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820)
- [21 CFR Part 11 - Electronic Records](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11)
- [21 CFR Part 803 - Medical Device Reporting](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-803)
- [FDA Guidance on AI/ML in Medical Devices](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices)
