# Heritage Document Recommender - User Study Protocol

## Study Overview

**Objective:** Evaluate the real-world effectiveness of the heritage document recommender system through controlled user studies measuring task success, user satisfaction, and comparative performance against baseline systems.

**Study Type:** Mixed-methods evaluation combining quantitative metrics and qualitative feedback

**Duration:** 45 minutes per participant

**Target Sample Size:** 50 participants (25 domain experts + 25 general users)

---

## 1. Participant Recruitment

### 1.1 Target Population

**Group A: Heritage Domain Experts (N=25)**
- Archaeologists, historians, heritage conservationists
- Museum curators, ASI officials
- Academic researchers in Indian heritage
- Minimum 2 years professional experience

**Group B: General Interest Users (N=25)**
- Students of history/archaeology
- Heritage tourism enthusiasts
- General public interested in cultural heritage
- No professional expertise required

### 1.2 Recruitment Criteria

**Inclusion Criteria:**
- Age 18 or older
- Fluent in English (reading level)
- Basic computer literacy
- Interest in Indian cultural heritage

**Exclusion Criteria:**
- Prior participation in system development/testing
- Inability to complete 45-minute session
- Visual impairments preventing screen reading

### 1.3 Recruitment Methods

- Email invitations to ASI, INTACH, heritage organizations
- University mailing lists (history/archaeology departments)
- Social media posts in heritage interest groups
- Snowball sampling through professional networks

### 1.4 Incentive

- ₹500 Amazon voucher per participant
- Certificate of participation for professional development records

---

## 2. Study Design

### 2.1 Study Structure

**Within-Subjects Design:** Each participant uses both systems
- System A: Heritage Document Recommender (Graph + Horn + Embedding)
- System B: Baseline (Embedding-only retrieval)
- Order counterbalanced to prevent learning effects

**Session Flow:**
1. Welcome & Consent (5 min)
2. Pre-study Questionnaire (5 min)
3. Training/Practice (5 min)
4. Task Block 1: System A or B (12 min)
5. Post-task Questionnaire 1 (3 min)
6. Task Block 2: System B or A (12 min)
7. Post-task Questionnaire 2 (3 min)
8. Comparative Questionnaire (3 min)
9. Semi-structured Interview (7 min)

### 2.2 Counterbalancing

| Participant # | First System | Second System |
|--------------|--------------|---------------|
| Odd (1,3,5...) | System A | System B |
| Even (2,4,6...) | System B | System A |

---

## 3. Search Tasks

### 3.1 Task Types

**Known-Item Search (5 tasks):** Find specific monuments
- Easy: Well-known monument (e.g., "Find information about Taj Mahal")
- Medium: Lesser-known monument (e.g., "Find Rani ki Vav stepwell")
- Hard: Obscure monument with alternate names

**Exploratory Search (5 tasks):** Discover related monuments
- Similarity: "Find monuments similar to Konark Sun Temple"
- Category: "Find Indo-Islamic architectural examples in Delhi"
- Temporal: "Find medieval Buddhist monuments in eastern India"
- Architectural: "Find monuments with Dravidian temple style"
- Thematic: "Find monuments related to Chola dynasty"

### 3.2 Task List

#### Known-Item Tasks (KI)

**KI-1 (Easy):** "Find information about the Red Fort in Delhi"
- Target: Red Fort document
- Success criterion: Target in top-5 results

**KI-2 (Medium):** "Find information about Sanchi Stupa"
- Target: Sanchi Stupa document
- Success criterion: Target in top-5 results

**KI-3 (Hard):** "Find information about Quwwat-ul-Islam Mosque (also known as Might of Islam)"
- Target: Qutub complex document
- Success criterion: Target in top-10 results

**KI-4 (Medium):** "Find Hampi Vittala Temple"
- Target: Vittala Temple document
- Success criterion: Target in top-5 results

**KI-5 (Easy):** "Find information about Ajanta Caves"
- Target: Ajanta Caves document
- Success criterion: Target in top-5 results

#### Exploratory Tasks (EX)

**EX-1 (Similarity):** "Find monuments architecturally similar to Taj Mahal"
- Success criterion: ≥3 Mughal monuments in top-10

**EX-2 (Category):** "Find examples of Indo-Islamic architecture in North India"
- Success criterion: ≥4 Indo-Islamic monuments in top-10

**EX-3 (Temporal):** "Find ancient Buddhist heritage sites from the Mauryan period"
- Success criterion: ≥3 relevant sites (Sanchi, Sarnath, etc.) in top-10

**EX-4 (Architectural):** "Find temples built in Dravidian style"
- Success criterion: ≥3 Dravidian temples in top-10

**EX-5 (Thematic):** "Find monuments associated with the Vijayanagara Empire"
- Success criterion: ≥3 Vijayanagara monuments (Hampi, etc.) in top-10

### 3.3 Task Randomization

- Each participant receives all 10 tasks (5 KI + 5 EX)
- Task order randomized within each block
- Same tasks used for both systems (different order)

---

## 4. Measurement Instruments

### 4.1 Quantitative Metrics

#### Task Performance Metrics

| Metric | Definition | Measurement |
|--------|------------|-------------|
| **Task Success Rate** | % of tasks completed successfully | Binary (success/fail per task) |
| **Time to Completion** | Seconds to complete each task | Automated timestamp logging |
| **Number of Clicks** | Clicks before finding target | Click event logging |
| **Scroll Depth** | How far user scrolled in results | Scroll tracking |
| **Query Reformulations** | # of query modifications | Query log analysis |

#### Relevance Judgments

For each top-10 result shown:
- **User Rating:** 4-point scale
  - 3 = Perfect (exactly what I was looking for)
  - 2 = Excellent (highly relevant)
  - 1 = Good (somewhat relevant)
  - 0 = Not Relevant

- **Did you click?** Yes/No
- **If clicked, dwell time:** Seconds spent viewing document

#### System-Level Metrics (Computed)
- Precision@5, Precision@10
- NDCG@5, NDCG@10
- Mean Reciprocal Rank (MRR)
- Click-Through Rate (CTR)
- Average dwell time

### 4.2 Qualitative Measures

#### Post-Task Questionnaire (After each system)

**Usability (7-point Likert scale: 1=Strongly Disagree, 7=Strongly Agree)**
1. The search results were relevant to my queries
2. I could easily find what I was looking for
3. The system understood my information needs
4. The results were well-organized and easy to scan
5. I discovered unexpected but useful information

**Satisfaction (7-point Likert scale)**
6. I am satisfied with this search system
7. I would use this system again
8. I would recommend this system to others

**Diversity Perception**
9. The results showed good variety (different time periods, regions, types)
10. The results were too similar to each other (reverse-scored)

**Explanation Quality (if shown)**
11. The system explained why results were recommended
12. The explanations helped me understand the connections

**Open-Ended Questions:**
- What did you like most about this system?
- What did you like least?
- Were there any tasks where you struggled to find relevant results?

#### Comparative Questionnaire (After both systems)

**Direct Comparison (Forced Choice)**
1. Which system provided more relevant results overall?
   - [ ] System A  [ ] System B  [ ] No difference

2. Which system was easier to use?
   - [ ] System A  [ ] System B  [ ] No difference

3. Which system helped you discover more interesting monuments?
   - [ ] System A  [ ] System B  [ ] No difference

4. Which system would you prefer to use?
   - [ ] System A  [ ] System B  [ ] No difference

**Explanation:**
5. Why did you prefer the system you chose? (Open-ended)

### 4.3 Semi-Structured Interview

**Duration:** 5-7 minutes

**Topics:**
1. **Search Strategies:**
   - How did you approach the exploratory tasks?
   - Did you modify your queries? Why?

2. **Result Quality:**
   - Were the results diverse enough?
   - Did you encounter any irrelevant results? Why might they have appeared?

3. **Discovery:**
   - Did you learn about any monuments you weren't aware of?
   - Were the connections between monuments clear?

4. **Improvement Suggestions:**
   - What features would make the system more useful?
   - Any frustrations or confusion?

5. **Expert-Specific (for Group A only):**
   - Would this system be useful in your professional work?
   - How does it compare to resources you currently use?

**Recording:** Audio recorded with consent, transcribed for thematic analysis

---

## 5. Experimental Procedure

### 5.1 Pre-Study

**Day Before:**
- Send confirmation email with study location/time
- Remind about consent and time commitment

**Study Day Setup:**
- Prepare two computers (System A and B)
- Test both systems with sample queries
- Prepare consent forms and questionnaires
- Ensure screen recording software is running

### 5.2 Session Protocol

**Welcome (5 minutes)**
- Greet participant
- Review consent form (IRB-approved)
- Explain study purpose (without revealing hypothesis)
- Answer questions
- Obtain written consent
- Assign participant ID

**Pre-Study Questionnaire (5 minutes)**
- Demographics (age, gender, education)
- Heritage knowledge self-rating (1-7 scale)
- Frequency of heritage research
- Familiarity with search systems

**Training (5 minutes)**
- Demonstrate system interface
- Show how to enter queries
- Explain result display
- Practice with 2 sample tasks (not from test set)
- Answer questions

**Task Block 1 (12 minutes)**
- Participant uses first system (A or B based on counterbalancing)
- 5 tasks (randomized mix of KI and EX)
- Experimenter observes but does not assist
- Screen and audio recorded
- "Think aloud" encouraged but not required

**Post-Task Questionnaire 1 (3 minutes)**
- Complete questionnaire for first system

**Task Block 2 (12 minutes)**
- Participant uses second system
- Same 5 tasks (different order)
- Screen and audio recorded

**Post-Task Questionnaire 2 (3 minutes)**
- Complete questionnaire for second system

**Comparative Questionnaire (3 minutes)**
- Direct comparison questions
- Preference and reasoning

**Interview (7 minutes)**
- Semi-structured interview
- Audio recorded

**Debrief & Payment**
- Explain study goals
- Reveal which system was which
- Provide voucher code
- Thank participant

### 5.3 Data Collection

**Automated Logging:**
- Query text and timestamp
- Click events (doc_id, rank, timestamp)
- Scroll events
- Time spent on each document
- Session duration

**Manual Recording:**
- Task success (binary)
- Relevance ratings (0-3 per result)
- Questionnaire responses
- Interview notes

**Qualitative Data:**
- Audio transcriptions
- Think-aloud observations
- Experimenter notes on struggles/frustrations

---

## 6. Analysis Plan

### 6.1 Quantitative Analysis

#### Primary Outcomes

**Task Success Rate**
- Statistical Test: McNemar's test (paired binary data)
- Hypothesis: System A > System B
- Effect size: Odds ratio

**NDCG@10**
- Statistical Test: Paired t-test (or Wilcoxon if non-normal)
- Hypothesis: System A NDCG > System B NDCG
- Effect size: Cohen's d

**User Satisfaction (Likert scale)**
- Statistical Test: Paired t-test
- Hypothesis: System A satisfaction > System B satisfaction
- Effect size: Cohen's d

**Time to Completion**
- Statistical Test: Paired t-test
- Hypothesis: No difference (or System A faster)

#### Secondary Analyses

**Task Type Interaction**
- Mixed ANOVA: System × Task Type (KI vs. EX)
- Hypothesis: System A advantage larger for exploratory tasks

**User Expertise Interaction**
- Mixed ANOVA: System × User Group (Expert vs. General)
- Explore if experts benefit more from advanced features

**Diversity Metrics**
- Wilcoxon signed-rank test for diversity perception ratings
- Hypothesis: System A perceived as more diverse

**Click-Through Rate**
- Chi-square test for CTR differences
- Analyze position bias

#### Statistical Power

- Sample size (N=50) provides 80% power to detect medium effect size (d=0.4) at α=0.05
- Paired design increases power vs. between-subjects

### 6.2 Qualitative Analysis

**Thematic Analysis of Interviews**
1. Transcribe all interviews
2. Initial coding by two independent coders
3. Identify recurring themes:
   - Usability issues
   - Discovery moments
   - Confusion points
   - Feature requests
4. Resolve disagreements through discussion
5. Categorize themes (positive/negative/neutral)
6. Quantify theme frequency

**Failure Mode Analysis**
- Identify queries where both systems failed
- Analyze common failure patterns
- Categorize errors:
  - Missing documents
  - Irrelevant results
  - Poor ranking
  - Query understanding failures

**Open-Ended Responses**
- Content analysis of post-task comments
- Categorize feedback by system component:
  - Result relevance
  - Result diversity
  - Interface usability
  - Explanation quality

### 6.3 Reporting

**Results Structure:**
1. Participant characteristics
2. Primary outcome analysis (task success, NDCG, satisfaction)
3. Secondary analyses (task type, user group)
4. Diversity and fairness perception
5. Qualitative findings
6. Limitations and future work

**Visualizations:**
- Box plots: NDCG by system and task type
- Bar charts: Task success rates
- Heatmaps: Relevance ratings by position and system
- Word clouds: Open-ended feedback themes

---

## 7. Ethical Considerations

### 7.1 Informed Consent

**Consent Form Includes:**
- Study purpose and duration
- Data collection methods (screen recording, audio)
- Voluntary participation
- Right to withdraw at any time
- Data anonymization and storage
- Compensation details

### 7.2 Data Privacy

**Anonymization:**
- Assign participant IDs (P001-P050)
- Remove names from all data files
- Store consent forms separately from data
- Audio files destroyed after transcription

**Data Security:**
- Encrypted storage for all data files
- Access restricted to research team
- Data retention: 5 years, then destroyed

### 7.3 IRB Approval

- Submit protocol to Institutional Review Board
- Obtain approval before recruitment
- Report any adverse events
- Annual review if study extends beyond 1 year

---

## 8. Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Preparation** | 2 weeks | IRB submission, system setup, questionnaire design |
| **Pilot Testing** | 1 week | 5 pilot participants, refine protocol |
| **Recruitment** | 2 weeks | Participant recruitment and scheduling |
| **Data Collection** | 4 weeks | Run 50 sessions (2-3 per day) |
| **Transcription** | 2 weeks | Transcribe interviews, clean data |
| **Analysis** | 3 weeks | Statistical analysis, thematic coding |
| **Reporting** | 2 weeks | Write report, create visualizations |
| **Total** | **16 weeks** | |

---

## 9. Budget

| Item | Cost (₹) | Notes |
|------|----------|-------|
| **Participant Incentives** | 25,000 | 50 × ₹500 vouchers |
| **Transcription Service** | 10,000 | 50 interviews @ ₹200/interview |
| **Equipment** | 5,000 | Screen recording software, backup storage |
| **IRB Fees** | 2,000 | Institutional review |
| **Printing** | 1,000 | Consent forms, questionnaires |
| **Miscellaneous** | 2,000 | Contingency |
| **Total** | **₹45,000** | (~$540 USD) |

---

## 10. Expected Outcomes

### 10.1 Quantitative Results

**Hypothesized Findings:**
- System A (Heritage Recommender) achieves:
  - 15-20% higher task success rate
  - 0.1-0.15 higher NDCG@10
  - 1-1.5 points higher satisfaction (7-point scale)
  - 10-15% higher diversity perception

- Exploratory tasks show larger advantage for System A
- Expert users show greater appreciation for graph-based connections

### 10.2 Qualitative Insights

**Expected Themes:**
- Appreciation for discovery of related monuments
- Frustration with query formulation for complex needs
- Value of explanations showing historical connections
- Requests for filtering (region, time period, type)

**Failure Modes:**
- Rare heritage types underrepresented
- Ambiguous monument names causing confusion
- Limited support for non-English queries

### 10.3 Actionable Improvements

**System Enhancements:**
1. Add faceted filtering (region, period, type)
2. Improve query understanding for ambiguous terms
3. Surface connections more prominently in UI
4. Add "related monuments" panel
5. Support Hindi/regional language queries

---

## 11. Limitations

**Sample Limitations:**
- Convenience sampling (not random)
- English-language bias
- Limited geographic diversity (likely Delhi-NCR based)

**Task Limitations:**
- Predefined tasks (not organic search needs)
- Laboratory setting (not real-world context)
- Short session duration (learning effects)

**Measurement Limitations:**
- Self-reported satisfaction (subject to bias)
- Binary success criteria (ignores partial success)
- No long-term retention measurement

**Mitigation Strategies:**
- Diverse recruitment across organizations
- Mix of task types (known-item + exploratory)
- Think-aloud protocol to capture partial successes
- Future: Deploy live system with analytics

---

## 12. Dissemination Plan

### 12.1 Research Outputs

**Academic Publication:**
- Conference paper: ACM SIGIR, CIKM, or RecSys
- Focus: Graph-enhanced recommendation for cultural heritage

**Technical Report:**
- Detailed methodology and findings
- Shared with ASI, INTACH, heritage organizations
- Open-access repository (arXiv, institutional repo)

**Presentation:**
- Heritage informatics workshop
- Digital humanities conference
- Demo at heritage organizations

### 12.2 Practitioner Engagement

**Workshops:**
- Training sessions for ASI librarians
- Demo for heritage tourism companies
- Educational webinar for history students

**System Deployment:**
- Open-source release on GitHub
- Web interface deployment (if resources available)
- API for integration with existing heritage databases

---

## Appendices

### Appendix A: Sample Consent Form

```
INFORMED CONSENT FOR RESEARCH

Study Title: Evaluation of Heritage Document Recommendation Systems
Principal Investigator: [Name, Affiliation]

Purpose: You are being asked to participate in a research study evaluating search
systems for Indian cultural heritage documents. This study compares different
approaches to recommending relevant heritage information.

Procedures: You will complete two search tasks sessions (approximately 30 minutes
total), answer questionnaires, and participate in a brief interview. Your screen
and voice will be recorded during the session.

Risks: There are no foreseeable risks beyond those of normal computer use.

Benefits: You will receive a ₹500 Amazon voucher. You may gain knowledge about
Indian heritage monuments through the search tasks.

Confidentiality: All data will be anonymized. Recordings will be destroyed after
transcription. Your identity will not be disclosed in any publications.

Voluntary Participation: Your participation is entirely voluntary. You may withdraw
at any time without penalty.

Contact: For questions, contact [Researcher Email/Phone]

Consent: I have read this form and agree to participate in this research study.

Signature: _____________________ Date: __________
Participant Name (printed): _____________________
```

### Appendix B: Pre-Study Questionnaire

```
Participant ID: P___

Demographics:
1. Age: ___
2. Gender: [ ] Male [ ] Female [ ] Other [ ] Prefer not to say
3. Highest Education: [ ] High School [ ] Bachelor's [ ] Master's [ ] PhD
4. Occupation: ___________________

Heritage Background:
5. How would you rate your knowledge of Indian cultural heritage?
   1 (Novice) - 2 - 3 - 4 - 5 - 6 - 7 (Expert)

6. How often do you search for heritage-related information?
   [ ] Daily [ ] Weekly [ ] Monthly [ ] Rarely [ ] Never

7. Which resources do you typically use? (Check all that apply)
   [ ] Google [ ] Wikipedia [ ] ASI website [ ] Academic databases
   [ ] Books [ ] Other: ___________

8. Have you visited historical monuments in India?
   [ ] Yes, more than 20 [ ] Yes, 10-20 [ ] Yes, 5-10 [ ] Yes, 1-5 [ ] No
```

### Appendix C: Data Collection Spreadsheet Template

| Participant | System_Order | Task_ID | System | Query | Success | Time_sec | Clicks | NDCG | Satisfaction |
|-------------|--------------|---------|--------|-------|---------|----------|--------|------|--------------|
| P001 | AB | KI-1 | A | red fort delhi | 1 | 12.3 | 1 | 1.0 | 6 |
| P001 | AB | EX-1 | A | similar to taj mahal | 1 | 45.2 | 4 | 0.85 | 5 |

---

**Document Version:** 1.0
**Last Updated:** November 2025
**Contact:** [Study Coordinator Contact Information]
