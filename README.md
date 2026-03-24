# Resume and Job Description Matching System

An AI-based hiring platform that bridges the gap between job seekers and employers by intelligently matching resumes to job descriptions, generating match scores, and providing personalised improvement suggestions.

---

## Project Details

| Field       | Details                            |
| ----------- | ---------------------------------- |
| Subject     | AI in Engineering & Science (AIES) |
| Class       | CSE (AIDS) — Panel A, Batch A3     |
| Institution | MIT - WPU                          |

### Group Members

| SR. No | Roll No | PRN        | Name               |
| ------ | ------- | ---------- | ------------------ |
| 1      | 46      | 1262240853 | Purvi Agarwal      |
| 2      | 49      | 1262240865 | Afiya Amanulla     |
| 3      | 51      | 1262240867 | Agnes Maria Thomas |

---

## Overview

In today's competitive job market, candidates often face rejection because their resumes don't clearly align with job requirements. At the same time, employers struggle to manually screen large volumes of applications efficiently.

This system solves both problems through two dedicated portals:

- **Job Seeker Portal** — Upload a resume, view a match score against a job posting, see which skills are missing, and get personalised suggestions to improve alignment.
- **Employer Portal** — Create a job posting, upload multiple resumes, and get candidates automatically ranked by relevance.

---

## Features

- PDF resume parsing and text extraction
- AI-based skill extraction with synonym handling (e.g. "ML" = "Machine Learning")
- TF-IDF based resume-to-job-description matching
- Match score displayed as a percentage
- Matched skills and missing skills clearly shown
- Personalised improvement suggestions for job seekers
- Candidate ranking table for employers
- Dual portal interface built with Streamlit

---

## Tech Stack

| Layer              | Technology              |
| ------------------ | ----------------------- |
| PDF parsing        | `pdfplumber`            |
| Skill extraction   | `spaCy`                 |
| Matching algorithm | `scikit-learn` (TF-IDF) |
| User interface     | `Streamlit`             |
| Language           | Python 3.x              |
| Version control    | Git + GitHub            |

---

## Project Structure

```
AIES-PROJECT/
│
├── notebooks/
│   ├── 01_resume_parser.ipynb       # PDF reading and text extraction
│   ├── 02_skill_extractor.ipynb     # Skill identification and synonym handling
│   ├── 03_matching_engine.ipynb     # TF-IDF scoring and gap analysis
│   └── 04_suggestions.ipynb         # Personalised improvement suggestions
│
├── resumes/
│   └── sample_resume.pdf            # Sample resume for testing
│
├── job_descriptions/
│   └── sample_job.txt               # Sample job description for testing
│
├── app/
│   ├── jobseeker_portal.py          # Streamlit UI for job seekers
│   └── employer_portal.py           # Streamlit UI for employers
│
├── data/
│   └── skills_database.py           # Master list of known skills
│
├── .gitignore
└── README.md
```

---

## How It Works

1. **Resume is uploaded** as a PDF file
2. **Text is extracted** using `pdfplumber`
3. **Skills are identified** from the extracted text using `spaCy` and matched against a skills database
4. **Job description is parsed** using the same skill extraction pipeline
5. **Match score is calculated** using TF-IDF cosine similarity between resume and job description
6. **Results are displayed** — score percentage, matched skills, missing skills, and suggestions

---

## Problem Statement & Gaps Addressed

| Gap Identified                                  | Our Solution                                                  |
| ----------------------------------------------- | ------------------------------------------------------------- |
| Over-dependence on exact keyword matching       | Synonym-aware skill extraction (ML = Machine Learning)        |
| No transparent feedback to candidates           | Matched skills, missing skills, and suggestions clearly shown |
| Poor skill extraction from unstructured resumes | AI-based parsing using spaCy                                  |
| Platforms are one-sided                         | Dual portal — one for job seekers, one for employers          |

---
