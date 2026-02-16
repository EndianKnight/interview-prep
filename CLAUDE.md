# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Markdown-only** interview preparation guide for senior software engineers. There is no source code to build, test, or lint — the entire repo is structured documentation.

## Repository Structure

Eight numbered sections cover the full senior interview spectrum:

- **01-dsa/** — 19 pattern-based topic guides + language refreshers (C++, Java, Python)
- **02-system-design/** — 28 fundamentals (complete) + 14 case studies (stubs)
- **03-low-level-design/** — 13 principles + 4 concurrency topics + 16 case studies
- **04-design-patterns/** — 19 GoF patterns (creational, structural, behavioral)
- **05-ai-engineering/** — ML fundamentals, MLOps, LLM engineering
- **06-behavioral/** — STAR method, leadership principles, common questions
- **07-engineering-excellence/** — Ship it (CI/CD, containers, IaC), Run it (SRE, incidents), Build it right (testing, security)
- **08-frameworks/** — Spring Boot ecosystem (Boot, Data JPA, Security, Hibernate)
- **resources/** — Books, courses, cheat sheets

## Progress Tracking

`.tasks/tasks.md` is the master checklist. Legend: `[x]` = done, `[/]` = stub with topics outlined, `[ ]` = empty stub. Sections 01-dsa topics and 02-system-design fundamentals are complete; most other sections are stubs or partially written.

## Content Standards

When writing or expanding content:

- **Detailed with examples** — real code and diagrams, not just bullet points
- **Multi-language** — DSA content should include C++, Java, and Python implementations
- **Interview-oriented** — structure content to match how interviews are conducted
- **Actionable** — include links to practice problems at the end of each topic
- Follow the patterns established in completed sections (e.g., `01-dsa/topics/` files for DSA, `02-system-design/fundamentals/` files for system design)

## Workflow

1. Check `.tasks/tasks.md` for unchecked items to work on
2. Follow the existing template/structure of completed files in the same section
3. Mark items as complete (`[x]`) in `.tasks/tasks.md` after writing content
