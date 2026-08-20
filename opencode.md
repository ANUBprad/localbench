# OpenCode Working Rules — Detailed Edition

## Purpose

This document defines the comprehensive engineering rules OpenCode must follow before and during every development phase in any software project.

The goal is to produce code that looks and behaves like it was written by a careful human engineer working on an existing project — not code generated in bulk by an AI. Every rule is anchored in the principle that the smallest legitimate change that solves the actual problem is preferable to larger, more "sophisticated" solutions.

This guide applies universally to:
- Backend services (Python, Java, Go, Node.js, Rust, etc.)
- Frontend applications (React, Vue, Angular, etc.)
- Full-stack projects
- Microservices and distributed systems
- Data pipelines and ML systems
- Infrastructure and DevOps code
- Mobile applications
- Desktop applications
- Libraries and frameworks
- Any codebase at any stage of maturity

---

## 1. Read Before You Change

### Core Principle
Never start implementing until you fully understand the existing system, the problem you are solving, and the ripple effects of your changes.

### Detailed Process

**1.1 Read the Relevant Existing Implementation**
- Locate all files that implement the feature or subsystem you are about to touch.
- Read the entire relevant module or service, not just the functions you plan to modify.
- Understand the existing data structures, patterns, and abstractions already in place.
- If the codebase is large, trace the main execution path for the feature you are modifying.
- Look for utility functions, helpers, constants, and error handling patterns already established.
- Do not skip "obvious" sections; they often contain critical context or edge case handling.

**1.2 Trace Real Execution Paths**
- Start from the user-facing entry point (API endpoint, component, event handler, CLI command).
- Follow the code path step by step to understand how data flows through the system.
- Identify every function call, database query, external API call, and state mutation along the way.
- Note where async operations, caching, or deferred execution might affect behavior.
- Understand the performance characteristics: is this path hot or cold? Are there bottlenecks?
- Trace error paths as well; understand what happens when things go wrong.

**1.3 Examine Existing Tests, Schemas, and Contracts**
- Read the test files that exercise the code you are about to change.
- Understand what behavior the tests expect and protect.
- Check API schemas (OpenAPI, GraphQL, Protobuf, or project-specific formats) for the endpoints involved.
- If the project uses TypeScript or similar, check type definitions and interfaces.
- Review frontend code that consumes the backend APIs you are modifying.
- Check database schemas, validation rules, and constraints that might be affected.
- Look for deprecation warnings, TODOs, or known limitations in the documentation or code comments.

**1.4 Check Git Status and Understand Unrelated Work**
- Before making changes, run `git status` to see if there is already in-progress work.
- Read any uncommitted changes to understand what is already being developed.
- If unrelated work is present, identify which files belong to which changes.
- Do not mix your changes into unrelated work or vice versa.
- If you discover that a file has unrelated changes, plan to stage and commit separately.

**1.5 Reuse Existing Patterns and Utilities**
- Search the codebase for similar implementations before writing new code.
- If a utility function, validation handler, or abstraction exists elsewhere, use it.
- If it is almost right but not perfect, prefer to improve the existing utility over creating a duplicate.
- Study how the project handles common tasks (logging, error handling, configuration, caching) and follow those patterns.
- If the project uses a specific library or framework consistently, use it the same way elsewhere.
- Examples across project types:
  - **Backend**: If a database access pattern exists, use it for new queries.
  - **Frontend**: If a state management pattern exists, use it for new components.
  - **Data pipelines**: If a data validation utility exists, use it for new steps.
  - **Infrastructure**: If a deployment pattern exists, use it for new services.

**1.6 Do Not Redesign Based on Aesthetics**
- If the existing implementation is functional and stable, do not redesign it just because you think a different structure looks cleaner.
- A working system that is slightly awkward to read is better than a rewritten system that introduces new bugs.
- If you discover that the existing design is genuinely problematic (performance, maintainability, correctness), report it separately.
- Cosmetic refactoring should never be mixed into a feature or bug-fix phase.

**1.7 Do Not Modify Based on Filename or Small Search Results**
- Never edit a file based solely on its name or a single grep result.
- Always read the full file to understand its purpose, scope, and any side effects.
- A file with a promising name might do something entirely different than you expect.
- Context matters; the same-looking code in different parts of the system might have entirely different meanings.

---

## 2. Work in Small, Bounded Phases

### Core Principle
Every change is a discrete, well-defined phase with clear goals, scope, verification criteria, and a completion report.

### Phase Definition

**2.1 Establish a Clear Objective**
- State the problem or requirement in one or two sentences.
- Example: "Fix the race strategy calculation to include pit stop timing when the track has debris."
- Do not state objectives as vague aspirations ("improve code quality") or meta-goals ("refactor the service").
- The objective should be testable; you should be able to verify that you completed it.

**2.2 Define Precise Scope**
- List the specific files or components expected to change.
- Identify files that might be affected but will not be changed, and why.
- State any files or subsystems deliberately excluded from this phase.
- Example: "Will modify `strategy_engine.py` and `pit_logic.py`. Will NOT touch telemetry ingestion (`telemetry.py`) as it is a separate layer."
- Scope boundaries protect against scope creep and unrelated work.

**2.3 Identify Expected Changes**
- List the specific functions, classes, or components that will be added, modified, or removed.
- Note which files will have new dependencies or imports added.
- Identify any database schema changes, API contract changes, or configuration additions.
- If tests will be added or modified, list them.
- Do not be vague; "fix logging" is not a specific change. "Add structured logging to pit stop decision points" is.

**2.4 Establish Verification Criteria**
- What tests must pass for this phase to be complete?
- What manual or integration testing is required?
- What should the end user observe when this phase is complete?
- What should NOT change or break?
- Examples: "All existing strategy tests pass. New pit stop test case passes. Frontend dashboard shows updated pit time estimates."

**2.5 Prohibition on Mixed Concerns**
Do not mix the following into a single phase without explicit justification:
- Cleanup or refactoring unrelated to the current feature.
- Security hardening.
- UI redesign or styling overhauls (frontend-specific).
- Database schema redesigns (backend-specific).
- Dependency upgrades.
- Architecture changes (e.g., "let's move this to a microservice", "let's switch to Redux").
- Performance optimization (unless required to solve the immediate problem).
- Linting or formatting fixes for unrelated code.
- Dead code removal that is not part of the feature.
- Infrastructure or deployment changes (unless required for the feature).
- API contract changes (unless required for the feature).

If you discover a security issue, code quality problem, or technical debt during development, document it separately as a discovered issue, not as part of the current phase scope.

**2.6 Separate Issue Reporting**
- If you find a bug, performance issue, or design problem unrelated to the current phase, report it.
- Track it as a separate task, issue, or TODO.
- Only include it in the current phase if it actively blocks the phase completion.
- Example: "During testing, discovered that the telemetry service drops updates under high load. Opened ticket #523 for investigation. Not blocking this phase."

---

## 3. Write Legitimate, Production-Quality Code

### Core Principle
Every line of code must be real, correct, and solve an actual problem. No fake implementations, hardcoded test data, or placeholder logic.

### What "Legitimate" Means

**3.1 Solve the Actual Problem**
- Understand what the real problem is before you write code to fix it.
- Trace through to confirm that your solution actually addresses the root cause, not a symptom.
- Test your solution against real data or realistic scenarios.
- If you cannot solve the real problem immediately, do not write fake code that pretends to work.
- Examples across project types:
  - **Backend**: If a user reports slow queries, trace to find the actual bottleneck (missing index, N+1 query, etc.), not just add generic caching.
  - **Frontend**: If a component is slow, profile to find the actual cause (re-renders, DOM operations, etc.), not just add arbitrary optimizations.
  - **Data pipeline**: If data is missing, trace to find the actual root (missing source, filter error, etc.), not just add dummy fallbacks.
  - **Infrastructure**: If deployment fails, trace to find the actual error (misconfiguration, missing permissions, etc.), not just add retry logic.

**3.2 Never Add Fake Implementations**
Do not add:
- Placeholder functions that return `None` or dummy values when the feature is supposed to be functional.
- Stub implementations marked as "TODO" that are presented as complete.
- Mock or test data in production code paths unless the feature explicitly requires mocking.
- Hardcoded responses that pretend to come from an ML model, external API, or calculation engine.
- Functions that look real but contain no logic.

If you cannot implement a feature completely, either:
- Implement the parts you can and document the remaining work clearly.
- Use feature flags or configuration to disable incomplete features.
- Return an explicit error indicating the feature is not yet available.
- Leave a clear TODO or GitHub issue number in the code.

**3.3 No Unused Abstractions**
- Do not create classes, modules, or helper functions that are not immediately used.
- Do not add "extensibility points" or "plugin architecture" for hypothetical future features.
- Do not create interfaces or base classes for a single implementation.
- If you add a utility, ensure it is used in at least one real code path within the same phase.

**3.4 No Unnecessary Wrappers**
- Do not wrap a simple function call in another function without a concrete reason.
- Do not add middleware, decorators, or adapters just to look sophisticated.
- Do not create abstraction layers around straightforward logic.
- Example: Don't wrap a database query in three layers of factory methods if the query is called once.

**3.5 No Speculative Extension Points**
- Do not add parameters, configuration options, or callbacks for future use cases.
- Do not build in flexibility for scenarios that might happen someday.
- Do not design for "extensibility" when the current requirement is fixed.
- If a requirement changes later, refactor to add the flexibility then.

**3.6 No Extra Configuration**
- Do not add configuration options that have no current consumer.
- Do not create environment variables for hypothetical deployment scenarios.
- If your code needs a configuration, add it only when it is actually used.
- Configuration creep makes systems harder to reason about.

**3.7 When Existing Implementation Is Correct, Leave It Alone**
- If the existing code already solves the problem correctly, do not modify it.
- Do not refactor working code just to align it with your preferred style.
- Do not upgrade or rewrite stable code because you think you know a better way.
- A tested, working solution is more valuable than an untested, "better" one.

**3.8 Prefer Small Correct Changes Over Large Improvements**
- When in doubt, make the smallest change that solves the problem.
- A 3-line fix is better than a 30-line refactor, even if the refactor would make the code cleaner.
- A single function addition is better than restructuring an entire module.
- Ship something that works now, not something that might be more elegant later.

---

## 4. Do Not Overcode

### Core Principle
This is one of the highest-priority rules. More code is not better code. Less code that solves the problem correctly is always preferable.

### How to Identify Overcoding

**4.1 Rewriting Entire Files for Small Fixes**
- When you find a bug or need to add a feature, identify the minimum change first.
- Do not rewrite the entire file just because you can improve it.
- If the file is 500 lines and you need to fix a 5-line function, touch only those 5 lines.
- Do not use a fix as an excuse to refactor everything around it.

**4.2 Large Refactors for Small Problems**
- Do not restructure an entire module to fix one bug.
- Do not split a small function into many files without a real need.
- Do not reorganize packages or folder structures as part of a feature addition.
- Do not rewrite tests that pass just to make them "better."

**4.3 Introducing Unnecessary Classes and Patterns**
Avoid:
- Creating a class when a function is sufficient.
- Introducing design patterns (Factory, Strategy, Builder, etc.) just to look sophisticated.
- Splitting logic into multiple classes when one would suffice.
- Creating base classes and inheritance hierarchies for a single implementation.
- Using abstract base classes when composition would be simpler.

**4.4 Duplicating Existing Helpers**
- Before writing a new utility function, search the codebase for similar implementations.
- If a function does 95% of what you need, modify it or compose with it rather than creating a duplicate.
- Duplicated logic makes maintenance harder and causes bugs when one copy is fixed but another is not.

**4.5 Adding Layers of Abstraction Around Simple Logic**
- Do not wrap simple operations in multiple layers of indirection.
- Do not create adapters and connectors for straightforward interactions.
- Do not build middleware for one-off needs.
- Example: Don't create a "FetchUserStrategy" class to wrap a simple database query.

**4.6 Renaming Unrelated Variables or Functions**
- When fixing one thing, do not rename or refactor everything else in the file.
- If you touch a file, resist the urge to "clean up" unrelated names.
- Renaming makes diffs harder to review and can introduce subtle bugs.
- Save cosmetic renaming for a dedicated cleanup phase.

**4.7 Formatting Unrelated Sections**
- Do not reformat or reorganize code that you are not changing functionally.
- When you make a fix, do not also "fix the indentation" or "clean up the spacing" in neighboring code.
- Formatting changes make diffs noisy and harder to review.
- Use a dedicated formatting/linting phase if needed, not as part of feature work.

**4.8 Measure Code Addition by Necessity**
- A good phase leaves the repository with the minimum code necessary to implement the requirement correctly.
- If you added 500 lines of code, ask: could this be done in 50 lines?
- If you added 10 functions, ask: do we need all 10?
- If you added 5 classes, ask: can we use 2?

---

## 5. Humanized Code

### Core Principle
Code should read naturally, as if written by a competent human engineer who understands the project, not as if generated by an algorithm.

### Clear Variable Names
- Use names that describe what the variable holds, not abbreviations.
- Prefer `driver_pit_stop_time` over `dpst` or `d_pst`.
- Use full words: `total_distance`, not `tot_dist`.
- For loop counters and temporary variables, short names are acceptable: `i`, `j`, `x`, `y`.
- For anything stored or passed around, use full descriptive names.
- Names should make the code understandable without comments.

### Straightforward Control Flow
- Write code that is easy to follow from top to bottom.
- Avoid deep nesting; prefer early returns or guard clauses.
- Use if/else in the natural order: check conditions that matter most first.
- Avoid complex boolean logic; break it into named variables or helper functions.
- Example: Instead of `if not (a and (b or c)) and not (d or (e and f)):`, write:
  ```python
  is_invalid_state = not a or (not b and not c)
  is_blocked = d or (e and f)
  if is_invalid_state or is_blocked:
      return
  ```

### Project Conventions
- Study how the project writes functions, classes, and modules.
- Follow the same patterns for imports, logging, error handling, and testing.
- Use the same naming conventions (snake_case, CamelCase, CONSTANT_CASE) as the rest of the codebase.
- Mirror the existing code style for indentation, line length, and spacing.
- If the project uses a specific library for HTTP, database, or configuration, use it the same way.

### Small Functions with Obvious Responsibilities
- Each function should do one thing and do it well.
- A function should fit on a screen without scrolling.
- If a function is longer than 30 lines, consider breaking it into smaller functions.
- Function names should describe what they do: `calculate_pit_stop_time()`, not `process()` or `execute()`.
- Avoid functions with complex branching or multiple exit points.

### Normal Python/TypeScript Idioms
- Use list comprehensions instead of for loops for simple transformations.
- Use context managers (`with` statements) for resource management.
- Use generator expressions for memory-efficient iteration.
- Prefer duck typing over explicit type checking.
- Use type hints in Python and TypeScript for clarity, but do not over-annotate.
- Follow PEP 8 (Python) and ESLint (TypeScript) conventions.

### Practical Error Handling
- Catch specific exceptions, not `Exception` or `BaseException`.
- Log errors with context; do not silently swallow them.
- Return meaningful error messages to the user.
- Use the project's error handling patterns (custom exceptions, error codes, error responses).
- Do not add defensive programming for hypothetical failures; handle real error cases.

### Comments Only Where They Help
- Do not comment obvious code.
- Do not describe what the code does; the code itself should be clear.
- Comments should explain why, not what.
- Comments should describe non-obvious intent, tradeoffs, or historical context.
- Remove comments that are outdated or no longer accurate.
- Good comments explain constraints, limitations, or unexpected behavior.

---

## 6. Comments: Keep Them Sparse

### Core Principle
Comments should be rare and meaningful. If the code is clear, let it speak for itself.

### Bad Comments (Do Not Write These)
```python
# Get the driver from the telemetry
driver = telemetry.driver

# Loop through each lap
for lap in laps:
    # Calculate lap time
    lap_time = lap.end_time - lap.start_time

# Check if speed is greater than 300
if speed > 300:
    is_high_speed = True
```

These comments add no value; the code already says what it does.

### Good Comments (Do Write These)
```python
# Telemetry can lag one tick behind the simulator state.
# Use the most recent value, but account for potential stale data.
driver = telemetry.driver

# Pit stop times include in-lap penalties but exclude exit time from pit lane.
# This is because the strategy engine models pit exit separately.
pit_stop_duration = pit_time - exit_time

# High-speed zones require different braking strategies; flag them early
# so the telemetry processor can apply zone-specific calculations.
if speed > 300:
    is_high_speed = True
```

These comments explain intent, context, or non-obvious design decisions.

### When to Comment
- Explain why a non-obvious implementation choice was made.
- Document assumptions about data format, timing, or external system behavior.
- Note performance implications or tradeoffs.
- Explain complex algorithms or mathematical operations.
- Reference external systems, specifications, or issues.
- Warn about subtle bugs or edge cases.

### When NOT to Comment
- The code is obvious and self-explanatory.
- The comment would just repeat the code.
- The function name and variable names already explain what is happening.
- The comment describes implementation details that might change; refactor instead.

### Comment Standards
- Comments should be complete sentences, starting with a capital letter.
- Avoid abbreviations and jargon.
- Keep comments concise; one or two lines is usually enough.
- Do not write multi-line AI-generated explanations.
- Update comments when you update code; outdated comments are worse than no comments.

---

## 7. Do Not Change Contracts Without a Reason

### Core Principle
Contracts include API responses, database schemas, function signatures, hook contracts, message formats, and service interfaces. These are "contracts" that other code depends on. Changing them is risky and requires thorough planning across all project types.

### Before Changing Any Contract

**7.1 Find All Consumers**
- Search for all code that uses the interface, function, or field you are changing.
- Examples across project types:
  - **Backend**: Database schema changes → find all queries, ORM mappings, migrations.
  - **Frontend**: Component props changes → find all usages of the component.
  - **API**: Response structure changes → find all client code consuming the API.
  - **Data pipeline**: Message format changes → find all producers and consumers.
  - **Library**: Public API changes → find all callers within and outside the project.
  - **Infrastructure**: Configuration changes → find all services consuming the config.
- Do not assume you have found all consumers; use a thorough code search tool.

**7.2 Check Tests**
- Run the tests that exercise the contract.
- Understand what behavior the tests expect.
- Identify tests that would break if you change the contract.
- Check whether there are integration tests or end-to-end tests that depend on the contract.

**7.3 Check All Downstream Consumers**
- For backend APIs: Search the frontend codebase for references to the API endpoint.
  - Look at how the response is parsed and used in components.
  - Identify which fields are required and which are optional.
  - Check whether the frontend has hardcoded assumptions about response structure.
  - Look for error handling that depends on the current response format.
- For data pipelines: Find all downstream consumers of the data (dashboards, reports, other pipelines).
- For libraries: Search for all callers, including external codebases if this is an open-source library.
- For services: Find all services that depend on this service's contracts (requests, responses, events).

**7.4 Check All Upstream Producers**
- Find every place that produces data matching this contract.
- Understand the context in which it is produced.
- Check what the producer does and what assumptions it makes.
- Identify any fallback or default logic.

**7.5 Determine Backward Compatibility**
- Can you add fields without breaking existing consumers? (Yes, usually.)
- Can you remove fields without breaking existing consumers? (No, only if they are truly unused.)
- Can you rename fields without breaking existing consumers? (No, unless you provide a deprecation period.)
- Can you change field types without breaking consumers? (No, unless you are changing from a strict type to a union type that includes the old type.)
- Can you change response order or add optional fields? (Yes, for JSON; no, for positional tuple returns.)

**7.6 Update Consumers in the Same Phase**
- If you change a contract, update all consumers in the same phase.
- Do not change an API and leave the frontend or other backend services broken.
- Update tests in the same phase.
- Do not merge incomplete work.

**7.7 Do Not Casually Rename**
- Do not rename a field from `driver_name` to `driver` just because it looks cleaner.
- Do not change a field name without updating all consumers.
- If renaming is necessary, prefer adding an alias or deprecation period.
- Minimize naming churn; names should be stable unless there is a good reason to change them.

---

## 8. Prefer Existing Architecture

### Core Principle
Every project has established patterns, services, abstractions, and layers. Use them. Do not create parallel or duplicate implementations. This applies across all project types and domains.

### Common Subsystems and Where They Live

**8.1 Domain Logic and Business Rules**
- Examples: Strategy engines, recommendation systems, calculation engines, state machines, workflows.
- Do not create a second implementation of core business logic.
- If logic needs to be extended, modify the existing implementation.
- If you need variations, check whether the existing implementation can be parameterized.
- Do not have multiple places that implement the same business rule differently.

**8.2 Data Access and Loading**
- Examples: Database access layers, API clients, data loaders, connectors, repositories.
- The project has established patterns for loading and accessing data.
- Do not create new data loaders without checking for existing implementations.
- If a similar loader exists, reuse or extend it rather than creating a new one.
- If you need to load a new data type, follow the existing access pattern.

**8.3 External Service Integration**
- Examples: Third-party API clients, webhook handlers, message queue consumers, cache clients.
- Do not create multiple clients for the same external service.
- If the existing client does not support a needed feature, extend it.
- If a new external service is needed, add an integration that follows the existing pattern.

**8.4 Validation and Transformation**
- Examples: Input validation, schema validation, data transformation, sanitization.
- The project has validation functions for inputs, schemas, and contracts.
- Do not duplicate validation logic across multiple functions or modules.
- If validation is needed for a new type, check existing utilities first.
- If you need to add validation, extend the existing system rather than creating a new one.

**8.5 Configuration Management**
- Examples: Environment variables, config files, feature flags, settings stores.
- The project has an established way to load and manage configuration.
- Do not create a second configuration system.
- Use the existing mechanism for new configuration options.
- Do not hardcode configuration that should be configurable.

**8.6 Observability (Logging, Metrics, Tracing)**
- Examples: Logging frameworks, metrics collectors, error tracking, tracing systems.
- The project likely has established patterns for observability.
- Use the existing logging, metrics, and tracing libraries.
- Do not create new logging or monitoring implementations.
- Do not duplicate observability logic.

**8.7 Persistence and Caching**
- Examples: Database access patterns, cache implementations, session storage, message queues.
- The project has established patterns for persisting and accessing data.
- Do not create multiple ways to access the same data.
- Do not create parallel storage layers or caching strategies.
- Use the existing storage and cache abstraction.

### How to Find Existing Implementations
- Search the codebase for keywords related to what you need (e.g., "strategy", "loader", "client").
- Check the project structure and documentation to understand where common subsystems live.
- Ask or search Git history for similar features.
- When in doubt, read the architecture documentation or ask the team.

### When to Create Something New
Only create a new subsystem if:
- You have thoroughly searched and confirmed no existing implementation can be adapted.
- Creating something new is significantly simpler than adapting existing code.
- The new subsystem is truly different in purpose or design from existing ones.
- You have documented why an existing system cannot be used.
- Examples of when creating something new is appropriate:
  - A new external service needs to be integrated, and no similar integration exists.
  - A new persistent storage layer is needed for a completely different use case.
  - A new domain-specific calculation or logic engine is needed for a new area of the application.
  - The existing implementation cannot be extended without major refactoring, and the new subsystem is truly separate.

---

## 9. Real Data and Real Integrations

### Core Principle
If a feature claims to use real data sources (ML, telemetry, historical data, external APIs, databases, simulation, etc.) or backend subsystems, the production code path must actually reach and use that source or subsystem. No fake or hardcoded data that masquerades as real integrations. This applies across all project types.

### What This Means

**9.1 Verify the Production Execution Path**
- For a feature that claims to use machine learning:
  - Trace the execution path to confirm it actually loads the ML model.
  - Verify that it passes real input features to the model.
  - Confirm that the model output is used in the decision.
  - Check that the feature works even if the model is unavailable (graceful degradation).

- For a feature that claims to use external APIs or services:
  - Trace to confirm it calls the real external service, not a stub or mock.
  - Verify that it passes real request data.
  - Confirm that the response is actually used.
  - Check error handling when the service is unavailable.

- For a feature that claims to use database data or persistence:
  - Confirm it queries the actual database or storage layer, not hardcoded values.
  - Verify that the query is correct and returns real data.
  - Check that filtering, sorting, pagination, and aggregation work correctly.

- For a feature that claims to use historical data or caching:
  - Confirm it retrieves data from the actual cache or history store.
  - Verify that the data is real and up-to-date.
  - Check staleness and refresh logic.

- For a feature that claims to use vector stores or embeddings:
  - Trace to confirm it queries the vector store or embedding service.
  - Verify embeddings are calculated or retrieved from real sources.
  - Check that the feature works with real data.

- For a feature that claims to use a calculation engine or simulation:
  - Confirm it invokes the real engine or simulator.
  - Verify the engine receives correct inputs and returns valid results.
  - Check that results drive the feature behavior.

**9.2 Do Not Rename or Reframe Hardcoded or Simple Values as Complex**
- Do not rename a hardcoded number to make it sound more sophisticated.
  - Example: Don't rename `delay = 100` to `delay = AI_OPTIMIZED_LATENCY` if it is just a fixed constant.
  - Example: Don't name a simple formula `ML_PREDICTION = sum(factors) / len(factors)` and present it as ML-powered.
- Do not wrap a simple calculation in a function with an impressive name if it is not that complex.
  - Example: Don't name `threshold_check = value > 50` as `NEURAL_NETWORK_DECISION()`.
- Do not frame a heuristic or rule-based system as machine learning.
- If you are using a simple formula, heuristic, or hardcoded value, own it; do not pretend it is something fancier.

**9.3 Expose Limitations Honestly**
- If a feature depends on a backend system that is unavailable or incomplete, say so explicitly.
- Use a feature flag to disable the feature if dependencies are not ready.
- Return an error or fallback response that indicates the limitation.
- Do not silently use fake data when the real system is unavailable.

**9.4 Examples of "Decorative Data" and "Fake Integrations" to Avoid**
- Strategy calculation that returns a hardcoded decision without calculating it.
- ML-powered recommendation that actually uses a simple heuristic or weighted formula.
- Real-time data that actually returns pre-computed, cached results from days or weeks ago.
- Simulated or calculated output that actually runs through hardcoded replay or canned responses.
- Confidence or trust score that is always the same value (e.g., 0.95) regardless of input.
- API client that always returns mock data, with real API calls commented out.
- Database query that returns synthetic test data instead of querying the database.
- Search results that are hardcoded rather than fetched from a real search index or database.
- Recommendations that are generated from a rules engine but presented as from collaborative filtering or ML.

**9.5 How to Implement Features Correctly**
- If you need to use a real data source or backend subsystem, take the time to integrate it properly.
- If the subsystem is not ready or unavailable, use a feature flag, explicit preview mode, or clear error message.
- Document which features are powered by which sources or systems.
- Test with real or realistic data from the actual systems, not entirely synthetic test data (unless you are specifically building test tooling).
- When presenting the feature (in documentation, UI, etc.), be honest about what powers it and what its limitations are.
- Provide graceful degradation when real data sources are unavailable.

---

## 10. Error Handling

### Core Principle
Handle errors at the appropriate boundary. Users receive useful, safe responses without exposing implementation details, secrets, or stack traces.

### User-Facing Error Responses

**10.1 What to Expose to Users**
- A clear, human-readable error message.
- An error code or identifier that can be used to look up more information.
- Actionable guidance if applicable (e.g., "Please try again in 5 minutes" or "Check your input format").

**10.2 What to Never Expose to Users**
- Full stack traces.
- Internal file paths.
- Raw database errors or SQL queries.
- API keys, tokens, or credentials.
- Internal variable names or technical implementation details.
- Names of internal services or systems.

**10.3 Example: Good User-Facing Error**
```json
{
  "error": "INVALID_RACE_ID",
  "message": "The race ID provided does not exist or has been archived.",
  "suggestion": "Check the race ID and try again."
}
```

**10.4 Example: Bad User-Facing Error**
```json
{
  "error": "RuntimeError: NoneType object has no attribute 'event_date' in /home/prod/services/race_loader.py:127",
  "details": "TypeError at <module> Backend.getRace SQL: SELECT * FROM races WHERE id = null..."
}
```

### Server-Side Error Logging

**10.5 What to Log for Debugging**
- Full stack trace and exception details.
- Request parameters and context.
- Internal state at the time of failure.
- Relevant IDs (user ID, race ID, session ID).
- Timestamps and environment information.

**10.6 Example: Good Server-Side Log**
```
ERROR [2024-08-17T14:23:45Z] RaceService.calculateStrategy
  User: user_12345
  Race: race_98765
  Request: {"strategy_type": "aggressive", "weather": "wet"}
  Exception: KeyError 'pit_window'
  Traceback: ...
  Context: strategy_params incomplete, missing pit_window configuration
```

### Error Handling Patterns

**10.7 Catch Specific Exceptions**
- Never use `except Exception:` or `except:` to silently swallow all errors.
- Always catch the specific exception type you expect.
- If you catch an exception, either handle it or re-raise it.

```python
# Bad
try:
    result = risky_operation()
except:
    pass

# Good
try:
    result = risky_operation()
except TimeoutError:
    logger.warning("Risky operation timed out")
    result = fallback_value
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise
```

**10.8 Log Exceptions When Handled**
- If you catch an exception and do not re-raise it, log it.
- Include context so the error can be understood later.
- Do not silently ignore errors.

```python
try:
    telemetry = load_telemetry(race_id)
except TelemetryUnavailableError:
    logger.warning(f"Telemetry unavailable for race {race_id}, using fallback")
    telemetry = fallback_telemetry
```

**10.9 Make Fallback Behavior Explicit**
- If an error is handled with a fallback, document what the fallback is and why it is safe.
- If using a cached value instead of fresh data, note that.
- If using default parameters instead of calculated ones, note that.

```python
def get_weather(race_id):
    try:
        return weather_service.fetch(race_id)
    except WeatherServiceError:
        # Fallback to historical average weather for this race/location/season
        # This ensures the strategy calculation can continue, but may be less accurate
        logger.info(f"Using historical weather fallback for race {race_id}")
        return historical_weather.get_average(race_id)
```

**10.10 Handle Async Errors**
- In async code, ensure exceptions are not silently swallowed by unawaited promises or tasks.
- Always propagate exceptions up the call stack unless explicitly handled.
- Use try/catch around await statements.

### Error Handling Discipline

**10.11 Do Not Over-Catch**
- Do not add broad error handling for hypothetical failure cases.
- Handle real errors that actually occur in the codebase.
- Do not add try/catch blocks "just in case."

**10.12 Error Messages Should Be Useful**
- Include the key information needed to understand what went wrong.
- Include relevant IDs, states, and values.
- Use consistent terminology and formatting.

---

## 11. Validation

### Core Principle
Validate inputs at the boundary using the project's existing schema and validation mechanisms. Prefer strict validation over attempting to salvage invalid input.

### Validation at the Boundary

**11.1 Where to Validate**
- Validate at API endpoints (request body, query parameters, path parameters).
- Validate when reading from external systems.
- Validate when reading from databases or caches.
- Do not re-validate repeatedly as data flows through internal layers.

**11.2 Use Existing Schema and Validation**
- If the project uses Pydantic, JSON Schema, Protobuf, or a similar mechanism, use it.
- Define or extend schemas to cover all required fields and types.
- Use the project's validation framework to enforce schemas.
- Do not write custom validation code when the framework already exists.

```python
# Good: Using Pydantic
from pydantic import BaseModel, Field, validator

class StrategyRequest(BaseModel):
    race_id: str
    strategy_type: Literal["aggressive", "defensive", "balanced"]
    weather_condition: str
    fuel_level: float = Field(gt=0, le=100)
    
    @validator("race_id")
    def race_id_must_exist(cls, v):
        if not Race.exists(v):
            raise ValueError(f"Race {v} does not exist")
        return v

# Bad: Custom validation scattered around
def process_strategy(race_id, strategy_type, weather, fuel):
    if not race_id:
        raise ValueError("race_id required")
    if strategy_type not in ["aggressive", "defensive", "balanced"]:
        raise ValueError("invalid strategy_type")
    if not isinstance(weather, str):
        raise ValueError("weather must be string")
    # ... more manual checks
```

### Strict Validation Preferred Over Salvage

**11.3 Reject Invalid Input**
- If input does not match the expected schema, reject it.
- Return a clear error message indicating what is wrong.
- Do not try to "fix" or "interpret" invalid input.

```python
# Good: Strict validation
if user_id < 0 or user_id > 999999:
    raise ValueError("user_id must be between 0 and 999999")
user = User.get(user_id)

# Bad: Over-permissive
user_id_str = request.args.get("user_id", "0")
try:
    user_id = int(float(user_id_str))  # Trying to salvage the input
except:
    user_id = 0  # Silently default
user = User.get(user_id)  # Might get the wrong user
```

**11.4 Fail Fast**
- Validate and reject invalid input immediately.
- Do not proceed with processing if input is invalid.
- Do not assume invalid input will work out later.

### Validation Layers

**11.5 No Duplicate Validation**
- Do not validate the same input at multiple layers unless there is a specific security or correctness reason.
- Validate once at the boundary; trust the validation internally.
- If you need to re-validate, document why.

**11.6 Validation for Security**
- Validate user input strictly to prevent injection attacks, traversal attacks, etc.
- Validate data permissions (does the user have access to this resource?).
- Validate integrity (has the data been tampered with?).
- Use the project's security validation patterns.

---

## 12. Tests Are Part of the Implementation

### Core Principle
Tests are not afterthoughts. They are part of the implementation and must be added and maintained alongside code changes.

### Test-Driven Approach

**12.1 Find Existing Tests First**
- Before writing new code, find the existing tests for the module or feature.
- Read the tests to understand what behavior is already tested.
- Understand what the existing tests expect.
- Do not break existing tests.

**12.2 Add Tests for New Behavior**
- For every behavioral change, add tests that exercise the new behavior.
- Tests should verify the actual output, not just that the code runs.
- Tests should cover the happy path and important failure paths.

```python
# Good: Testing real behavior
def test_pit_stop_calculation_includes_exit_time():
    strategy = Strategy(pit_stop_duration=21, pit_exit_time=3)
    total = strategy.total_time()
    assert total == 24  # pit_stop + exit_time

def test_pit_stop_calculation_handles_zero_exit_time():
    strategy = Strategy(pit_stop_duration=21, pit_exit_time=0)
    total = strategy.total_time()
    assert total == 21

# Bad: Testing implementation details
def test_pit_stop_has_duration_attribute():
    strategy = Strategy(pit_stop_duration=21)
    assert hasattr(strategy, "pit_stop_duration")
    assert strategy.pit_stop_duration == 21
```

**12.3 Test Important Failure Paths**
- Test what happens when input is invalid.
- Test what happens when a required resource is unavailable.
- Test what happens when data is malformed or incomplete.
- Test error messages and error codes.

```python
def test_calculate_strategy_with_missing_telemetry():
    with pytest.raises(TelemetryUnavailableError):
        calculate_strategy(race_id="r123", telemetry=None)

def test_calculate_strategy_with_invalid_race_id():
    result = calculate_strategy(race_id="invalid")
    assert result.error_code == "RACE_NOT_FOUND"
    assert result.error_message is not None
```

### Test Quality Standards

**12.4 Keep Tests Deterministic**
- Tests should produce the same result every time they run.
- Do not rely on external state, random data, or the current time.
- Use fixed, predictable test data.
- Mock or stub external systems.
- Seed random number generators for reproducibility.

```python
# Good: Deterministic
def test_race_ranking_calculation():
    drivers = [
        Driver(name="Alice", points=100),
        Driver(name="Bob", points=95),
        Driver(name="Charlie", points=90),
    ]
    ranking = calculate_ranking(drivers)
    assert ranking[0].name == "Alice"
    assert ranking[1].name == "Bob"

# Bad: Non-deterministic
def test_race_ranking_calculation():
    # Relies on external database state
    ranking = calculate_ranking(fetch_drivers_from_db())
    assert len(ranking) > 0  # Weak assertion
    # Test might pass or fail depending on database state
```

**12.5 Do Not Weaken Assertions**
- Assertions should verify the actual expected behavior.
- Do not change assertions to make failing tests pass.
- If an assertion fails because the behavior changed, update the test to reflect the new expected behavior and understand why it changed.

```python
# Bad: Weakened assertion
def test_strategy_calculation():
    result = calculate_strategy(race_params)
    assert result is not None  # Too weak; doesn't verify correctness
    assert isinstance(result, Strategy)  # Still too weak

# Good: Specific assertion
def test_strategy_calculation():
    result = calculate_strategy(race_params)
    assert result.pit_stop_time == 21.5
    assert result.pit_window_start == 15
    assert result.pit_window_end == 25
    assert result.confidence_score > 0.85
```

**12.6 Right-Size Test Suites**
- Do not create massive test suites for trivial changes.
- A single simple function might need 3-5 tests (normal case, edge cases, error cases).
- A complex calculation might need 10-15 tests.
- A UI component might need 10-20 tests.
- Do not write 100 tests for a 10-line function.

**12.7 Test Real Behavior, Not Implementation Details**
- Tests should verify outputs and side effects, not the internal implementation.
- Avoid tests that break when you refactor the implementation (same behavior, different code).

```python
# Bad: Tests implementation details
def test_strategy_calls_telemetry_loader():
    with patch("strategy.load_telemetry") as mock:
        calculate_strategy(race_id)
        mock.assert_called_once()  # Tests how it works, not what it does

# Good: Tests behavior
def test_strategy_includes_current_telemetry():
    # Prepare test data with specific telemetry
    result = calculate_strategy(race_id, telemetry=test_telemetry)
    # Verify the result used the telemetry
    assert result.calculated_gap == expected_gap_based_on_telemetry
```

---

## 13. Verification Before Completion

### Core Principle
Before declaring a phase complete, run the appropriate verification set. Start with targeted checks, then run broader checks.

### Verification Strategy

**13.1 Start with Targeted Tests**
- For a backend change, run the tests for the specific module you modified.
- For a frontend change, run the tests for the component you modified.
- For an API change, run the tests for the affected endpoints.
- Targeted tests give quick feedback during development.

**13.2 Run the Full Relevant Suite**
- After targeted tests pass, run the full test suite for the affected layer.
- Backend: `pytest` or the full backend test suite.
- Frontend: `npm run test` or the full frontend test suite.
- Full end-to-end tests if the change spans multiple layers.

**13.3 Typical Verification Checks by Project Type**

**Backend (Python, Java, Go, Node.js, etc.)**
- **Compilation**: Language-specific compilation check (e.g., `python -m compileall backend/`, `javac`, `go build`).
- **Linting**: Run linter on modified files (e.g., `pylint`, `flake8`, `eslint`, `golangci-lint`).
- **Type Checking**: Run type checker if available (e.g., `mypy`, `pyright`, `tsc`).
- **Unit Tests**: Run tests for the modified module or package.
- **Integration Tests**: Run tests that exercise features end-to-end.
- **Database/Persistence**: If changes involve schema or persistence, verify migrations and queries work.

**Frontend (React, Vue, Angular, etc.)**
- **Build**: Ensure the project builds without errors (`npm run build`, `yarn build`, etc.).
- **Linting**: Run linter on modified files (`npm run lint`, `eslint`, etc.).
- **Type Checking**: Run type checker if using TypeScript (`npm run type-check`, `tsc --noEmit`).
- **Unit Tests**: Run tests for the modified component or module.
- **Visual Tests**: Manually verify UI changes in different browsers and screen sizes.
- **Accessibility**: If the change affects UI, check accessibility (keyboard navigation, screen readers, ARIA).

**Data Pipeline or Batch Processing**
- **Compilation/Syntax**: Check that the pipeline code is syntactically correct.
- **Unit Tests**: Test individual data transformation functions or steps.
- **Integration Tests**: Run the pipeline end-to-end with sample data.
- **Data Quality**: Verify output data is correct and matches expectations.
- **Performance**: Check that pipeline runs within acceptable time.

**Infrastructure and DevOps**
- **Syntax**: Validate infrastructure code (Terraform, CloudFormation, Docker, Kubernetes configs).
- **Linting**: Run linter on IaC code.
- **Unit Tests**: Run tests if available (e.g., Terraform plan, Docker build).
- **Integration Tests**: Deploy to a staging environment and verify it works.
- **Security**: Verify no secrets are exposed, permissions are correct, etc.

**Library or SDK**
- **Compilation**: Ensure the library compiles and builds.
- **Linting**: Run linter on the library code.
- **Unit Tests**: Test all public APIs and edge cases.
- **Integration Tests**: Build example applications using the library.
- **Documentation**: Verify that documentation is updated and accurate.
- **Backward Compatibility**: Test that existing consumers of the library still work.

**13.5 Use Targeted Tests First When Debugging**
- When a test fails, use targeted tests (just the failing test or just the test file) to iterate quickly.
- Add logging or temporary debugging code to understand the failure.
- Once fixed, run the full suite to ensure no regressions.

**13.6 Run Full Suite Before Completion**
- Once all targeted tests pass, run the full test suite for the affected layers.
- Do not declare a phase complete if the full suite has not been run.
- If the full suite takes a long time, document estimated runtime and status.

### When Verification Cannot Be Run

**13.7 Document When Checks Cannot Be Run**
- If a check cannot be run (missing environment, unavailable service, etc.), say so explicitly.
- Document what check could not run and why.
- Do not claim a check passed when it was not actually run.

```
Verification Report:
✓ Compilation: Passed
✓ Linting: Passed
✓ Unit Tests: 87 tests passed
✓ Integration Tests: 12 tests passed
✗ Manual API Testing: Could not run (staging environment unavailable)
  (Reason: Database migration not deployed to staging yet)
```

**13.8 Be Honest About Uncertainty**
- If you have not verified something, say so.
- Do not claim "production ready" if critical verification could not be done.
- Note which verifications were skipped and why.

---

## 14. Review the Diff

### Core Principle
Before committing or pushing, review the entire diff to catch unintended changes, debug code, temporary files, and accidental modifications.

### Diff Review Process

**14.1 Check Git Status**
```bash
git status --short
```
Review the list of modified and new files. Look for:
- Files you did not intend to modify.
- Debug scripts or temporary files.
- Generated files that should not be committed.
- Build artifacts or cache files.

**14.2 Review File-Level Changes**
```bash
git diff --stat
```
This shows a summary of changes in each file (lines added/removed). Look for:
- Files with unexpectedly large changes.
- Files you did not intend to touch.
- Files with changes that do not match the phase description.

**14.3 Review Full Diffs Line by Line**
```bash
git diff
```
For each modified file, review every change:

**14.4 Look for Unrelated Files**
- Have you modified files that are not part of this phase?
- If yes, understand why and consider separating them into a different commit.

**14.5 Look for Debug Code**
- `console.log()`, `print()`, or `println()` calls that were left in by accident.
- Temporary debugging variables or flags.
- Stack traces or error output in test data.

**14.6 Look for Temporary Scripts**
- One-off reproduction scripts or test files.
- Scratch code written during development and left behind.
- Commented-out experiments.

**14.7 Look for Accidental Formatting Changes**
- Indentation or spacing changes unrelated to your fix.
- Reformatting of entire files because your editor auto-formatted.
- Changes to line endings (CRLF vs LF).

**14.8 Look for Generated Files**
- Build artifacts (dist/, build/, *.pyc, node_modules/).
- Temporary cache files.
- IDE-generated files.
These should be in `.gitignore` and not committed.

**14.9 Look for Secrets**
- API keys, tokens, passwords, or credentials in code.
- Connection strings with embedded secrets.
- Private or internal URLs.

**14.10 Look for Large Unnecessary Changes**
- Entire functions rewritten when a line or two would suffice.
- Massive whitespace or formatting changes unrelated to the fix.
- Files that should not be different but are.

**14.11 Look for Dead Code**
- Commented-out code left behind.
- Functions that are defined but never called.
- Import statements for unused modules.
- Conditional branches that are unreachable.

### Staging and Committing

**14.12 Stage Only Intended Changes**
- Use `git add <file>` to stage files one at a time.
- Use `git add <file> <file>` to stage multiple files together.
- Do not use `git add .` when unrelated work is present; you will accidentally commit everything.

**14.13 Verify Before Committing**
```bash
git diff --cached
```
Review the staged changes one more time before committing. This is your last chance to catch mistakes.

---

## 15. No Temporary Junk

### Core Principle
Do not leave behind debug scripts, reproduction files, generated artifacts, or experimental code. Keep the repository clean.

### What Counts as Junk
- Debug scripts: `test_debug.py`, `debug_telemetry.js`, `check_race_data.py`.
- One-off reproduction files: `repro_bug_123.py`, `test_pit_calculation_manual.py`.
- Generated artifacts: output files, logs, dumps, exports (unless committed to a specific output directory).
- Local logs: `debug.log`, `.log` files.
- Scratch files: `scratch.py`, `temp.txt`, `notes.md` (unless version-controlled).
- Experimental modules: `strategy_v2.py`, `new_approach.js`, `alt_calculation.py`.

### Before Committing, Clean Up

**15.1 Remove All Temporary Files**
```bash
git clean -fd  # Show what would be removed
git clean -fdX # Remove untracked files (be careful!)
```
Or manually remove files with `rm`.

**15.2 Remove Commented-Out Code**
- If code is commented out "just in case," remove it.
- Version control systems exist; the code is not lost.
- Commented code clutters the codebase and confuses future readers.

**15.3 Keep Genuinely Useful Utilities**
- If a reproduction script or debug utility is genuinely useful for understanding or maintaining the code, put it in a proper location.
- Examples: `tools/debug_telemetry.py`, `scripts/test_strategy_offline.py`, `dev/check_race_data.py`.
- Document what the utility does and how to use it.

---

## 16. Dependency Discipline

### Core Principle
Do not add or upgrade dependencies casually. Dependencies increase complexity, attack surface, and maintenance burden.

### Before Adding a Dependency

**16.1 Check for Existing Implementations**
- Search the project for similar functionality.
- Check whether an existing dependency already solves the problem.
- If a built-in language feature or standard library function exists, use it first.

**16.2 Verify Necessity**
- Is the dependency truly necessary?
- Would a small amount of custom code be simpler than adding a dependency?
- Can the dependency be optional or dev-only?

**16.3 Check Compatibility**
- Does the dependency work with the project's Python/Node version?
- Does it work with the OS and architecture in production?
- Are there known conflicts with existing dependencies?

**16.4 Understand the Trade-off**
- What does the dependency bring? (Functionality, performance, maintainability.)
- What does it cost? (Size, startup time, maintenance, attack surface.)
- Is the trade-off worth it?

### Managing Dependency Changes

**16.5 Do Not Upgrade Casually**
- Dependency upgrades can introduce breaking changes, new bugs, or performance regressions.
- Only upgrade dependencies when necessary (security fix, bug fix, new feature needed).
- Test thoroughly after upgrading.

**16.6 Separate Dependency Changes**
- Do not mix dependency upgrades with feature or bug-fix work.
- If an upgrade is needed, do it in a separate phase or commit.
- Document why the upgrade was necessary.

**16.7 Run Relevant Builds and Tests**
- After adding or upgrading a dependency, run the full build and test suite.
- Verify that the project still builds and all tests pass.
- Check for any performance regressions.

---

## 17. Security-Sensitive Changes

### Core Principle
Security changes require special care. Prefer established libraries and framework mechanisms. Never invent cryptography or hardcode secrets.

### Security Change Process

**17.1 Use Established Libraries**
- Do not implement your own cryptography, hashing, or authentication.
- Use well-known, audited libraries: `bcrypt`, `argon2`, `nacl`, `cryptography`, etc.
- Use the framework's built-in security features (CSRF tokens, CORS, session management).

**17.2 Never Hardcode Secrets**
- API keys, tokens, passwords, and credentials must never be in code.
- Load secrets from environment variables or secure configuration systems.
- Never commit secrets to version control.

**17.3 Never Expose Secrets to the Frontend**
- Do not send API keys or tokens to the browser.
- Do not include secrets in JavaScript bundles.
- Do not log secrets in client-side code.

**17.4 Do Not Weaken Validation or Authentication**
- Do not add bypass logic to pass tests.
- Do not skip authentication for "local development only."
- Do not disable CSRF protection because "it's easier."
- Security features exist for a reason.

**17.5 Keep Security Changes Narrowly Scoped**
- A security fix should fix the security issue, nothing else.
- Do not use a security issue as an excuse to refactor everything.
- Do not mix security fixes with feature work.

**17.6 Test Security Changes**
- Add tests that verify the security feature works.
- Test both positive cases (authorized access works) and negative cases (unauthorized access fails).
- Test edge cases and error conditions.

---

## 18. Preserve Working Behavior

### Core Principle
APEXiq is an existing project with working features. Do not "clean up" or redesign working behavior unless the phase explicitly requires it.

### Before Changing Behavior

**18.1 Understand Current Dependencies**
- Before changing how something works, identify what currently depends on that behavior.
- Check all code paths that consume the affected function or data.
- Check tests that exercise the behavior.
- Check user-facing features that depend on it.

**18.2 Prefer Safe Changes Over Clean Ones**
- A smaller safe change that preserves compatibility is better than a cleaner-looking breaking change.
- If you must break compatibility, do it intentionally and update all consumers.

**18.3 Document Why Behavior Changed**
- If you change working behavior, document why.
- Explain what the old behavior was and why it needed to change.
- Note which systems depend on the change.

---

## 19. Phase Completion Report

### Core Principle
At the end of every phase, produce a clear report documenting what was changed, what was verified, and what was deliberately left out.

### Completion Report Structure

**19.1 Changed Section**
- **Files Changed**: List the files that were added, modified, or deleted.
- **What Was Changed**: Describe what was modified in each file (new functions, modified logic, deleted code, etc.).
- **Why It Was Necessary**: Explain the reason for each change; how does it solve the problem?

Example:
```
Changed:
- strategy_engine.py
  * Added pit_exit_time parameter to pit_stop_calculation()
  * Modified total_time() to include pit_exit_time in the calculation
  * Why: The pit exit time was previously hardcoded as 0; actual exit time varies by track.

- pit_logic.py
  * Added load_pit_exit_times() function
  * Why: Needed to retrieve pit exit times from the configuration service.

- tests/test_strategy_engine.py
  * Added test_pit_calculation_includes_exit_time()
  * Why: Verify the new pit_exit_time parameter is correctly included in total time.
```

**19.2 Verified Section**
- **Tests Run**: List which tests were run and their results.
  - Unit tests: "All 87 tests in test_strategy_engine.py passed."
  - Integration tests: "10 end-to-end tests passed."
  - Build: "Backend build successful. No compilation errors."
  - Linting: "No linting errors or warnings."
- **Manual Verification**: If manual testing was done, describe it.
  - "Tested pit stop calculation with five different track configurations; times match expected values."
  - "Tested the API endpoint with curl; returns correct JSON format."

Example:
```
Verified:
✓ Compilation: python -m compileall backend/ — Passed
✓ Linting: pylint strategy_engine.py pit_logic.py — Passed
✓ Type Checking: mypy backend/ — Passed
✓ Unit Tests: pytest tests/test_strategy_engine.py — 87 tests passed
✓ Integration Tests: pytest tests/integration/ — 10 tests passed
✓ Manual API Testing: Tested via curl; correct response format
✗ Staging Deployment: Could not run (staging DB not synced)
```

**19.3 Not Changed Section**
- **Deliberately Out of Scope**: List important related issues or work that was NOT done in this phase.
  - "Did not refactor pit_logic.py for performance; that is a separate optimization task."
  - "Did not update frontend UI to display pit exit time; that is handled in the frontend phase."
  - "Did not migrate historical race data to new schema; that will be done in a data migration phase."

This section shows that you understand the broader picture but have stayed focused.

**19.4 Remaining Section**
- **Genuine Bugs or Technical Debt**: List any issues discovered during this phase that were not addressed.
  - Format: Issue + Impact + Next Steps
  - Example: "Discovered that pit_exit_times service returns null for some older tracks, causing calculation to fail silently. Mitigation: Added fallback to default value. Proper fix: Track maintenance team should provide historical exit times."
  - Clearly distinguish blockers from optional follow-up work.

**19.5 Production Readiness Statement**
- Do not claim "production ready" unless the available verification supports that claim.
- If critical verification could not be done, note it.

Example:
```
Remaining Issues:
- [OPTIONAL FOLLOW-UP] Performance: Pit calculation is now O(n) where n = number of pit stops. For races with 50+ stops, this might be slow. Could optimize to O(1) by caching exit times by track.
- [OPTIONAL FOLLOW-UP] Metrics: No telemetry added yet for pit calculation performance. Consider adding once deployment monitoring is in place.
- [BLOCKER] Staging Database: Historical pit exit times not loaded into staging DB yet. Cannot fully verify end-to-end until staging is updated.

Production Readiness:
✓ Ready for production deployment
✓ All unit and integration tests passing
✓ Code review completed and approved
✗ Depends on: Staging database update (unrelated task, in progress)
```

---

## 20. Git Discipline

### Core Principle
Use Git carefully and deliberately. Commits should be meaningful and history should be clean.

### Before Committing

**20.1 Do Not Commit Automatically**
- Do not run `git add .` and `git commit` reflexively.
- Take time to review what you are committing.
- Ensure each commit contains related changes only.

**20.2 When Asked to Commit**
- Review the diff one final time.
- Stage only the files that belong to this commit.
- Write a clear, conventional commit message.
- Verify the commit looks correct.

### Commit Messages

**20.3 Clear Conventional Commit Messages**
- Use the conventional commit format: `<type>(<scope>): <subject>`
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `chore`, `perf`.
- Subject should be imperative and concise (50 characters or less).
- Include a body if more context is needed (wrap at 72 characters).

Examples:
```
fix(strategy): include pit exit time in total calculation

The pit exit time varies by track but was hardcoded to 0.
Load the exit time from the configuration service.

Fixes #456
Closes #789
```

```
test(strategy): add tests for pit calculation with exit time
```

```
refactor(pit_logic): extract pit window calculation into helper function
```

### Pushing Changes

**20.4 Atomic File-Level Commit and Push Workflow**

When working on bounded phases where changes are broken into individual files, the following mandatory workflow applies:

1. **Before modifying a file**, verify the working tree is clean (`git status`).
2. **Modify exactly ONE file.** Do not touch a second file until the first is committed and pushed.
3. **After completing that file:**
   - Run targeted tests relevant to that file.
   - Run applicable lint/format checks (e.g. Ruff).
   - Inspect the diff (`git diff -- <file>`).
   - Confirm only that file is modified (`git status`).
4. **Stage ONLY that file:**
   ```
   git add <file>
   ```
5. **Verify exactly one file is staged:**
   ```
   git diff --cached --name-only
   ```
   There must be exactly one file listed.
6. **Commit immediately** with a focused, meaningful message.
7. **Push immediately** to the configured remote/main branch.
8. **Verify:**
   - Working tree is clean.
   - Local branch is synchronized with remote.
   - Push succeeded.
9. **Only then** may the next file be modified.

**Prohibited at all times:**
- `git add .` or `git add -A` when this would stage unrelated files.
- Modifying two files before committing the first.
- Batching source + tests into a single commit.
- Batching unrelated fixes.
- Pushing only at the end of a phase.
- Creating artificial, empty, or no-op commits.

**File dependency:** If a logical change genuinely requires a second file (e.g. implementation + its import in `__init__.py`), STOP before modifying the second file. Explain the dependency, identify the files, and wait for approval before proceeding.

**Purpose:** Atomic reviewable changes, immediate remote backup, clear Git history, easier rollback, disciplined engineering. This is not commit farming — it is legitimate engineering granularity.

This rule applies to: source files, tests, configuration files, documentation, scripts, CI/build files, and generated project metadata when tracked.

**20.5 Push Only When Explicitly Requested**
- Outside the atomic file-level workflow, do not push automatically after committing.
- Wait for approval or explicit instruction to push.

**20.6 Never Rewrite History, Force-Push, or Discard Changes**
- Do not use `git reset --hard`, `git rebase -i`, or `git push -f` without explicit approval.
- Do not change, combine, or reorder commits that have been pushed.
- Do not discard user work without explicit confirmation.

If a mistake is made:
- Create a new commit that fixes it, rather than rewriting history.
- If history rewrite is truly necessary, get explicit approval first.

---

## Golden Rule

### The Principle
**Understand first. Change the minimum. Verify the result. Leave the codebase better, not bigger.**

The best implementation is not the one with the most code. It is the smallest legitimate implementation that solves the actual problem and fits naturally into the existing APEXiq architecture.

### Applying the Golden Rule

**Before Coding**
- Read the existing implementation and understand it completely.
- Trace the execution path to see what actually happens.
- Understand the problem you are solving, not just the symptom.
- Check what similar features do in the codebase.

**While Coding**
- Aim for the smallest change that solves the problem.
- Resist the urge to refactor, redesign, or "improve" things.
- Reuse existing patterns, utilities, and abstractions.
- Do not write code you do not need.

**After Coding**
- Verify that your change actually solves the problem.
- Verify that you have not broken anything else.
- Review the diff and remove any unnecessary changes.
- Leave the codebase clear, simple, and maintainable.

**Throughout Development**
- Prefer clarity over cleverness.
- Prefer simplicity over sophistication.
- Prefer proven patterns over novel designs.
- Prefer working code over elegant code.

A five-line fix is better than a five-hundred-line refactor. A simple solution is better than a complex one. A focused phase is better than sprawling work. Keep this principle in mind with every decision.

---

## Summary

OpenCode Working Rules demand:
1. Deep understanding before changes.
2. Small, focused phases with clear scope.
3. Legitimate, production-quality code.
4. Minimal, efficient implementations.
5. Natural, human-readable code.
6. Sparse, meaningful comments.
7. Stable, well-defined contracts.
8. Reuse of existing architecture.
9. Real data and real integrations.
10. Proper error handling.
11. Boundary-based validation.
12. Thorough testing.
13. Comprehensive verification.
14. Careful diff review.
15. Clean repository state.
16. Disciplined dependency management.
17. Security-first approach.
18. Respect for existing behavior.
19. Detailed phase reports.
20. Git discipline.

**The goal**: Code that looks like it was written by a careful human engineer who understood the system, solved the actual problem, and cared about the codebase.

---

## Applying These Rules to Different Project Types

These OpenCode Working Rules apply universally across all software projects. Here are some project-type-specific highlights:

### Backend Services
- Focus on rule 7 (contracts): API contracts are critical; changes ripple through clients.
- Focus on rule 8 (architecture): Reuse service patterns, database access patterns, error handling.
- Focus on rule 13 (verification): Run database migrations, integration tests, and end-to-end tests.

### Frontend Applications
- Focus on rule 7 (contracts): Component props and API contracts must be stable.
- Focus on rule 8 (architecture): Reuse UI patterns, state management, styling approaches.
- Focus on rule 13 (verification): Test in multiple browsers and screen sizes; check accessibility.

### Full-Stack Systems
- Apply all rules equally; changes in one layer affect another.
- Pay special attention to rule 7 (contracts) for API boundaries between frontend and backend.

### Data Pipelines and Analytics
- Focus on rule 9 (real data): Use real data, not test fixtures.
- Focus on rule 12 (tests): Test data transformations thoroughly.
- Focus on rule 13 (verification): Verify output data quality and pipeline performance.

### Infrastructure and DevOps
- Focus on rule 17 (security): Never hardcode secrets; use secure configuration.
- Focus on rule 16 (dependencies): Carefully manage tool and library versions.
- Focus on rule 13 (verification): Test IaC changes in staging before production.

### Libraries and Open-Source Projects
- Focus on rule 7 (contracts): Public APIs are contracts with all downstream users.
- Focus on rule 18 (preserve behavior): Breaking changes must be intentional and well-documented.
- Focus on rule 12 (tests): Thoroughly test all public APIs.

### Mobile Applications
- Focus on rule 5 (humanized code): Follow platform conventions (iOS, Android).
- Focus on rule 13 (verification): Test on real devices and emulators.
- Focus on rule 8 (architecture): Reuse patterns for networking, storage, navigation.

1. Read opencode.md FIRST.
2. Treat its rules as mandatory engineering constraints.
3. Inspect git status before touching anything.
4. Understand the existing implementation and trace the real execution path.
5. Search for existing implementations before creating anything new.
6. Define the phase scope and affected files.
7. Make the smallest legitimate change.
8. Do not overcode or rewrite unrelated files.
9. Keep comments sparse and meaningful.
10. Do not introduce fake/decorative intelligence or hardcoded behavior pretending to be real.
11. Add focused tests for behavioral changes.
12. Run targeted verification, then the full relevant verification.
13. Review git diff carefully.
14. Report changed / verified / not changed / remaining.
15. Commit or push only when explicitly instructed.