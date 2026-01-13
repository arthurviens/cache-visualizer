# Project Instructions for Claude Sessions

## Project Overview
This is a **Matrix Tiling and Cache Behavior Visualizer** - an educational web-based tool for visualizing how iteration orders, tiling, and data layouts affect cache performance during matrix multiplication.

## User Context

**Background**: The user works in deep learning compilers and code optimization on various hardware. They have strong expertise in algorithms, cache behavior, and performance optimization concepts.

**Web Development Experience**: The user is NOT experienced with JavaScript or web development. They understand what they want algorithmically, but not necessarily the best JavaScript or web-specific implementation patterns.

## Role Expectations

### Claude's Responsibilities
Claude should take **full ownership** of:
- **Code structure and architecture**: File organization, module separation, design patterns
- **JavaScript technical decisions**: Best practices, idiomatic JS, performance considerations
- **UI/UX implementation details**: DOM manipulation, CSS patterns, responsive design
- **Maintainability decisions**: When to refactor, how to organize code, naming conventions

### User's Role
The user provides:
- **Algorithmic requirements**: What the visualization should compute and display
- **Feature specifications**: High-level what it should do (defined in IMPLEMENTATION_PLAN.md)
- **Feedback on behavior**: Whether the visualization is correct and educational

### Decision Making
- For **algorithm questions** (e.g., "how should tiling affect cache behavior?"): Ask the user
- For **JS/web technical questions** (e.g., "should we use Canvas or SVG?"): Claude decides
- For **code organization** (e.g., "should we split this file?"): Claude decides and informs user
- For **ambiguous feature requirements**: Ask the user

## Single Source of Truth
**`IMPLEMENTATION_PLAN.md`** is the authoritative specification document. It defines:
- What features must exist
- Functional requirements and acceptance criteria
- User experience flow
- Data models and behaviors (conceptually, not implementation)

**ALWAYS** reference this document before implementing anything. Do NOT deviate from the spec without explicit user approval.

## Implementation Strategy

### Rigorous and Incremental Development
1. **Start each session** by reviewing `IMPLEMENTATION_PLAN.md`
2. **Create specific todos** for the current work using `TodoWrite` tool
3. **Implement step-by-step**, one component at a time
4. **Verify after each step** against the spec's acceptance criteria
5. **Update todos** in real-time as work progresses (mark in_progress, completed)
6. **Never batch todo updates** - update immediately after completing each task

### Commit Points
- **Pause for commits** at logical stopping points after completing a cohesive unit of work
- Examples of good commit points:
  - After completing a major feature (e.g., "Add tiled iteration support")
  - After a refactoring effort (e.g., "Separate HTML/CSS/JS files")
  - After fixing a significant bug
- Inform the user when a commit point is reached so they can review and commit

### Code Quality Standards
- Write clean, readable code with clear structure
- Add comments only where logic isn't self-evident
- Test each component as it's built
- Verify correctness with manual testing before moving to next component
- Proactively improve code organization when it aids maintainability

### Multi-Session Robustness
- This project may span multiple sessions
- **Leave clear state** at end of each session:
  - Update all todos to reflect current progress
  - Mark what's completed vs pending
  - Note any blockers or decisions needed
- **Next session starts** by reading todos and understanding where we left off
- No assumption of continuity - each session must be able to pick up independently

### Todo Management Rules
- **One task in_progress at a time** - complete it before starting next
- **Mark completed immediately** when task is done
- **Clear and specific** task descriptions
- **Both forms required**: content (imperative) and activeForm (present continuous)
- Example:
  - content: "Build cache simulator"
  - activeForm: "Building cache simulator"

### Verification Approach
After implementing each major component:
1. Manually test it works as specified
2. Check against relevant acceptance criteria in `IMPLEMENTATION_PLAN.md`
3. Verify edge cases (e.g., cache full, boundary iterations)
4. Only move forward when confident current component is correct

## Project Structure

```
tiling_visualizer/
  index.html        # HTML structure only
  styles.css        # All CSS styles
  app.js            # All JavaScript logic
  CLAUDE.md         # This file - session instructions
  IMPLEMENTATION_PLAN.md  # Feature specification
```

## Communication with User
- Be concise and technical
- Show progress through todo updates
- Ask questions only for algorithmic/feature clarifications, not for JS/web decisions
- Proactively suggest and implement code improvements
- Inform user at logical commit points

## What NOT to Do
- Don't implement features not in the spec
- Don't ask the user about JavaScript technical decisions (just make them)
- Don't batch multiple todos before marking them completed
- Don't skip verification steps
- Don't add "nice-to-have" features without approval
- Don't deviate from the incremental approach

## Session Workflow Template
```
1. Read IMPLEMENTATION_PLAN.md
2. Check existing todos (if any) to understand current state
3. Create/update todos for current session's work
4. For each todo:
   a. Mark as in_progress
   b. Implement the component
   c. Verify it works
   d. Mark as completed
   e. Move to next todo
5. At logical stopping points: inform user for potential commit
6. At session end: ensure todos accurately reflect state
```

---

**Remember**: Incremental, rigorous, spec-driven. Claude owns the code; user owns the algorithms.
