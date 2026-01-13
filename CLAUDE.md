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

---

## Architecture & Extensibility

### Design Philosophy
This project may grow to support:
- **Partial tiling**: Tile some dimensions but not others (e.g., tile i and j, but not k)
- **Different operations**: Convolution, other tensor operations beyond matmul
- **Dynamic tensor configurations**: Variable number of input/output tensors

Code should be structured to make these additions **safe and incremental**. New features should not break existing ones.

### Current Coupling Points (Refactor Candidates)

When adding new features, be aware of these tightly-coupled areas that may need abstraction:

#### 1. Operation Definition (currently hardcoded to matmul)
```
Location: executeStep(), renderAllMatrices(), updateStateDisplay(), code display
Current: C[i][j] += A[i][k] * B[k][j] hardcoded everywhere
Future: Abstract "Operation" concept with:
  - Tensor list (names, dimensions, layouts)
  - Access pattern (which indices access which tensor)
  - Iteration space (loop dimensions and bounds)
  - Display strings (for code and state panels)
```

#### 2. Tiling Configuration (currently all-or-nothing)
```
Location: state.tilingEnabled (boolean), generateTiledIterations()
Current: Either all dimensions tiled or none
Future: Per-dimension config, e.g.:
  { i: { tiled: true, size: 4 }, j: { tiled: true, size: 4 }, k: { tiled: false } }
```

#### 3. Tensor/Matrix Configuration (currently fixed A, B, C)
```
Location: state.stats, state.layoutA/B/C, BASE_A/B/C, HTML canvas elements
Current: Exactly 3 matrices with hardcoded names
Future: Dynamic tensor list with configurable properties
```

#### 4. Loop Dimensions (currently fixed to 3)
```
Location: All loop order arrays, iteration generators
Current: Always [i, j, k] with 6 permutations
Future: Variable dimension count (convolution has 7: n, c, h, w, kc, kh, kw)
```

#### 5. UI Elements (currently static HTML)
```
Location: index.html matrix containers, stats displays
Current: Fixed 3-matrix layout
Future: Dynamically generated based on operation
```

### Refactoring Guidelines

**When to abstract**: Only when actually implementing a feature that needs it. Don't pre-abstract.

**How to abstract safely**:
1. Write tests/verification for existing behavior first
2. Extract the abstraction
3. Verify existing behavior still works
4. Add new feature using the abstraction
5. Verify both old and new work

**Abstraction priority** (when the time comes):
1. **Operation abstraction** - Most impactful, enables different computations
2. **Per-dimension tiling** - Natural extension, moderate refactor
3. **Dynamic tensors** - Requires HTML generation, larger change

### Code Organization Principles

**Current structure** (adequate for matmul-only):
```
app.js - Single file with sections:
  - Constants
  - State
  - Iteration generation
  - Cache model (already well-encapsulated)
  - Memory addressing
  - Rendering
  - Simulation
  - Code display
  - UI handlers
```

**Future structure** (when complexity warrants):
```
src/
  operations/
    matmul.js       - Matmul-specific: access pattern, code template
    convolution.js  - Conv-specific: access pattern, code template
  core/
    cache.js        - CacheSimulator (already modular)
    iteration.js    - Generic iteration generator
    state.js        - Application state management
  ui/
    renderer.js     - Matrix/tensor rendering
    controls.js     - Playback, configuration
    code-display.js - Loop code generation
  app.js            - Main entry, wiring
```

**Don't split prematurely**. Current single-file structure is fine for matmul. Split when:
- Adding a second operation type
- File exceeds ~1500 lines
- Clear module boundaries emerge from features

### Testing Strategy for New Features

Before implementing a new feature:
1. **Document expected behavior** for edge cases
2. **Verify current behavior** is correct (manual testing with known configurations)
3. **Identify what should NOT change** (regression prevention)

After implementing:
1. **Test the new feature** with various configurations
2. **Re-test existing features** to catch regressions
3. **Test combinations** (e.g., partial tiling + column-major)

### Feature Addition Checklist

When adding a feature like "partial tiling" or "convolution":

- [ ] Update IMPLEMENTATION_PLAN.md with feature spec
- [ ] Identify which coupling points need abstraction
- [ ] Create todos for incremental implementation
- [ ] Implement abstraction layer (if needed)
- [ ] Verify existing features still work
- [ ] Implement new feature
- [ ] Update UI controls
- [ ] Update code display
- [ ] Test combinations with existing features
- [ ] Update this architecture section if patterns change

---

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
  index.html              # HTML structure only
  styles.css              # All CSS styles
  app.js                  # All JavaScript logic
  CLAUDE.md               # This file - session instructions
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
- **Don't pre-abstract** - only refactor when a feature actually needs it

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

**Remember**: Incremental, rigorous, spec-driven. Claude owns the code; user owns the algorithms. Refactor when needed, not before.
