/**
 * Matrix Tiling and Cache Behavior Visualizer
 *
 * An educational tool for visualizing how iteration orders, tiling strategies,
 * and data layouts affect cache behavior during matrix operations.
 */

// =============================================================================
// OPERATION ABSTRACTION
// =============================================================================
//
// An Operation defines:
// - tensors: The tensors involved (name, dimensions, base address)
// - loopDims: The loop dimensions (e.g., ['i', 'j', 'k'] for matmul)
// - getAccesses(indices): Given loop indices, returns array of memory accesses
// - codeTemplate: How to display the inner statement
// - describeOp(indices): Human-readable description of current operation
//
// This abstraction allows adding new operations (convolution, etc.) without
// changing the core simulation logic.
// =============================================================================

/**
 * Creates the matrix multiplication operation.
 * C[i][j] += A[i][k] * B[k][j]
 *
 * @param {number} size - Matrix dimension (e.g., 12)
 * @param {number} elementSize - Bytes per element (e.g., 4)
 * @returns {Object} Operation definition
 */
function createMatmulOperation(size, elementSize) {
    const elementsPerMatrix = size * size;

    return {
        name: 'matmul',
        displayName: 'Matrix Multiplication',
        size: size,
        elementSize: elementSize,

        // Loop dimensions for this operation
        loopDims: ['i', 'j', 'k'],

        // All valid loop orderings
        loopOrders: {
            'ijk': ['i', 'j', 'k'],
            'ikj': ['i', 'k', 'j'],
            'jik': ['j', 'i', 'k'],
            'jki': ['j', 'k', 'i'],
            'kij': ['k', 'i', 'j'],
            'kji': ['k', 'j', 'i']
        },

        // Tensor definitions
        // Each tensor has: name, baseAddress, accessIndices (which loop vars index it)
        tensors: [
            {
                name: 'A',
                baseAddress: 0,
                rows: size,
                cols: size,
                // A[i][k] - row is i, col is k
                getIndices: (iter) => ({ row: iter.i, col: iter.k })
            },
            {
                name: 'B',
                baseAddress: elementsPerMatrix * elementSize,
                rows: size,
                cols: size,
                // B[k][j] - row is k, col is j
                getIndices: (iter) => ({ row: iter.k, col: iter.j })
            },
            {
                name: 'C',
                baseAddress: 2 * elementsPerMatrix * elementSize,
                rows: size,
                cols: size,
                // C[i][j] - row is i, col is j
                getIndices: (iter) => ({ row: iter.i, col: iter.j })
            }
        ],

        // Inner statement for code display
        codeTemplate: 'C[i][j] += A[i][k] * B[k][j]',

        // Human-readable operation description
        describeOp: (iter) => `A[${iter.i}][${iter.k}] × B[${iter.k}][${iter.j}] → C[${iter.i}][${iter.j}]`,

        // Total iterations
        getTotalIterations: () => size * size * size
    };
}

// =============================================================================
// CONSTANTS
// =============================================================================

const MATRIX_SIZE = 12;
const ELEMENT_SIZE = 4; // bytes per element

// Create the operation (can be swapped for different operations in the future)
const operation = createMatmulOperation(MATRIX_SIZE, ELEMENT_SIZE);

const TOTAL_ITERATIONS = operation.getTotalIterations();

// Rendering constants
const CELL_SIZE = 20; // pixels per cell
const COLORS = {
    background: '#ffffff',
    grid: '#cccccc',
    tileGrid: '#666666',
    cached: 'rgba(40, 167, 69, 0.6)',
    current: '#000000',
    currentOutline: '#667eea'
};

// =============================================================================
// APPLICATION STATE
// =============================================================================

/**
 * Initialize tensor state (layouts, stats) from operation definition.
 */
function createTensorState(op) {
    const layouts = {};
    const stats = {};

    for (const tensor of op.tensors) {
        layouts[tensor.name] = 'row'; // default to row-major
        stats[tensor.name] = { accesses: 0, hits: 0 };
    }

    return { layouts, stats };
}

const tensorState = createTensorState(operation);

const state = {
    // Configuration (from UI)
    loopOrder: 'ijk',
    tilingEnabled: false,
    tileSize: 4,
    layouts: tensorState.layouts,  // { A: 'row', B: 'row', C: 'row' }
    elementsPerLine: 16,  // elements per cache line
    numCacheLines: 4,     // number of cache lines

    // Simulation state
    currentIteration: 0,
    isPlaying: false,
    speed: 10,

    // Generated data
    iterations: [],
    cache: null,

    // Statistics (per-tensor)
    stats: tensorState.stats,  // { A: {accesses, hits}, B: {...}, C: {...} }

    // History for timeline
    history: []  // Array of { A: hit/miss, B: hit/miss, C: hit/miss }
};

// Canvas contexts
let canvasContexts = {};  // { A: ctx, B: ctx, C: ctx }
let ctxTimeline;

// Animation state
let animationId = null;
let lastFrameTime = 0;
let snapshots = [];

// =============================================================================
// ITERATION GENERATOR
// =============================================================================

/**
 * Generate iteration sequence for non-tiled execution.
 *
 * @param {Object} op - Operation definition
 * @param {string} loopOrder - Loop nesting order (e.g., 'ijk')
 * @returns {Array} Sequence of index objects
 */
function generateIterations(op, loopOrder) {
    const iterations = [];
    const order = op.loopOrders[loopOrder];

    if (!order) {
        throw new Error(`Unknown loop order: ${loopOrder}`);
    }

    const size = op.size;

    for (let outer = 0; outer < size; outer++) {
        for (let middle = 0; middle < size; middle++) {
            for (let inner = 0; inner < size; inner++) {
                const indices = {};
                indices[order[0]] = outer;
                indices[order[1]] = middle;
                indices[order[2]] = inner;

                // Create iteration object with all loop dimensions
                const iter = {};
                for (const dim of op.loopDims) {
                    iter[dim] = indices[dim];
                }
                iterations.push(iter);
            }
        }
    }

    return iterations;
}

/**
 * Generate iteration sequence for tiled execution.
 *
 * @param {Object} op - Operation definition
 * @param {string} loopOrder - Loop nesting order
 * @param {number} tileSize - Tile dimension
 * @returns {Array} Sequence of index objects with tile info
 */
function generateTiledIterations(op, loopOrder, tileSize) {
    const iterations = [];
    const order = op.loopOrders[loopOrder];

    if (!order) {
        throw new Error(`Unknown loop order: ${loopOrder}`);
    }

    const size = op.size;
    const numTiles = size / tileSize;

    // Outer tile loops
    for (let tOuter = 0; tOuter < numTiles; tOuter++) {
        for (let tMiddle = 0; tMiddle < numTiles; tMiddle++) {
            for (let tInner = 0; tInner < numTiles; tInner++) {
                // Calculate tile base indices
                const tileIndices = {};
                tileIndices['t' + order[0]] = tOuter * tileSize;
                tileIndices['t' + order[1]] = tMiddle * tileSize;
                tileIndices['t' + order[2]] = tInner * tileSize;

                // Inner element loops
                for (let eOuter = 0; eOuter < tileSize; eOuter++) {
                    for (let eMiddle = 0; eMiddle < tileSize; eMiddle++) {
                        for (let eInner = 0; eInner < tileSize; eInner++) {
                            const elemIndices = {};
                            elemIndices[order[0]] = eOuter;
                            elemIndices[order[1]] = eMiddle;
                            elemIndices[order[2]] = eInner;

                            // Build iteration object
                            const iter = {};
                            for (const dim of op.loopDims) {
                                const tileBase = tileIndices['t' + dim];
                                iter[dim] = tileBase + elemIndices[dim];
                                iter['t' + dim] = tileBase;  // tile base
                                iter['l' + dim] = elemIndices[dim];  // local offset
                            }
                            iterations.push(iter);
                        }
                    }
                }
            }
        }
    }

    return iterations;
}

/**
 * Generate all iterations based on current configuration.
 */
function generateAllIterations() {
    if (state.tilingEnabled) {
        return generateTiledIterations(operation, state.loopOrder, state.tileSize);
    }
    return generateIterations(operation, state.loopOrder);
}

// =============================================================================
// CACHE MODEL
// =============================================================================

/**
 * LRU Cache Simulator
 */
class CacheSimulator {
    constructor(capacityBytes, lineSize) {
        this.capacityBytes = capacityBytes;
        this.lineSize = lineSize;
        this.maxLines = Math.floor(capacityBytes / lineSize);
        this.lines = [];
        this.totalAccesses = 0;
        this.hits = 0;
        this.misses = 0;
    }

    getLineAddress(address) {
        return Math.floor(address / this.lineSize) * this.lineSize;
    }

    access(address) {
        this.totalAccesses++;
        const lineAddr = this.getLineAddress(address);
        const index = this.lines.indexOf(lineAddr);

        if (index !== -1) {
            this.lines.splice(index, 1);
            this.lines.push(lineAddr);
            this.hits++;
            return true;
        } else {
            this.misses++;
            if (this.lines.length >= this.maxLines) {
                this.lines.shift();
            }
            this.lines.push(lineAddr);
            return false;
        }
    }

    isAddressCached(address) {
        const lineAddr = this.getLineAddress(address);
        return this.lines.includes(lineAddr);
    }

    reset() {
        this.lines = [];
        this.totalAccesses = 0;
        this.hits = 0;
        this.misses = 0;
    }

    snapshot() {
        return {
            lines: [...this.lines],
            totalAccesses: this.totalAccesses,
            hits: this.hits,
            misses: this.misses
        };
    }

    restore(snap) {
        this.lines = [...snap.lines];
        this.totalAccesses = snap.totalAccesses;
        this.hits = snap.hits;
        this.misses = snap.misses;
    }
}

// =============================================================================
// MEMORY ADDRESS CALCULATION
// =============================================================================

/**
 * Calculate memory address for a tensor element.
 *
 * @param {Object} tensor - Tensor definition from operation
 * @param {number} row - Row index
 * @param {number} col - Column index
 * @param {string} layout - 'row' or 'col'
 * @returns {number} Memory address in bytes
 */
function getTensorAddress(tensor, row, col, layout) {
    const size = operation.size;
    if (layout === 'col') {
        return tensor.baseAddress + (col * size + row) * operation.elementSize;
    }
    return tensor.baseAddress + (row * size + col) * operation.elementSize;
}

/**
 * Get memory address for a tensor access given current iteration.
 */
function getAccessAddress(tensor, iter) {
    const indices = tensor.getIndices(iter);
    const layout = state.layouts[tensor.name];
    return getTensorAddress(tensor, indices.row, indices.col, layout);
}

/**
 * Check if a tensor element is in cache.
 */
function isElementInCache(tensorName, row, col) {
    if (!state.cache) return false;

    const tensor = operation.tensors.find(t => t.name === tensorName);
    if (!tensor) return false;

    const layout = state.layouts[tensorName];
    const address = getTensorAddress(tensor, row, col, layout);
    return state.cache.isAddressCached(address);
}

// =============================================================================
// RENDERING
// =============================================================================

/**
 * Initialize canvas contexts for all tensors.
 */
function initCanvases() {
    for (const tensor of operation.tensors) {
        const canvas = document.getElementById('matrix' + tensor.name);
        if (canvas) {
            canvasContexts[tensor.name] = canvas.getContext('2d');
        }
    }

    ctxTimeline = document.getElementById('timeline').getContext('2d');
    const timelineCanvas = document.getElementById('timeline');
    timelineCanvas.width = timelineCanvas.offsetWidth;
    timelineCanvas.height = 60;
}

/**
 * Render a single tensor grid.
 */
function renderTensor(tensorName, currentRow, currentCol) {
    const ctx = canvasContexts[tensorName];
    if (!ctx) return;

    const canvas = ctx.canvas;
    const size = operation.size;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Cached elements
    if (state.cache) {
        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                if (isElementInCache(tensorName, row, col)) {
                    ctx.fillStyle = COLORS.cached;
                    ctx.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
            }
        }
    }

    // Grid lines
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= size; i++) {
        ctx.beginPath();
        ctx.moveTo(i * CELL_SIZE, 0);
        ctx.lineTo(i * CELL_SIZE, canvas.height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i * CELL_SIZE);
        ctx.lineTo(canvas.width, i * CELL_SIZE);
        ctx.stroke();
    }

    // Tile boundaries
    if (state.tilingEnabled && state.tileSize > 1) {
        ctx.strokeStyle = COLORS.tileGrid;
        ctx.lineWidth = 2;
        for (let i = 0; i <= size; i += state.tileSize) {
            ctx.beginPath();
            ctx.moveTo(i * CELL_SIZE, 0);
            ctx.lineTo(i * CELL_SIZE, canvas.height);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, i * CELL_SIZE);
            ctx.lineTo(canvas.width, i * CELL_SIZE);
            ctx.stroke();
        }
    }

    // Current access indicator
    if (currentRow !== null && currentCol !== null) {
        ctx.fillStyle = COLORS.current;
        ctx.fillRect(currentCol * CELL_SIZE + 3, currentRow * CELL_SIZE + 3, CELL_SIZE - 6, CELL_SIZE - 6);
        ctx.strokeStyle = COLORS.currentOutline;
        ctx.lineWidth = 2;
        ctx.strokeRect(currentCol * CELL_SIZE + 1, currentRow * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2);
    }
}

/**
 * Render all tensors.
 */
function renderAllTensors() {
    const iter = state.iterations[state.currentIteration];

    for (const tensor of operation.tensors) {
        if (iter) {
            const indices = tensor.getIndices(iter);
            renderTensor(tensor.name, indices.row, indices.col);
        } else {
            renderTensor(tensor.name, null, null);
        }
    }
}

/**
 * Render the cache hit timeline.
 */
function renderTimeline() {
    const canvas = ctxTimeline.canvas;
    const width = canvas.width;
    const height = canvas.height;
    const numTensors = operation.tensors.length;

    ctxTimeline.fillStyle = '#1a1a2e';
    ctxTimeline.fillRect(0, 0, width, height);

    if (state.history.length === 0) return;

    const rowHeight = height / numTensors;
    const labelOffset = 15;
    const barWidthScaled = (width - labelOffset) / TOTAL_ITERATIONS;

    // Labels
    ctxTimeline.fillStyle = '#666';
    ctxTimeline.font = '10px monospace';
    operation.tensors.forEach((tensor, idx) => {
        ctxTimeline.fillText(tensor.name, 2, idx * rowHeight + rowHeight / 2 + 3);
    });

    // History bars
    for (let i = 0; i < state.history.length; i++) {
        const x = labelOffset + i * barWidthScaled;
        const h = state.history[i];
        const barW = Math.max(1, barWidthScaled - 0.5);

        operation.tensors.forEach((tensor, idx) => {
            ctxTimeline.fillStyle = h[tensor.name] ? '#28a745' : '#dc3545';
            ctxTimeline.fillRect(x, idx * rowHeight + 2, barW, rowHeight - 4);
        });
    }

    // Current position indicator
    if (state.currentIteration > 0) {
        const x = labelOffset + (state.currentIteration - 1) * barWidthScaled;
        ctxTimeline.strokeStyle = '#667eea';
        ctxTimeline.lineWidth = 2;
        ctxTimeline.beginPath();
        ctxTimeline.moveTo(x + barWidthScaled, 0);
        ctxTimeline.lineTo(x + barWidthScaled, height);
        ctxTimeline.stroke();
    }
}

/**
 * Update statistics display.
 */
function updateStatsDisplay() {
    // Per-tensor stats in header
    for (const tensor of operation.tensors) {
        const statsEl = document.getElementById('stats' + tensor.name);
        if (statsEl) {
            const s = state.stats[tensor.name];
            statsEl.textContent = `mem:${s.accesses} hit:${s.hits}`;
        }
    }

    // Overall stats
    let totalAccesses = 0;
    let totalHits = 0;
    for (const tensor of operation.tensors) {
        totalAccesses += state.stats[tensor.name].accesses;
        totalHits += state.stats[tensor.name].hits;
    }
    const hitRate = totalAccesses > 0 ? (totalHits / totalAccesses * 100).toFixed(0) : 0;

    document.getElementById('totalMem').textContent = totalAccesses;
    document.getElementById('totalHits').textContent = totalHits;
    document.getElementById('hitRate').textContent = hitRate + '%';

    // Detailed per-tensor stats
    for (const tensor of operation.tensors) {
        const detailEl = document.getElementById('detailStats' + tensor.name);
        if (detailEl) {
            const s = state.stats[tensor.name];
            detailEl.textContent = `${s.hits}/${s.accesses} hits`;
        }
    }
}

/**
 * Update current state display.
 */
function updateStateDisplay() {
    const iter = state.iterations[state.currentIteration];

    document.getElementById('currentIteration').textContent = `${state.currentIteration} of ${TOTAL_ITERATIONS}`;

    if (iter) {
        // Build indices string
        let indicesStr = '';
        if (state.tilingEnabled && iter.ti !== undefined) {
            // Show tile indices first, then element indices
            const tileParts = operation.loopDims.map(d => `t${d}=${iter['t' + d]}`);
            const elemParts = operation.loopDims.map(d => `${d}=${iter[d]}`);
            indicesStr = tileParts.join(', ') + ', ' + elemParts.join(', ');
        } else {
            indicesStr = operation.loopDims.map(d => `${d}=${iter[d]}`).join(', ');
        }
        document.getElementById('currentIndices').textContent = indicesStr;
        document.getElementById('currentOp').textContent = operation.describeOp(iter);
    } else {
        document.getElementById('currentIndices').textContent = '-';
        document.getElementById('currentOp').textContent = '-';
    }

    document.getElementById('jumpIteration').max = TOTAL_ITERATIONS - 1;
    document.getElementById('totalIterations').textContent = TOTAL_ITERATIONS;
}

/**
 * Full render of all visualization components.
 */
function render() {
    renderAllTensors();
    renderTimeline();
    updateStatsDisplay();
    updateStateDisplay();
}

// =============================================================================
// SIMULATION LOGIC
// =============================================================================

/**
 * Execute one simulation step.
 * Uses the operation abstraction to determine memory accesses.
 */
function executeStep() {
    if (state.currentIteration >= state.iterations.length) {
        return null;
    }

    const iter = state.iterations[state.currentIteration];
    const result = {};

    // Access each tensor according to the operation's access pattern
    for (const tensor of operation.tensors) {
        const address = getAccessAddress(tensor, iter);
        const hit = state.cache.access(address);

        state.stats[tensor.name].accesses++;
        state.stats[tensor.name].hits += hit ? 1 : 0;
        result[tensor.name] = hit;
    }

    state.history.push(result);
    state.currentIteration++;

    return result;
}

/**
 * Reset simulation to initial state.
 */
function resetSimulation() {
    stopAnimation();

    state.currentIteration = 0;
    state.history = [];
    snapshots = [];

    // Reset stats for all tensors
    for (const tensor of operation.tensors) {
        state.stats[tensor.name] = { accesses: 0, hits: 0 };
    }

    if (state.cache) {
        state.cache.reset();
    }

    document.getElementById('playPauseBtn').textContent = '▶';
    render();
    updateCodeDisplay();
}

/**
 * Jump to a specific iteration.
 */
function jumpToIteration(targetIteration) {
    targetIteration = Math.max(0, Math.min(targetIteration, state.iterations.length));

    if (targetIteration < state.currentIteration) {
        resetSimulation();
    }

    while (state.currentIteration < targetIteration) {
        executeStep();
    }

    render();
    updateCodeDisplay();
}

/**
 * Step forward one iteration.
 */
function stepForward() {
    if (state.currentIteration < state.iterations.length) {
        snapshots.push({
            cache: state.cache.snapshot(),
            stats: JSON.parse(JSON.stringify(state.stats)),
            historyLength: state.history.length,
            iteration: state.currentIteration
        });

        executeStep();
        render();
        updateCodeDisplay();
    }
}

/**
 * Step backward one iteration.
 */
function stepBackward() {
    if (snapshots.length > 0) {
        const snapshot = snapshots.pop();
        state.cache.restore(snapshot.cache);
        state.stats = snapshot.stats;
        state.history = state.history.slice(0, snapshot.historyLength);
        state.currentIteration = snapshot.iteration;

        render();
        updateCodeDisplay();
    }
}

/**
 * Animation loop.
 */
function animate(timestamp) {
    if (!state.isPlaying) return;

    const elapsed = timestamp - lastFrameTime;
    const interval = 1000 / state.speed;

    if (elapsed >= interval) {
        if (state.currentIteration < state.iterations.length) {
            if (snapshots.length > 100) {
                snapshots = snapshots.slice(-50);
            }

            snapshots.push({
                cache: state.cache.snapshot(),
                stats: JSON.parse(JSON.stringify(state.stats)),
                historyLength: state.history.length,
                iteration: state.currentIteration
            });

            executeStep();
            render();
            updateCodeDisplay();
            lastFrameTime = timestamp;
        } else {
            stopAnimation();
        }
    }

    if (state.isPlaying) {
        animationId = requestAnimationFrame(animate);
    }
}

function startAnimation() {
    if (!state.isPlaying && state.currentIteration < state.iterations.length) {
        state.isPlaying = true;
        document.getElementById('playPauseBtn').textContent = '⏸';
        lastFrameTime = performance.now();
        animationId = requestAnimationFrame(animate);
    }
}

function stopAnimation() {
    state.isPlaying = false;
    document.getElementById('playPauseBtn').textContent = '▶';
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
}

function togglePlayPause() {
    if (state.isPlaying) {
        stopAnimation();
    } else {
        startAnimation();
    }
}

/**
 * Apply configuration from UI controls.
 *
 * Note on cache line count: With 3-tensor operations (A, B, C), fewer than 4 cache
 * lines causes severe thrashing. Each iteration accesses all 3 tensors, so with
 * N < 4 lines, at least one tensor's line is evicted before its next access.
 * This makes spatial locality benefits impossible to observe. UI restricts to >= 4 lines.
 */
function applyConfiguration() {
    state.loopOrder = document.getElementById('loopOrder').value;
    state.tilingEnabled = document.getElementById('tilingEnabled').checked;
    state.tileSize = parseInt(document.getElementById('tileSize').value);

    // Cache configuration (minimum 4 lines to avoid thrashing with 3 tensors)
    state.elementsPerLine = parseInt(document.getElementById('elementsPerLine').value);
    state.numCacheLines = parseInt(document.getElementById('numCacheLines').value);

    // Read layouts for all tensors
    for (const tensor of operation.tensors) {
        const layoutEl = document.getElementById('layout' + tensor.name);
        if (layoutEl) {
            state.layouts[tensor.name] = layoutEl.value;
        }
    }

    // Compute cache parameters
    const cacheLineSize = state.elementsPerLine * ELEMENT_SIZE;
    const cacheCapacity = cacheLineSize * state.numCacheLines;

    state.iterations = generateAllIterations();
    state.cache = new CacheSimulator(cacheCapacity, cacheLineSize);

    resetSimulation();
}

/**
 * Update the byte equivalent displays for cache configuration.
 */
function updateCacheDisplays() {
    const elementsPerLine = parseInt(document.getElementById('elementsPerLine').value);
    const numCacheLines = parseInt(document.getElementById('numCacheLines').value);

    const lineBytes = elementsPerLine * ELEMENT_SIZE;
    const totalBytes = lineBytes * numCacheLines;

    document.getElementById('lineSizeBytes').textContent = `= ${lineBytes}B`;
    document.getElementById('cacheSizeBytes').textContent = `= ${totalBytes}B`;
}

// =============================================================================
// CODE DISPLAY
// =============================================================================

/**
 * Get the loop order array for the current configuration.
 */
function getLoopOrderArray() {
    return operation.loopOrders[state.loopOrder] || operation.loopDims;
}

/**
 * Update the loop code display.
 */
function updateCodeDisplay() {
    const codeDiv = document.getElementById('codeDisplay');
    const order = getLoopOrderArray();

    if (state.tilingEnabled) {
        codeDiv.innerHTML = generateTiledCodeHTML(order, state.tileSize);
    } else {
        codeDiv.innerHTML = generateNonTiledCodeHTML(order);
    }
}

/**
 * Generate HTML for non-tiled loop code.
 */
function generateNonTiledCodeHTML(order) {
    const iter = state.iterations[state.currentIteration];
    const size = operation.size;
    let html = '';
    const indent = ['', '  ', '    ', '      '];
    const numLoops = order.length;

    for (let level = 0; level < numLoops; level++) {
        const varName = order[level];
        const isInnermost = level === numLoops - 1;
        const isCurrent = isInnermost && state.currentIteration < state.iterations.length;

        html += `<div class="code-line${isCurrent ? ' current' : ''}">`;
        html += `${indent[level]}<span class="code-keyword">for</span> `;
        html += `<span class="code-var">${varName}</span> `;
        html += `<span class="code-keyword">in</span> `;
        html += `<span class="code-number">0</span>..<span class="code-number">${size}</span>:`;
        if (isCurrent && iter) {
            html += ` <span class="code-comment">← ${varName}=${iter[varName]}</span>`;
        }
        html += '</div>';
    }

    html += `<div class="code-line">${indent[numLoops]}${operation.codeTemplate}</div>`;
    return html;
}

/**
 * Generate HTML for tiled loop code.
 */
function generateTiledCodeHTML(order, tileSize) {
    const iter = state.iterations[state.currentIteration];
    const size = operation.size;
    const indent = ['', '  ', '    ', '      ', '        ', '          ', '            '];
    const numLoops = order.length;
    let html = '';

    // Tile loops
    for (let level = 0; level < numLoops; level++) {
        const varName = 't' + order[level];
        html += `<div class="code-line">`;
        html += `${indent[level]}<span class="code-keyword">for</span> `;
        html += `<span class="code-var">${varName}</span> `;
        html += `<span class="code-keyword">in</span> `;
        html += `<span class="code-number">0</span>..<span class="code-number">${size}</span> `;
        html += `<span class="code-keyword">step</span> <span class="code-number">${tileSize}</span>:`;
        if (iter && iter[varName] !== undefined) {
            html += ` <span class="code-comment">← ${varName}=${iter[varName]}</span>`;
        }
        html += '</div>';
    }

    // Inner element loops
    for (let level = 0; level < numLoops; level++) {
        const varName = order[level];
        const tileVar = 't' + varName;
        const isInnermost = level === numLoops - 1;
        const isCurrent = isInnermost && state.currentIteration < state.iterations.length;

        html += `<div class="code-line${isCurrent ? ' current' : ''}">`;
        html += `${indent[level + numLoops]}<span class="code-keyword">for</span> `;
        html += `<span class="code-var">${varName}</span> `;
        html += `<span class="code-keyword">in</span> `;
        html += `<span class="code-var">${tileVar}</span>..<span class="code-var">${tileVar}</span>+<span class="code-number">${tileSize}</span>:`;
        if (isCurrent && iter) {
            html += ` <span class="code-comment">← ${varName}=${iter[varName]}</span>`;
        }
        html += '</div>';
    }

    html += `<div class="code-line">${indent[2 * numLoops]}${operation.codeTemplate}</div>`;
    return html;
}

// =============================================================================
// UI EVENT HANDLERS
// =============================================================================

function setupEventHandlers() {
    // Tiling toggle
    document.getElementById('tilingEnabled').addEventListener('change', (e) => {
        document.getElementById('tileSize').disabled = !e.target.checked;
    });

    // Cache configuration - update byte displays on change
    document.getElementById('elementsPerLine').addEventListener('change', updateCacheDisplays);
    document.getElementById('numCacheLines').addEventListener('change', updateCacheDisplays);

    // Configuration
    document.getElementById('applyConfig').addEventListener('click', applyConfiguration);

    // Playback
    document.getElementById('playPauseBtn').addEventListener('click', togglePlayPause);
    document.getElementById('stepFwdBtn').addEventListener('click', stepForward);
    document.getElementById('stepBackBtn').addEventListener('click', stepBackward);
    document.getElementById('resetBtn').addEventListener('click', resetSimulation);

    // Speed slider
    document.getElementById('speedSlider').addEventListener('input', (e) => {
        const speed = Math.round(Math.pow(10, e.target.value / 50));
        state.speed = speed;
        document.getElementById('speedDisplay').textContent = speed + 'x';
    });

    // Jump to iteration
    document.getElementById('jumpIteration').addEventListener('change', (e) => {
        const target = Math.min(Math.max(0, parseInt(e.target.value) || 0), TOTAL_ITERATIONS - 1);
        jumpToIteration(target);
        e.target.value = state.currentIteration;
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        switch (e.code) {
            case 'Space':
                e.preventDefault();
                togglePlayPause();
                break;
            case 'ArrowRight':
                e.preventDefault();
                stepForward();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                stepBackward();
                break;
            case 'KeyR':
                e.preventDefault();
                resetSimulation();
                break;
        }
    });

    // Window resize
    window.addEventListener('resize', () => {
        const timelineCanvas = document.getElementById('timeline');
        timelineCanvas.width = timelineCanvas.offsetWidth;
        renderTimeline();
    });
}

// =============================================================================
// INITIALIZATION
// =============================================================================

function init() {
    // Skip initialization if essential DOM elements are missing (e.g., in test environment)
    if (!document.getElementById('matrixA')) {
        console.log('Skipping UI initialization (test mode)');
        return;
    }

    console.log('Cache Visualizer initialized');
    console.log(`Operation: ${operation.displayName}`);
    console.log(`Tensors: ${operation.tensors.map(t => t.name).join(', ')}`);

    initCanvases();
    setupEventHandlers();
    applyConfiguration();
}

// Browser: initialize on DOM ready
if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', init);
}

// Node.js: export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        createMatmulOperation,
        generateIterations,
        generateTiledIterations,
        CacheSimulator
    };
}
