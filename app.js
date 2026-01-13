/**
 * Matrix Tiling and Cache Behavior Visualizer
 *
 * An educational tool for visualizing how iteration orders, tiling strategies,
 * and data layouts affect cache behavior during matrix multiplication.
 *
 * @author Claude (with user guidance)
 */

// =============================================================================
// CONFIGURATION & CONSTANTS
// =============================================================================

const MATRIX_SIZE = 12;
const ELEMENT_SIZE = 4; // bytes per element (float/int32)
const CACHE_LINE_SIZE = 64; // bytes per cache line
const TOTAL_ITERATIONS = MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE; // 1728

// Base addresses for matrices in simulated memory (non-overlapping regions)
const BASE_A = 0;
const BASE_B = MATRIX_SIZE * MATRIX_SIZE * ELEMENT_SIZE; // 576
const BASE_C = 2 * MATRIX_SIZE * MATRIX_SIZE * ELEMENT_SIZE; // 1152

// Rendering constants
const CELL_SIZE = 20; // pixels per cell (240px canvas / 12 cells)
const COLORS = {
    background: '#ffffff',
    grid: '#cccccc',
    tileGrid: '#666666',
    cached: 'rgba(220, 53, 69, 0.6)',
    cachedLight: 'rgba(220, 53, 69, 0.3)',
    current: '#000000',
    currentOutline: '#667eea'
};

// =============================================================================
// APPLICATION STATE
// =============================================================================

const state = {
    // Configuration (from UI)
    loopOrder: 'ijk',
    tilingEnabled: false,
    tileSize: 4,
    layoutA: 'row',
    layoutB: 'row',
    layoutC: 'row',
    cacheSize: 256,

    // Simulation state
    currentIteration: 0,
    isPlaying: false,
    speed: 10, // iterations per second

    // Generated data
    iterations: [],
    cache: null,

    // Statistics
    stats: {
        A: { accesses: 0, hits: 0 },
        B: { accesses: 0, hits: 0 },
        C: { accesses: 0, hits: 0 }
    },

    // History for timeline visualization
    history: []
};

// Canvas contexts (initialized on load)
let ctxA, ctxB, ctxC, ctxTimeline;

// Animation state
let animationId = null;
let lastFrameTime = 0;
let snapshots = []; // For step-backward functionality

// =============================================================================
// ITERATION GENERATOR
// =============================================================================

/**
 * Generates the sequence of (i, j, k) index tuples for non-tiled matrix multiplication.
 *
 * The loop order determines which index varies fastest (innermost loop).
 * For example, 'ijk' means: for i { for j { for k { ... } } }
 *
 * @param {string} loopOrder - One of: 'ijk', 'ikj', 'jik', 'jki', 'kij', 'kji'
 * @param {number} size - Matrix dimension
 * @returns {Array<{i: number, j: number, k: number}>}
 */
function generateIterations(loopOrder, size) {
    const iterations = [];
    const loopOrders = {
        'ijk': ['i', 'j', 'k'],
        'ikj': ['i', 'k', 'j'],
        'jik': ['j', 'i', 'k'],
        'jki': ['j', 'k', 'i'],
        'kij': ['k', 'i', 'j'],
        'kji': ['k', 'j', 'i']
    };

    const order = loopOrders[loopOrder];
    if (!order) {
        throw new Error(`Unknown loop order: ${loopOrder}`);
    }

    for (let outer = 0; outer < size; outer++) {
        for (let middle = 0; middle < size; middle++) {
            for (let inner = 0; inner < size; inner++) {
                const indices = {};
                indices[order[0]] = outer;
                indices[order[1]] = middle;
                indices[order[2]] = inner;
                iterations.push({ i: indices.i, j: indices.j, k: indices.k });
            }
        }
    }

    return iterations;
}

/**
 * Returns the loop variable order for display purposes.
 * @param {string} loopOrder
 * @returns {Array<string>}
 */
function getLoopOrderArray(loopOrder) {
    const orders = {
        'ijk': ['i', 'j', 'k'],
        'ikj': ['i', 'k', 'j'],
        'jik': ['j', 'i', 'k'],
        'jki': ['j', 'k', 'i'],
        'kij': ['k', 'i', 'j'],
        'kji': ['k', 'j', 'i']
    };
    return orders[loopOrder] || ['i', 'j', 'k'];
}

/**
 * Generates index tuples for TILED matrix multiplication.
 *
 * Tiled iteration uses 6 nested loops:
 * - 3 outer tile loops (ti, tj, tk) stepping by tileSize
 * - 3 inner loops (i, j, k) within each tile
 *
 * @param {string} loopOrder - Loop nesting order
 * @param {number} size - Matrix dimension
 * @param {number} tileSize - Tile dimension
 * @returns {Array<{i, j, k, ti, tj, tk, li, lj, lk}>}
 */
function generateTiledIterations(loopOrder, size, tileSize) {
    const iterations = [];
    const orderMap = {
        'ijk': ['i', 'j', 'k'],
        'ikj': ['i', 'k', 'j'],
        'jik': ['j', 'i', 'k'],
        'jki': ['j', 'k', 'i'],
        'kij': ['k', 'i', 'j'],
        'kji': ['k', 'j', 'i']
    };

    const order = orderMap[loopOrder];
    if (!order) {
        throw new Error(`Unknown loop order: ${loopOrder}`);
    }

    const numTiles = size / tileSize;

    // Outer tile loops
    for (let tOuter = 0; tOuter < numTiles; tOuter++) {
        for (let tMiddle = 0; tMiddle < numTiles; tMiddle++) {
            for (let tInner = 0; tInner < numTiles; tInner++) {
                const tileIndices = {};
                tileIndices['t' + order[0]] = tOuter * tileSize;
                tileIndices['t' + order[1]] = tMiddle * tileSize;
                tileIndices['t' + order[2]] = tInner * tileSize;

                const ti = tileIndices.ti;
                const tj = tileIndices.tj;
                const tk = tileIndices.tk;

                // Inner element loops
                for (let eOuter = 0; eOuter < tileSize; eOuter++) {
                    for (let eMiddle = 0; eMiddle < tileSize; eMiddle++) {
                        for (let eInner = 0; eInner < tileSize; eInner++) {
                            const elemIndices = {};
                            elemIndices[order[0]] = eOuter;
                            elemIndices[order[1]] = eMiddle;
                            elemIndices[order[2]] = eInner;

                            iterations.push({
                                i: ti + elemIndices.i,
                                j: tj + elemIndices.j,
                                k: tk + elemIndices.k,
                                ti, tj, tk,
                                li: elemIndices.i,
                                lj: elemIndices.j,
                                lk: elemIndices.k
                            });
                        }
                    }
                }
            }
        }
    }

    return iterations;
}

/**
 * Generates iterations based on current configuration.
 */
function generateAllIterations() {
    if (state.tilingEnabled) {
        return generateTiledIterations(state.loopOrder, MATRIX_SIZE, state.tileSize);
    }
    return generateIterations(state.loopOrder, MATRIX_SIZE);
}

// =============================================================================
// CACHE MODEL
// =============================================================================

/**
 * LRU Cache Simulator
 *
 * Simulates a simple cache with:
 * - Configurable capacity
 * - Fixed cache line size (64 bytes = 16 elements)
 * - LRU (Least Recently Used) eviction policy
 */
class CacheSimulator {
    constructor(capacityBytes, lineSize = CACHE_LINE_SIZE) {
        this.capacityBytes = capacityBytes;
        this.lineSize = lineSize;
        this.maxLines = Math.floor(capacityBytes / lineSize);
        this.lines = []; // LRU order: front = oldest, back = newest
        this.totalAccesses = 0;
        this.hits = 0;
        this.misses = 0;
    }

    /**
     * Get cache line base address for a memory address.
     */
    getLineAddress(address) {
        return Math.floor(address / this.lineSize) * this.lineSize;
    }

    /**
     * Access a memory address. Returns true for hit, false for miss.
     */
    access(address) {
        this.totalAccesses++;
        const lineAddr = this.getLineAddress(address);
        const index = this.lines.indexOf(lineAddr);

        if (index !== -1) {
            // Hit: move to MRU position
            this.lines.splice(index, 1);
            this.lines.push(lineAddr);
            this.hits++;
            return true;
        } else {
            // Miss: evict LRU if full, add new line
            this.misses++;
            if (this.lines.length >= this.maxLines) {
                this.lines.shift();
            }
            this.lines.push(lineAddr);
            return false;
        }
    }

    /**
     * Check if address is cached (without affecting LRU order).
     */
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

    restore(snapshot) {
        this.lines = [...snapshot.lines];
        this.totalAccesses = snapshot.totalAccesses;
        this.hits = snapshot.hits;
        this.misses = snapshot.misses;
    }
}

// =============================================================================
// MEMORY ADDRESS CALCULATION
// =============================================================================

/**
 * Calculate memory address for a matrix element.
 *
 * @param {number} row - Row index
 * @param {number} col - Column index
 * @param {number} baseAddress - Matrix base address
 * @param {string} layout - 'row' for row-major, 'col' for column-major
 */
function getAddress(row, col, baseAddress, layout) {
    if (layout === 'col') {
        // Column-major: elements in same column are contiguous
        return baseAddress + (col * MATRIX_SIZE + row) * ELEMENT_SIZE;
    }
    // Row-major: elements in same row are contiguous
    return baseAddress + (row * MATRIX_SIZE + col) * ELEMENT_SIZE;
}

function getAddressA(i, k) {
    return getAddress(i, k, BASE_A, state.layoutA);
}

function getAddressB(k, j) {
    return getAddress(k, j, BASE_B, state.layoutB);
}

function getAddressC(i, j) {
    return getAddress(i, j, BASE_C, state.layoutC);
}

/**
 * Check if a matrix element is currently in cache.
 */
function isElementInCache(matrix, row, col) {
    if (!state.cache) return false;

    let address;
    switch (matrix) {
        case 'A':
            address = getAddress(row, col, BASE_A, state.layoutA);
            break;
        case 'B':
            address = getAddress(row, col, BASE_B, state.layoutB);
            break;
        case 'C':
            address = getAddress(row, col, BASE_C, state.layoutC);
            break;
        default:
            return false;
    }
    return state.cache.isAddressCached(address);
}

// =============================================================================
// RENDERING
// =============================================================================

/**
 * Initialize canvas contexts.
 */
function initCanvases() {
    ctxA = document.getElementById('matrixA').getContext('2d');
    ctxB = document.getElementById('matrixB').getContext('2d');
    ctxC = document.getElementById('matrixC').getContext('2d');
    ctxTimeline = document.getElementById('timeline').getContext('2d');

    const timelineCanvas = document.getElementById('timeline');
    timelineCanvas.width = timelineCanvas.offsetWidth;
    timelineCanvas.height = 60;
}

/**
 * Render a single matrix grid.
 */
function renderMatrix(ctx, matrixName, currentRow, currentCol) {
    const canvas = ctx.canvas;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Cached elements
    if (state.cache) {
        for (let row = 0; row < MATRIX_SIZE; row++) {
            for (let col = 0; col < MATRIX_SIZE; col++) {
                if (isElementInCache(matrixName, row, col)) {
                    ctx.fillStyle = COLORS.cached;
                    ctx.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
            }
        }
    }

    // Grid lines
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= MATRIX_SIZE; i++) {
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
        for (let i = 0; i <= MATRIX_SIZE; i += state.tileSize) {
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
 * Render all three matrices.
 */
function renderAllMatrices() {
    const iter = state.iterations[state.currentIteration];
    if (iter) {
        renderMatrix(ctxA, 'A', iter.i, iter.k);
        renderMatrix(ctxB, 'B', iter.k, iter.j);
        renderMatrix(ctxC, 'C', iter.i, iter.j);
    } else {
        renderMatrix(ctxA, 'A', null, null);
        renderMatrix(ctxB, 'B', null, null);
        renderMatrix(ctxC, 'C', null, null);
    }
}

/**
 * Render the cache hit timeline.
 */
function renderTimeline() {
    const canvas = ctxTimeline.canvas;
    const width = canvas.width;
    const height = canvas.height;

    ctxTimeline.fillStyle = '#1a1a2e';
    ctxTimeline.fillRect(0, 0, width, height);

    if (state.history.length === 0) return;

    const rowHeight = height / 3;
    const labelOffset = 15;
    const barWidthScaled = (width - labelOffset) / TOTAL_ITERATIONS;

    // Labels
    ctxTimeline.fillStyle = '#666';
    ctxTimeline.font = '10px monospace';
    ctxTimeline.fillText('A', 2, rowHeight / 2 + 3);
    ctxTimeline.fillText('B', 2, rowHeight + rowHeight / 2 + 3);
    ctxTimeline.fillText('C', 2, 2 * rowHeight + rowHeight / 2 + 3);

    // History bars
    for (let i = 0; i < state.history.length; i++) {
        const x = labelOffset + i * barWidthScaled;
        const h = state.history[i];
        const barW = Math.max(1, barWidthScaled - 0.5);

        ctxTimeline.fillStyle = h.A ? '#dc3545' : '#333';
        ctxTimeline.fillRect(x, 2, barW, rowHeight - 4);

        ctxTimeline.fillStyle = h.B ? '#dc3545' : '#333';
        ctxTimeline.fillRect(x, rowHeight + 2, barW, rowHeight - 4);

        ctxTimeline.fillStyle = h.C ? '#dc3545' : '#333';
        ctxTimeline.fillRect(x, 2 * rowHeight + 2, barW, rowHeight - 4);
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
    document.getElementById('statsA').textContent = `mem:${state.stats.A.accesses} hit:${state.stats.A.hits}`;
    document.getElementById('statsB').textContent = `mem:${state.stats.B.accesses} hit:${state.stats.B.hits}`;
    document.getElementById('statsC').textContent = `mem:${state.stats.C.accesses} hit:${state.stats.C.hits}`;

    const totalAccesses = state.stats.A.accesses + state.stats.B.accesses + state.stats.C.accesses;
    const totalHits = state.stats.A.hits + state.stats.B.hits + state.stats.C.hits;
    const hitRate = totalAccesses > 0 ? (totalHits / totalAccesses * 100).toFixed(0) : 0;

    document.getElementById('totalMem').textContent = totalAccesses;
    document.getElementById('totalHits').textContent = totalHits;
    document.getElementById('hitRate').textContent = hitRate + '%';

    document.getElementById('detailStatsA').textContent = `${state.stats.A.hits}/${state.stats.A.accesses} hits`;
    document.getElementById('detailStatsB').textContent = `${state.stats.B.hits}/${state.stats.B.accesses} hits`;
    document.getElementById('detailStatsC').textContent = `${state.stats.C.hits}/${state.stats.C.accesses} hits`;
}

/**
 * Update current state display.
 */
function updateStateDisplay() {
    const iter = state.iterations[state.currentIteration];

    document.getElementById('currentIteration').textContent = `${state.currentIteration} of ${TOTAL_ITERATIONS}`;

    if (iter) {
        if (state.tilingEnabled && iter.ti !== undefined) {
            document.getElementById('currentIndices').textContent =
                `ti=${iter.ti}, tj=${iter.tj}, tk=${iter.tk}, i=${iter.i}, j=${iter.j}, k=${iter.k}`;
        } else {
            document.getElementById('currentIndices').textContent = `i=${iter.i}, j=${iter.j}, k=${iter.k}`;
        }
        document.getElementById('currentOp').textContent =
            `A[${iter.i}][${iter.k}] × B[${iter.k}][${iter.j}] → C[${iter.i}][${iter.j}]`;
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
    renderAllMatrices();
    renderTimeline();
    updateStatsDisplay();
    updateStateDisplay();
}

// =============================================================================
// SIMULATION LOGIC
// =============================================================================

/**
 * Execute one simulation step.
 */
function executeStep() {
    if (state.currentIteration >= state.iterations.length) {
        return null;
    }

    const iter = state.iterations[state.currentIteration];

    const addrA = getAddressA(iter.i, iter.k);
    const addrB = getAddressB(iter.k, iter.j);
    const addrC = getAddressC(iter.i, iter.j);

    const hitA = state.cache.access(addrA);
    const hitB = state.cache.access(addrB);
    const hitC = state.cache.access(addrC);

    state.stats.A.accesses++;
    state.stats.A.hits += hitA ? 1 : 0;
    state.stats.B.accesses++;
    state.stats.B.hits += hitB ? 1 : 0;
    state.stats.C.accesses++;
    state.stats.C.hits += hitC ? 1 : 0;

    const result = { A: hitA, B: hitB, C: hitC };
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
    state.stats = {
        A: { accesses: 0, hits: 0 },
        B: { accesses: 0, hits: 0 },
        C: { accesses: 0, hits: 0 }
    };

    if (state.cache) {
        state.cache.reset();
    }

    document.getElementById('playPauseBtn').textContent = '▶';
    render();
    updateCodeDisplay();
}

/**
 * Jump to a specific iteration (requires replay from start if going backward).
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
            // Limit snapshot history to prevent memory issues
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
 */
function applyConfiguration() {
    state.loopOrder = document.getElementById('loopOrder').value;
    state.tilingEnabled = document.getElementById('tilingEnabled').checked;
    state.tileSize = parseInt(document.getElementById('tileSize').value);
    state.layoutA = document.getElementById('layoutA').value;
    state.layoutB = document.getElementById('layoutB').value;
    state.layoutC = document.getElementById('layoutC').value;
    state.cacheSize = parseInt(document.getElementById('cacheSize').value) || 256;

    state.iterations = generateAllIterations();
    state.cache = new CacheSimulator(state.cacheSize);

    resetSimulation();
}

// =============================================================================
// CODE DISPLAY
// =============================================================================

/**
 * Update the loop code display.
 */
function updateCodeDisplay() {
    const codeDiv = document.getElementById('codeDisplay');
    const order = getLoopOrderArray(state.loopOrder);

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
    let html = '';
    const indent = ['', '  ', '    ', '      '];

    for (let level = 0; level < 3; level++) {
        const varName = order[level];
        const isInnermost = level === 2;
        const isCurrent = isInnermost && state.currentIteration < state.iterations.length;

        html += `<div class="code-line${isCurrent ? ' current' : ''}">`;
        html += `${indent[level]}<span class="code-keyword">for</span> `;
        html += `<span class="code-var">${varName}</span> `;
        html += `<span class="code-keyword">in</span> `;
        html += `<span class="code-number">0</span>..<span class="code-number">${MATRIX_SIZE}</span>:`;
        if (isCurrent && iter) {
            html += ` <span class="code-comment">← ${varName}=${iter[varName]}</span>`;
        }
        html += '</div>';
    }

    html += `<div class="code-line">${indent[3]}C[i][j] += A[i][k] * B[k][j]</div>`;
    return html;
}

/**
 * Generate HTML for tiled loop code.
 */
function generateTiledCodeHTML(order, tileSize) {
    const iter = state.iterations[state.currentIteration];
    const indent = ['', '  ', '    ', '      ', '        ', '          ', '            '];
    let html = '';

    // Tile loops
    for (let level = 0; level < 3; level++) {
        const varName = 't' + order[level];
        html += `<div class="code-line">`;
        html += `${indent[level]}<span class="code-keyword">for</span> `;
        html += `<span class="code-var">${varName}</span> `;
        html += `<span class="code-keyword">in</span> `;
        html += `<span class="code-number">0</span>..<span class="code-number">${MATRIX_SIZE}</span> `;
        html += `<span class="code-keyword">step</span> <span class="code-number">${tileSize}</span>:`;
        if (iter && iter[varName] !== undefined) {
            html += ` <span class="code-comment">← ${varName}=${iter[varName]}</span>`;
        }
        html += '</div>';
    }

    // Inner element loops
    for (let level = 0; level < 3; level++) {
        const varName = order[level];
        const tileVar = 't' + varName;
        const isInnermost = level === 2;
        const isCurrent = isInnermost && state.currentIteration < state.iterations.length;

        html += `<div class="code-line${isCurrent ? ' current' : ''}">`;
        html += `${indent[level + 3]}<span class="code-keyword">for</span> `;
        html += `<span class="code-var">${varName}</span> `;
        html += `<span class="code-keyword">in</span> `;
        html += `<span class="code-var">${tileVar}</span>..<span class="code-var">${tileVar}</span>+<span class="code-number">${tileSize}</span>:`;
        if (isCurrent && iter) {
            html += ` <span class="code-comment">← ${varName}=${iter[varName]}</span>`;
        }
        html += '</div>';
    }

    html += `<div class="code-line">${indent[6]}C[i][j] += A[i][k] * B[k][j]</div>`;
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
    console.log('Matrix Tiling & Cache Visualizer initialized');
    initCanvases();
    setupEventHandlers();
    applyConfiguration();
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', init);
