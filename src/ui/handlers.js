/**
 * Event Handlers and Simulation Logic
 */

import { OPERATIONS } from '../operations/index.js';
import { CacheSimulator, generateIterations, generateTiledIterations, getAccessAddress } from '../simulation/index.js';
import { ELEMENT_SIZE, renderTensor, renderTimeline, renderMemoryLayout } from '../rendering/index.js';
import {
    state, operation, currentMode, canvasContexts, ctxTimeline, ctxMemoryLayout,
    animationId, lastFrameTime, snapshots,
    setCurrentMode, setOperation, createTensorState,
    setAnimationId, setLastFrameTime, setSnapshots, pushSnapshot, popSnapshot
} from './state.js';
import { generateTensorUI, initCanvases, generateLoopOrderOptions, generateTileSizeOptions } from './generation.js';
import { updateCodeDisplay } from './code-display.js';

/**
 * Generate all iterations based on current configuration.
 */
function generateAllIterations() {
    if (state.tilingEnabled) {
        return generateTiledIterations(operation, state.loopOrder, state.tileSize);
    }
    return generateIterations(operation, state.loopOrder);
}

/**
 * Render all tensors.
 */
function renderAllTensors() {
    const iter = state.iterations[state.currentIteration];

    for (const tensor of operation.tensors) {
        const ctx = canvasContexts[tensor.name];
        if (!ctx) continue;
        const indices = iter ? tensor.getIndices(iter) : null;
        renderTensor(ctx, tensor, indices, state.cache, state.layouts, operation.elementSize, state.tilingEnabled, state.tileSize);
    }
}

/**
 * Update statistics display.
 */
function updateStatsDisplay() {
    for (const tensor of operation.tensors) {
        const statsEl = document.getElementById('stats' + tensor.name);
        if (statsEl) {
            const s = state.stats[tensor.name];
            statsEl.textContent = `mem:${s.accesses} hit:${s.hits}`;
        }
    }

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

    document.getElementById('currentIteration').textContent = `${state.currentIteration} of ${operation.getTotalIterations()}`;

    if (iter) {
        let indicesStr = '';
        const tileableDims = new Set(operation.tileableDims || operation.loopDims);
        const hasTileInfo = state.tilingEnabled && operation.tileableDims &&
                           iter['t' + operation.tileableDims[0]] !== undefined;

        if (hasTileInfo) {
            const tileParts = operation.loopDims
                .filter(d => tileableDims.has(d))
                .map(d => `t${d}=${iter['t' + d]}`);
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

    document.getElementById('jumpIteration').max = operation.getTotalIterations() - 1;
    document.getElementById('totalIterations').textContent = operation.getTotalIterations();
}

/**
 * Full render of all visualization components.
 */
export function render() {
    renderAllTensors();
    renderMemoryLayout(ctxMemoryLayout, operation, state.iterations, state.currentIteration, state.cache, state.layouts, state.elementsPerLine, operation.elementSize);
    renderTimeline(ctxTimeline, operation, state.history, state.currentIteration);
    updateStatsDisplay();
    updateStateDisplay();
}

/**
 * Execute one simulation step.
 */
function executeStep() {
    if (state.currentIteration >= state.iterations.length) {
        return null;
    }

    const iter = state.iterations[state.currentIteration];
    const result = {};

    for (const tensor of operation.tensors) {
        const address = getAccessAddress(tensor, iter, state.layouts, operation.elementSize);
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
 * Stop animation.
 */
export function stopAnimation() {
    state.isPlaying = false;
    document.getElementById('playPauseBtn').textContent = '▶';
    if (animationId) {
        cancelAnimationFrame(animationId);
        setAnimationId(null);
    }
}

/**
 * Reset simulation to initial state.
 */
export function resetSimulation() {
    stopAnimation();

    state.currentIteration = 0;
    state.history = [];
    setSnapshots([]);

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
export function jumpToIteration(targetIteration) {
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
export function stepForward() {
    if (state.currentIteration < state.iterations.length) {
        pushSnapshot({
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
export function stepBackward() {
    if (snapshots.length > 0) {
        const snapshot = popSnapshot();
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
                setSnapshots(snapshots.slice(-50));
            }

            pushSnapshot({
                cache: state.cache.snapshot(),
                stats: JSON.parse(JSON.stringify(state.stats)),
                historyLength: state.history.length,
                iteration: state.currentIteration
            });

            executeStep();
            render();
            updateCodeDisplay();
            setLastFrameTime(timestamp);
        } else {
            stopAnimation();
        }
    }

    if (state.isPlaying) {
        setAnimationId(requestAnimationFrame(animate));
    }
}

/**
 * Start animation.
 */
export function startAnimation() {
    if (!state.isPlaying && state.currentIteration < state.iterations.length) {
        state.isPlaying = true;
        document.getElementById('playPauseBtn').textContent = '⏸';
        setLastFrameTime(performance.now());
        setAnimationId(requestAnimationFrame(animate));
    }
}

/**
 * Toggle play/pause.
 */
export function togglePlayPause() {
    if (state.isPlaying) {
        stopAnimation();
    } else {
        startAnimation();
    }
}

/**
 * Apply configuration from UI controls.
 */
export function applyConfiguration() {
    state.loopOrder = document.getElementById('loopOrder').value;
    state.tilingEnabled = document.getElementById('tilingEnabled').checked;
    state.tileSize = parseInt(document.getElementById('tileSize').value);

    state.elementsPerLine = parseInt(document.getElementById('elementsPerLine').value);
    state.numCacheLines = parseInt(document.getElementById('numCacheLines').value);

    for (const tensor of operation.tensors) {
        const layoutEl = document.getElementById('layout' + tensor.name);
        if (layoutEl) {
            state.layouts[tensor.name] = layoutEl.value;
        }
    }

    const cacheLineSize = state.elementsPerLine * ELEMENT_SIZE;
    const cacheCapacity = cacheLineSize * state.numCacheLines;

    state.iterations = generateAllIterations();
    state.cache = new CacheSimulator(cacheCapacity, cacheLineSize);

    resetSimulation();
}

/**
 * Update the byte equivalent displays for cache configuration.
 */
export function updateCacheDisplays() {
    const elementsPerLine = parseInt(document.getElementById('elementsPerLine').value);
    const numCacheLines = parseInt(document.getElementById('numCacheLines').value);

    const lineBytes = elementsPerLine * ELEMENT_SIZE;
    const totalBytes = lineBytes * numCacheLines;

    document.getElementById('lineSizeBytes').textContent = `= ${lineBytes}B`;
    document.getElementById('cacheSizeBytes').textContent = `= ${totalBytes}B`;
}

/**
 * Switch to a different operation mode (matmul, conv2d, etc.)
 */
export function switchMode(mode) {
    if (mode === currentMode) return;

    const opConfig = OPERATIONS[mode];
    if (!opConfig || !opConfig.create) {
        console.log(`Mode "${mode}" not yet implemented`);
        return;
    }

    stopAnimation();

    setCurrentMode(mode);

    const newOp = opConfig.create();
    setOperation(newOp);

    const newTensorState = createTensorState(newOp);
    state.layouts = newTensorState.layouts;
    state.stats = newTensorState.stats;

    document.querySelectorAll('.mode-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.mode === mode);
    });
    document.getElementById('pageTitle').textContent = opConfig.title;

    generateTensorUI();
    generateLoopOrderOptions();
    generateTileSizeOptions();

    initCanvases();

    state.loopOrder = opConfig.defaultLoopOrder;
    document.getElementById('loopOrder').value = state.loopOrder;

    applyConfiguration();

    console.log(`Switched to mode: ${mode}`);
    console.log(`Tensors: ${newOp.tensors.map(t => t.name).join(', ')}`);
    console.log(`Total iterations: ${newOp.getTotalIterations()}`);
}

/**
 * Set up all event handlers.
 */
export function setupEventHandlers() {
    document.querySelectorAll('.mode-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            switchMode(tab.dataset.mode);
        });
    });

    document.getElementById('tilingEnabled').addEventListener('change', (e) => {
        document.getElementById('tileSize').disabled = !e.target.checked;
    });

    document.getElementById('elementsPerLine').addEventListener('change', updateCacheDisplays);
    document.getElementById('numCacheLines').addEventListener('change', updateCacheDisplays);

    const applyBtn = document.getElementById('applyConfig');
    applyBtn.addEventListener('click', () => {
        applyConfiguration();
        applyBtn.classList.remove('needs-apply');
    });

    const staticConfigInputs = [
        'loopOrder', 'tilingEnabled', 'tileSize',
        'elementsPerLine', 'numCacheLines'
    ];
    staticConfigInputs.forEach(id => {
        document.getElementById(id).addEventListener('change', () => {
            applyBtn.classList.add('needs-apply');
        });
    });

    document.getElementById('layoutControls').addEventListener('change', (e) => {
        if (e.target.tagName === 'SELECT') {
            applyBtn.classList.add('needs-apply');
        }
    });

    document.getElementById('playPauseBtn').addEventListener('click', togglePlayPause);
    document.getElementById('stepFwdBtn').addEventListener('click', stepForward);
    document.getElementById('stepBackBtn').addEventListener('click', stepBackward);
    document.getElementById('resetBtn').addEventListener('click', resetSimulation);

    document.getElementById('speedSlider').addEventListener('input', (e) => {
        const speed = Math.round(Math.pow(10, e.target.value / 50));
        state.speed = speed;
        document.getElementById('speedDisplay').textContent = speed + 'x';
    });

    document.getElementById('jumpIteration').addEventListener('change', (e) => {
        const target = Math.min(Math.max(0, parseInt(e.target.value) || 0), operation.getTotalIterations() - 1);
        jumpToIteration(target);
        e.target.value = state.currentIteration;
    });

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

    window.addEventListener('resize', () => {
        const timelineCanvas = document.getElementById('timeline');
        timelineCanvas.width = timelineCanvas.offsetWidth;
        const memoryLayoutCanvas = document.getElementById('memoryLayout');
        memoryLayoutCanvas.width = memoryLayoutCanvas.offsetWidth;
        renderMemoryLayout(ctxMemoryLayout, operation, state.iterations, state.currentIteration, state.cache, state.layouts, state.elementsPerLine, operation.elementSize);
        renderTimeline(ctxTimeline, operation, state.history, state.currentIteration);
    });
}
