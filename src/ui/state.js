/**
 * Application State Management
 */

import { OPERATIONS, createMatmulOperation } from '../operations/index.js';
import { MATRIX_SIZE, ELEMENT_SIZE } from '../rendering/config.js';

// Current mode and operation
export let currentMode = 'matmul';
export let operation = OPERATIONS[currentMode].create();

/**
 * Initialize tensor state (layouts, stats) from operation definition.
 */
export function createTensorState(op) {
    const layouts = {};
    const stats = {};

    for (const tensor of op.tensors) {
        if (tensor.layoutOptions && tensor.layoutOptions.length > 0) {
            layouts[tensor.name] = tensor.layoutOptions[0].value;
        } else {
            layouts[tensor.name] = 'row';
        }
        stats[tensor.name] = { accesses: 0, hits: 0 };
    }

    return { layouts, stats };
}

const initialTensorState = createTensorState(operation);

// Main application state
export const state = {
    // Configuration (from UI)
    loopOrder: 'ijk',
    tilingEnabled: false,
    tileSize: 4,
    layouts: initialTensorState.layouts,
    elementsPerLine: 16,
    numCacheLines: 4,

    // Simulation state
    currentIteration: 0,
    isPlaying: false,
    speed: 10,

    // Generated data
    iterations: [],
    cache: null,

    // Statistics (per-tensor)
    stats: initialTensorState.stats,

    // History for timeline
    history: []
};

// Canvas contexts
export let canvasContexts = {};
export let ctxTimeline = null;
export let ctxMemoryLayout = null;

// Animation state
export let animationId = null;
export let lastFrameTime = 0;
export let snapshots = [];

// Setters for module-level variables
export function setCurrentMode(mode) {
    currentMode = mode;
}

export function setOperation(op) {
    operation = op;
}

export function setCanvasContexts(contexts) {
    canvasContexts = contexts;
}

export function setCtxTimeline(ctx) {
    ctxTimeline = ctx;
}

export function setCtxMemoryLayout(ctx) {
    ctxMemoryLayout = ctx;
}

export function setAnimationId(id) {
    animationId = id;
}

export function setLastFrameTime(time) {
    lastFrameTime = time;
}

export function setSnapshots(snaps) {
    snapshots = snaps;
}

export function pushSnapshot(snap) {
    snapshots.push(snap);
}

export function popSnapshot() {
    return snapshots.pop();
}
