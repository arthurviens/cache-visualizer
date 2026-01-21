/**
 * Dynamic UI Generation
 * Creates UI elements from operation definition
 */

import { operation, canvasContexts, setCanvasContexts, setCtxTimeline, setCtxMemoryLayout } from './state.js';
import { getTensorCanvasSize } from '../rendering/index.js';

/**
 * Generate all dynamic UI elements from the current operation definition.
 */
export function generateTensorUI() {
    generateLayoutControls();
    generateTensorCanvases();
    generateTensorStatsCards();
    updateTimelineLabel();
}

/**
 * Generate layout control dropdowns for each tensor.
 */
export function generateLayoutControls() {
    const container = document.getElementById('layoutControls');
    container.innerHTML = '';

    for (const tensor of operation.tensors) {
        const item = document.createElement('div');
        item.className = 'layout-item';

        let optionsHTML;
        if (tensor.layoutOptions && tensor.layoutOptions.length > 0) {
            optionsHTML = tensor.layoutOptions
                .map(opt => `<option value="${opt.value}">${opt.label}</option>`)
                .join('');
        } else {
            optionsHTML = `
                <option value="row">Row-major</option>
                <option value="col">Col-major</option>
            `;
        }

        item.innerHTML = `
            <span>${tensor.name}</span>
            <select id="layout${tensor.name}">${optionsHTML}</select>
        `;
        container.appendChild(item);
    }
}

/**
 * Generate tensor canvas wrappers and canvases.
 */
export function generateTensorCanvases() {
    const container = document.getElementById('tensorsContainer');
    container.innerHTML = '';

    const tensors = operation.tensors;
    const operators = operation.tensorOperators || [];

    for (let i = 0; i < tensors.length; i++) {
        const tensor = tensors[i];
        const { width, height } = getTensorCanvasSize(tensor);

        const wrapper = document.createElement('div');
        wrapper.className = 'matrix-wrapper';
        wrapper.innerHTML = `
            <div class="matrix-header">
                <span class="matrix-name">${tensor.name}</span>
                <span class="matrix-stats" id="stats${tensor.name}">mem:0 hit:0</span>
            </div>
            <canvas id="matrix${tensor.name}" class="matrix-canvas" width="${width}" height="${height}"></canvas>
        `;
        container.appendChild(wrapper);

        if (i < operators.length) {
            const operatorSpan = document.createElement('span');
            operatorSpan.className = 'operator';
            operatorSpan.textContent = operators[i];
            container.appendChild(operatorSpan);
        }
    }
}

/**
 * Generate per-tensor statistics cards.
 */
export function generateTensorStatsCards() {
    const container = document.getElementById('tensorStatsContainer');
    container.innerHTML = '';

    for (const tensor of operation.tensors) {
        const card = document.createElement('div');
        card.className = 'matrix-stat-card';
        card.innerHTML = `
            <div class="name">${tensor.name}</div>
            <div class="value" id="detailStats${tensor.name}">0/0 hits</div>
        `;
        container.appendChild(card);
    }
}

/**
 * Update the timeline label with current tensor names.
 */
export function updateTimelineLabel() {
    const names = operation.tensors.map(t => t.name).join(' | ');
    document.getElementById('timelineLabel').textContent = `Cache Hit Global Timeline (${names})`;
}

/**
 * Initialize canvas contexts for all tensors.
 */
export function initCanvases() {
    const contexts = {};
    for (const tensor of operation.tensors) {
        const canvas = document.getElementById('matrix' + tensor.name);
        if (canvas) {
            contexts[tensor.name] = canvas.getContext('2d');
        }
    }
    setCanvasContexts(contexts);

    const timelineCanvas = document.getElementById('timeline');
    setCtxTimeline(timelineCanvas.getContext('2d'));
    timelineCanvas.width = timelineCanvas.offsetWidth;
    timelineCanvas.height = 60;

    const memoryLayoutCanvas = document.getElementById('memoryLayout');
    setCtxMemoryLayout(memoryLayoutCanvas.getContext('2d'));
    memoryLayoutCanvas.width = memoryLayoutCanvas.offsetWidth;
    memoryLayoutCanvas.height = 80;
}

/**
 * Generate loop order dropdown options based on current operation.
 */
export function generateLoopOrderOptions() {
    const select = document.getElementById('loopOrder');
    select.innerHTML = '';

    for (const [key, order] of Object.entries(operation.loopOrders)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = order.join(' → ');
        select.appendChild(option);
    }
}

/**
 * Generate tile size dropdown options based on current operation.
 */
export function generateTileSizeOptions() {
    const select = document.getElementById('tileSize');
    const currentValue = parseInt(select.value);
    select.innerHTML = '';

    const tileSizes = operation.tileSizes || [2, 4, 6];
    const tileableDims = operation.tileableDims || operation.loopDims;

    for (const size of tileSizes) {
        const option = document.createElement('option');
        option.value = size;
        if (tileableDims.length < operation.loopDims.length) {
            option.textContent = `${size}×${size} (${tileableDims.join(', ')})`;
        } else {
            option.textContent = `${size}×${size}`;
        }
        select.appendChild(option);
    }

    if (tileSizes.includes(currentValue)) {
        select.value = currentValue;
    } else {
        select.value = tileSizes[0];
    }
}
