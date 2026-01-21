/**
 * Timeline and memory layout visualizations
 */

import { isElementInCacheByCoords } from './tensors.js';

/**
 * Render the cache hit timeline.
 */
export function renderTimeline(ctx, operation, history, currentIteration) {
    const canvas = ctx.canvas;
    const width = canvas.width;
    const height = canvas.height;
    const numTensors = operation.tensors.length;

    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);

    if (history.length === 0) return;

    const rowHeight = height / numTensors;
    const labelOffset = 15;
    const barWidthScaled = (width - labelOffset) / operation.getTotalIterations();

    ctx.fillStyle = '#666';
    ctx.font = '10px monospace';
    operation.tensors.forEach((tensor, idx) => {
        ctx.fillText(tensor.name, 2, idx * rowHeight + rowHeight / 2 + 3);
    });

    for (let i = 0; i < history.length; i++) {
        const x = labelOffset + i * barWidthScaled;
        const h = history[i];
        const barW = Math.max(1, barWidthScaled - 0.5);

        operation.tensors.forEach((tensor, idx) => {
            ctx.fillStyle = h[tensor.name] ? '#28a745' : '#dc3545';
            ctx.fillRect(x, idx * rowHeight + 2, barW, rowHeight - 4);
        });
    }

    if (currentIteration > 0) {
        const x = labelOffset + (currentIteration - 1) * barWidthScaled;
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x + barWidthScaled, 0);
        ctx.lineTo(x + barWidthScaled, height);
        ctx.stroke();
    }
}

/**
 * Render the memory layout visualization.
 */
export function renderMemoryLayout(ctx, operation, iterations, currentIteration, cache, layouts, elementsPerLine, elementSize) {
    const canvas = ctx.canvas;
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);

    const numTensors = operation.tensors.length;
    const tensorSizes = operation.tensors.map(t => t.getTotalElements());
    const maxElements = Math.max(...tensorSizes);

    const rowHeight = height / numTensors;
    const labelOffset = 45;
    const barWidth = width - labelOffset;

    const iter = iterations[currentIteration];

    operation.tensors.forEach((tensor, tensorIdx) => {
        const y = tensorIdx * rowHeight;
        const layout = layouts[tensor.name];
        const numElements = tensor.getTotalElements();
        const elemWidth = barWidth / maxElements;

        const xStart = labelOffset;

        ctx.fillStyle = '#888';
        ctx.font = '9px monospace';
        ctx.fillText(tensor.name, 2, y + rowHeight / 2 + 3);

        let currentLinearIndex = -1;
        if (iter) {
            currentLinearIndex = tensor.getLinearIndex(iter, layout);
        }

        for (let linearIdx = 0; linearIdx < numElements; linearIdx++) {
            const x = xStart + linearIdx * elemWidth;
            const coords = tensor.getCoordinatesFromLinear(linearIdx, layout);
            const inCache = cache ? isElementInCacheByCoords(tensor, coords, cache, layouts, elementSize) : false;

            if (linearIdx === currentLinearIndex) {
                ctx.fillStyle = '#000000';
                ctx.fillRect(x, y + 3, Math.max(1, elemWidth - 0.5), rowHeight - 6);
                ctx.strokeStyle = '#667eea';
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y + 2, Math.max(1, elemWidth), rowHeight - 4);
            } else if (inCache) {
                ctx.fillStyle = '#28a745';
                ctx.fillRect(x, y + 3, Math.max(1, elemWidth - 0.5), rowHeight - 6);
            } else {
                ctx.fillStyle = '#2a2a4a';
                ctx.fillRect(x, y + 3, Math.max(1, elemWidth - 0.5), rowHeight - 6);
            }
        }

        ctx.strokeStyle = '#555';
        ctx.lineWidth = 1;
        for (let lineStart = 0; lineStart <= numElements; lineStart += elementsPerLine) {
            const x = xStart + lineStart * elemWidth;
            ctx.beginPath();
            ctx.moveTo(x, y + 1);
            ctx.lineTo(x, y + rowHeight - 1);
            ctx.stroke();
        }

        const xEnd = xStart + numElements * elemWidth;
        ctx.strokeStyle = '#888';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(xEnd, y + 1);
        ctx.lineTo(xEnd, y + rowHeight - 1);
        ctx.stroke();
    });
}
