/**
 * Loop Structure Code Display
 */

import { state, operation } from './state.js';

/**
 * Get the loop order array for the current configuration.
 */
function getLoopOrderArray() {
    return operation.loopOrders[state.loopOrder] || operation.loopDims;
}

/**
 * Update the loop code display.
 */
export function updateCodeDisplay() {
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
    const bounds = operation.loopBounds;
    let html = '';
    const numLoops = order.length;

    const getIndent = (level) => '  '.repeat(level);

    for (let level = 0; level < numLoops; level++) {
        const varName = order[level];
        const bound = bounds[varName];
        const isInnermost = level === numLoops - 1;
        const isCurrent = isInnermost && state.currentIteration < state.iterations.length;

        html += `<div class="code-line${isCurrent ? ' current' : ''}">`;
        html += `${getIndent(level)}<span class="code-keyword">for</span> `;
        html += `<span class="code-var">${varName}</span> `;
        html += `<span class="code-keyword">in</span> `;
        html += `<span class="code-number">0</span>..<span class="code-number">${bound}</span>:`;
        if (isCurrent && iter) {
            html += ` <span class="code-comment">← ${varName}=${iter[varName]}</span>`;
        }
        html += '</div>';
    }

    html += `<div class="code-line">${getIndent(numLoops)}${operation.codeTemplate}</div>`;
    return html;
}

/**
 * Generate HTML for tiled loop code.
 */
function generateTiledCodeHTML(order, tileSize) {
    const iter = state.iterations[state.currentIteration];
    const bounds = operation.loopBounds;
    const tileableDims = new Set(operation.tileableDims || operation.loopDims);
    let html = '';
    let indentLevel = 0;

    const getIndent = (level) => '  '.repeat(level);

    const firstTiledIdx = order.findIndex(d => tileableDims.has(d));
    if (firstTiledIdx === -1) {
        return generateNonTiledCodeHTML(order);
    }

    // Phase 1: Non-tiled dims before first tiled dim
    for (let i = 0; i < firstTiledIdx; i++) {
        const dim = order[i];
        const bound = bounds[dim];
        html += `<div class="code-line">`;
        html += `${getIndent(indentLevel)}<span class="code-keyword">for</span> `;
        html += `<span class="code-var">${dim}</span> `;
        html += `<span class="code-keyword">in</span> `;
        html += `<span class="code-number">0</span>..<span class="code-number">${bound}</span>:`;
        if (iter) {
            html += ` <span class="code-comment">← ${dim}=${iter[dim]}</span>`;
        }
        html += '</div>';
        indentLevel++;
    }

    // Phase 2: Tile loops for tiled dims
    for (const dim of order) {
        if (tileableDims.has(dim)) {
            const varName = 't' + dim;
            const bound = bounds[dim];
            html += `<div class="code-line">`;
            html += `${getIndent(indentLevel)}<span class="code-keyword">for</span> `;
            html += `<span class="code-var">${varName}</span> `;
            html += `<span class="code-keyword">in</span> `;
            html += `<span class="code-number">0</span>..<span class="code-number">${bound}</span> `;
            html += `<span class="code-keyword">step</span> <span class="code-number">${tileSize}</span>:`;
            if (iter && iter[varName] !== undefined) {
                html += ` <span class="code-comment">← ${varName}=${iter[varName]}</span>`;
            }
            html += '</div>';
            indentLevel++;
        }
    }

    // Phase 3: Everything from first tiled dim onwards
    for (let i = firstTiledIdx; i < order.length; i++) {
        const dim = order[i];
        const bound = bounds[dim];
        const isLastLoop = i === order.length - 1;
        const isCurrent = isLastLoop && state.currentIteration < state.iterations.length;

        if (tileableDims.has(dim)) {
            const tileVar = 't' + dim;
            html += `<div class="code-line${isCurrent ? ' current' : ''}">`;
            html += `${getIndent(indentLevel)}<span class="code-keyword">for</span> `;
            html += `<span class="code-var">${dim}</span> `;
            html += `<span class="code-keyword">in</span> `;
            html += `<span class="code-var">${tileVar}</span>..<span class="code-var">${tileVar}</span>+<span class="code-number">${tileSize}</span>:`;
            if (isCurrent && iter) {
                html += ` <span class="code-comment">← ${dim}=${iter[dim]}</span>`;
            }
            html += '</div>';
        } else {
            html += `<div class="code-line${isCurrent ? ' current' : ''}">`;
            html += `${getIndent(indentLevel)}<span class="code-keyword">for</span> `;
            html += `<span class="code-var">${dim}</span> `;
            html += `<span class="code-keyword">in</span> `;
            html += `<span class="code-number">0</span>..<span class="code-number">${bound}</span>:`;
            if (isCurrent && iter) {
                html += ` <span class="code-comment">← ${dim}=${iter[dim]}</span>`;
            }
            html += '</div>';
        }
        indentLevel++;
    }

    html += `<div class="code-line">${getIndent(indentLevel)}${operation.codeTemplate}</div>`;
    return html;
}
