/**
 * UI Module
 * Application UI, state, and event handling
 */

import { operation } from './state.js';
import { generateTensorUI, initCanvases, generateLoopOrderOptions, generateTileSizeOptions } from './generation.js';
import { setupEventHandlers, applyConfiguration } from './handlers.js';
import { setupTourHandlers } from './tour.js';

export { state, operation, currentMode, createTensorState } from './state.js';
export { generateTensorUI, initCanvases, generateLoopOrderOptions, generateTileSizeOptions } from './generation.js';
export {
    render, resetSimulation, jumpToIteration, stepForward, stepBackward,
    startAnimation, stopAnimation, togglePlayPause,
    applyConfiguration, updateCacheDisplays, switchMode, setupEventHandlers
} from './handlers.js';
export { updateCodeDisplay } from './code-display.js';
export { tour, setupTourHandlers } from './tour.js';

/**
 * Initialize the application.
 */
export function init() {
    if (!document.getElementById('tensorsContainer')) {
        console.log('Skipping UI initialization (test mode)');
        return;
    }

    console.log('Cache Visualizer initialized');
    console.log(`Operation: ${operation.displayName}`);
    console.log(`Tensors: ${operation.tensors.map(t => t.name).join(', ')}`);

    generateTensorUI();
    generateLoopOrderOptions();
    generateTileSizeOptions();

    initCanvases();
    setupEventHandlers();
    setupTourHandlers();
    applyConfiguration();

    const helpBtn = document.getElementById('helpBtn');
    helpBtn.classList.add('attention');
    helpBtn.addEventListener('animationend', () => {
        helpBtn.classList.remove('attention');
    });
}
