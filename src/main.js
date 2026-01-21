/**
 * Main Entry Point
 * Exports for browser initialization and test compatibility
 */

// Re-export operations
export { createMatmulOperation, createConv2dOperation, OPERATIONS } from './operations/index.js';

// Re-export simulation
export { CacheSimulator, generateIterations, generateTiledIterations, getLinearIndex, getTensorAddress, getAccessAddress } from './simulation/index.js';

// Re-export rendering
export { MATRIX_SIZE, ELEMENT_SIZE, CELL_SIZE, COLORS, ISO } from './rendering/index.js';

// Re-export UI
export { init, state, operation } from './ui/index.js';
