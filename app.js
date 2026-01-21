/**
 * Tensor Operations Cache Behavior Visualizer
 * Browser entry point - imports from modular source
 */

import { init } from './src/ui/index.js';

// Re-export everything for compatibility
export {
    createMatmulOperation,
    createConv2dOperation,
    OPERATIONS
} from './src/operations/index.js';

export {
    CacheSimulator,
    generateIterations,
    generateTiledIterations,
    getLinearIndex,
    getTensorAddress,
    getAccessAddress
} from './src/simulation/index.js';

export {
    MATRIX_SIZE,
    ELEMENT_SIZE,
    CELL_SIZE,
    COLORS,
    ISO
} from './src/rendering/index.js';

export {
    state,
    operation
} from './src/ui/index.js';

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
