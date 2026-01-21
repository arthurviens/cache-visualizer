/**
 * Guided Tour
 */

import { operation } from './state.js';

/**
 * Get tour steps based on current operation.
 */
function getTourSteps() {
    const opIntro = operation.name === 'matmul'
        ? 'We compute C = A × B, stepping through all i,j,k iterations.'
        : 'We compute Output = Input * Kernel, stepping through all output positions, input channels, and kernel positions.';

    const tensorDesc = operation.name === 'matmul'
        ? 'Each cell shows a matrix element.'
        : 'For convolution, tensors have channels shown side-by-side. The kernel shows all c_out × c_in filter slices.';

    return [
        {
            target: '.matrices-container',
            title: operation.displayName,
            content: `${opIntro} ${tensorDesc} Green highlighting indicates the element is currently in cache. The black dot shows which element is being accessed.`,
            position: 'bottom'
        },
        {
            target: '#memoryLayout',
            title: 'Linear Memory Layout',
            content: 'Tensors are stored as flat arrays in memory. This bar shows each tensor\'s linear address space. Green = in cache. Vertical lines mark cache line boundaries. Watch how access patterns create different locality behaviors.',
            position: 'top'
        },
        {
            target: '.config-panel',
            title: 'Configuration',
            content: 'Control the simulation parameters here. Loop order changes which index varies fastest. Data layout (row/col major) affects how indices map to linear memory addresses. Cache settings control the simulated cache size.',
            position: 'bottom'
        },
        {
            target: '.playback-controls',
            title: 'Playback Controls',
            content: 'Step through iterations one by one, or play continuously. Use the speed slider to control animation speed. You can also jump to any iteration directly.',
            position: 'top'
        },
        {
            target: '.stats-bar',
            title: 'Cache Statistics',
            content: 'Track total memory accesses and cache hits. The hit rate shows cache efficiency. Better locality = higher hit rate = faster real-world performance.',
            position: 'top'
        },
        {
            target: '#timeline',
            title: 'Cache Hit Timeline',
            content: 'History of cache hits (green) and misses (red) for each tensor over time. Patterns here reveal locality behavior: clustered green = good locality, scattered red = poor locality.',
            position: 'top'
        },
        {
            target: '.side-panel',
            title: 'Loop Structure & State',
            content: 'See the actual loop code being executed, with current indices highlighted. The state panel shows exactly which iteration and operation is happening.',
            position: 'left'
        }
    ];
}

/**
 * Tour state and controller.
 */
export const tour = {
    active: false,
    currentStep: 0,
    highlightedElement: null,

    start() {
        this.active = true;
        this.currentStep = 0;
        document.getElementById('tourOverlay').classList.remove('hidden');
        document.getElementById('tourTooltip').classList.remove('hidden');
        this.showStep(0);
    },

    end() {
        this.active = false;
        document.getElementById('tourOverlay').classList.add('hidden');
        document.getElementById('tourTooltip').classList.add('hidden');
        this.clearHighlight();
    },

    showStep(index) {
        const steps = getTourSteps();
        if (index < 0 || index >= steps.length) return;

        this.currentStep = index;
        const step = steps[index];

        document.getElementById('tourTitle').textContent = step.title;
        document.getElementById('tourContent').textContent = step.content;
        document.getElementById('tourStepIndicator').textContent = `${index + 1} / ${steps.length}`;

        document.getElementById('tourPrev').disabled = index === 0;
        const nextBtn = document.getElementById('tourNext');
        nextBtn.textContent = index === steps.length - 1 ? 'Finish' : 'Next';

        this.highlightElement(step.target);
        this.positionTooltip(step.target, step.position);
    },

    highlightElement(selector) {
        this.clearHighlight();
        const element = document.querySelector(selector);
        if (element) {
            element.classList.add('tour-highlight');
            this.highlightedElement = element;
            element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    },

    clearHighlight() {
        if (this.highlightedElement) {
            this.highlightedElement.classList.remove('tour-highlight');
            this.highlightedElement = null;
        }
    },

    positionTooltip(selector, preferredPosition) {
        const tooltip = document.getElementById('tourTooltip');
        const target = document.querySelector(selector);
        if (!target) return;

        const targetRect = target.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        const padding = 16;
        const arrowSpace = 12;

        const spaceTop = targetRect.top;
        const spaceBottom = window.innerHeight - targetRect.bottom;
        const spaceLeft = targetRect.left;
        const spaceRight = window.innerWidth - targetRect.right;

        let position = preferredPosition;
        if (position === 'auto') {
            const spaces = { top: spaceTop, bottom: spaceBottom, left: spaceLeft, right: spaceRight };
            position = Object.entries(spaces).sort((a, b) => b[1] - a[1])[0][0];
        }

        const tooltipHeight = tooltipRect.height;
        const tooltipWidth = tooltipRect.width;

        if (position === 'bottom' && spaceBottom < tooltipHeight + padding) {
            position = 'top';
        } else if (position === 'top' && spaceTop < tooltipHeight + padding) {
            position = 'bottom';
        } else if (position === 'left' && spaceLeft < tooltipWidth + padding) {
            position = 'right';
        } else if (position === 'right' && spaceRight < tooltipWidth + padding) {
            position = 'left';
        }

        let top, left;

        switch (position) {
            case 'top':
                top = targetRect.top - tooltipHeight - arrowSpace;
                left = targetRect.left + (targetRect.width - tooltipWidth) / 2;
                break;
            case 'bottom':
                top = targetRect.bottom + arrowSpace;
                left = targetRect.left + (targetRect.width - tooltipWidth) / 2;
                break;
            case 'left':
                top = targetRect.top + (targetRect.height - tooltipHeight) / 2;
                left = targetRect.left - tooltipWidth - arrowSpace;
                break;
            case 'right':
                top = targetRect.top + (targetRect.height - tooltipHeight) / 2;
                left = targetRect.right + arrowSpace;
                break;
        }

        top = Math.max(padding, Math.min(top, window.innerHeight - tooltipHeight - padding));
        left = Math.max(padding, Math.min(left, window.innerWidth - tooltipWidth - padding));

        tooltip.style.top = `${top}px`;
        tooltip.style.left = `${left}px`;
    },

    next() {
        const steps = getTourSteps();
        if (this.currentStep < steps.length - 1) {
            this.showStep(this.currentStep + 1);
        } else {
            this.end();
        }
    },

    prev() {
        if (this.currentStep > 0) {
            this.showStep(this.currentStep - 1);
        }
    }
};

/**
 * Set up tour event handlers.
 */
export function setupTourHandlers() {
    document.getElementById('helpBtn').addEventListener('click', () => tour.start());
    document.getElementById('tourClose').addEventListener('click', () => tour.end());
    document.getElementById('tourNext').addEventListener('click', () => tour.next());
    document.getElementById('tourPrev').addEventListener('click', () => tour.prev());

    document.addEventListener('keydown', (e) => {
        if (e.code === 'Escape' && tour.active) {
            tour.end();
        }
    });

    window.addEventListener('resize', () => {
        if (tour.active) {
            const steps = getTourSteps();
            const step = steps[tour.currentStep];
            tour.positionTooltip(step.target, step.position);
        }
    });
}
