/**
 * LRU Cache Simulator
 *
 * Supports multi-level cache hierarchies. Each instance represents one cache level.
 * The `level` property indicates which level this cache represents (1=L1, 2=L2, 3=L3).
 */

export class CacheSimulator {
    /**
     * @param {number} capacityBytes - Total cache capacity in bytes
     * @param {number} lineSize - Cache line size in bytes
     * @param {number} level - Cache level (1=L1, 2=L2, 3=L3). Defaults to 1.
     */
    constructor(capacityBytes, lineSize, level = 1) {
        this.capacityBytes = capacityBytes;
        this.lineSize = lineSize;
        this.level = level;
        this.maxLines = Math.floor(capacityBytes / lineSize);
        this.lines = [];
        this.totalAccesses = 0;
        this.hits = 0;
        this.misses = 0;
    }

    getLineAddress(address) {
        return Math.floor(address / this.lineSize) * this.lineSize;
    }

    /**
     * Access an address in the cache.
     * @param {number} address - Memory address to access
     * @returns {{ hit: boolean, level: number | null }} - hit indicates cache hit/miss,
     *          level is this cache's level if hit, null if miss
     */
    access(address) {
        this.totalAccesses++;
        const lineAddr = this.getLineAddress(address);
        const index = this.lines.indexOf(lineAddr);

        if (index !== -1) {
            this.lines.splice(index, 1);
            this.lines.push(lineAddr);
            this.hits++;
            return { hit: true, level: this.level };
        } else {
            this.misses++;
            if (this.lines.length >= this.maxLines) {
                this.lines.shift();
            }
            this.lines.push(lineAddr);
            return { hit: false, level: null };
        }
    }

    /**
     * Check if an address is currently cached.
     * @param {number} address - Memory address to check
     * @returns {number | null} - This cache's level if cached, null if not
     */
    isAddressCached(address) {
        const lineAddr = this.getLineAddress(address);
        return this.lines.includes(lineAddr) ? this.level : null;
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

    restore(snap) {
        this.lines = [...snap.lines];
        this.totalAccesses = snap.totalAccesses;
        this.hits = snap.hits;
        this.misses = snap.misses;
    }
}
