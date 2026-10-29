'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const deepmerge = require('./index.js');

test('an own __proto__ key in the source does not replace the prototype', () => {
	const source = JSON.parse('{"__proto__": {"polluted": true}, "kept": 1}');
	for (const target of [null, 'text', {}, { kept: 0 }]) {
		const merged = deepmerge(target, source);
		assert.equal(Object.getPrototypeOf(merged), Object.prototype);
		assert.equal(merged.polluted, undefined);
	}
	assert.equal({}.polluted, undefined);
});

test('an own __proto__ key in the target does not replace the prototype', () => {
	const target = JSON.parse('{"__proto__": {"polluted": true}, "a": 1}');
	const merged = deepmerge(target, { b: 2 });
	assert.equal(Object.getPrototypeOf(merged), Object.prototype);
	assert.equal(merged.polluted, undefined);
	assert.deepEqual(merged, { a: 1, b: 2 });
});

test('merging behaves like deepmerge 4.3.1', () => {
	const replaceArrays = { arrayMerge: (target, source) => source ?? target };
	assert.deepEqual(
		deepmerge({ a: { x: 1, list: [1] }, b: 1 }, { a: { y: 2, list: [2] }, c: 3 }),
		{ a: { x: 1, y: 2, list: [1, 2] }, b: 1, c: 3 }
	);
	assert.deepEqual(deepmerge({ list: [1] }, { list: [2] }, replaceArrays), { list: [2] });
	assert.deepEqual(deepmerge.all([{ a: 1 }, { b: 2 }, { a: 3 }]), { a: 3, b: 2 });
	const nested = { deep: { value: 1 } };
	const merged = deepmerge({}, nested);
	assert.notEqual(merged.deep, nested.deep);
});
