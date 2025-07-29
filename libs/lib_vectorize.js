import glyphRouterMap from './glyph_router_map.js';

/*
 * lib_vectorize.js – Build a semantic index of animation glyph helpers.
 *
 * Each exported object contains:
 *   codePoint: Unicode code of the glyph (e.g., 0xE200)
 *   animationName: semantic name of the animation helper
 *   description: brief human description of the effect
 *   defaultOptions: defaults applied by the handler
 */

const FALLBACK_NAME = 'fadeInScaleUp';
const FALLBACK_DESC = 'Default fade-in + scale-up animation';

export function buildAnimationIndex() {
  const index = [];
  for (const [codePoint, handler] of glyphRouterMap.entries()) {
    const meta = handler.meta || {};
    const name = meta.name || FALLBACK_NAME;
    const description = meta.description || FALLBACK_DESC;
    const defaultOptions = handler.defaultOptions || {};
    index.push({ codePoint, animationName: name, description, defaultOptions });
  }
  return index;
}

export const animationIndex = buildAnimationIndex();
export const FALLBACK_NAME_CONST = FALLBACK_NAME; 