/* perspective_dcide/libs/glyph_router_map.js
   Runtime registry mapping glyph code-points (0xE200–0xE21F) to animation
   handlers powered by anime.js. Import this map in glyph_router.js and invoke
   the handler when an element renders a glyph in this range.
*/
import { loadAnimeV3 } from './anime_v3/anime_loader.js';

const glyphRouterMap = new Map();

function makeBasicAnimation(code) {
  // Return a handler that runs when the glyph element is mounted
  return async (element, opts = {}) => {
    const anime = await loadAnimeV3();
    switch (code) {
      case 0xE200: // simple fadeIn
        anime({ targets: element, opacity: [0, 1], duration: opts.duration || 500, easing: 'easeOutQuad' });
        break;
      case 0xE201: // pulse scale
        anime({ targets: element, scale: [0.8, 1], duration: 600, easing: 'easeOutBack' });
        break;
      case 0xE202: // rotate
        anime({ targets: element, rotate: ['-10deg', '10deg'], direction: 'alternate', loop: 3, duration: 400 });
        break;
      default:
        // fallback fade + slight up motion
        anime({ targets: element, opacity: [0, 1], translateY: [10, 0], duration: 500, easing: 'easeOutCubic' });
    }
  };
}

for (let code = 0xE200; code <= 0xE21F; code++) {
  glyphRouterMap.set(code, makeBasicAnimation(code));
}

export default glyphRouterMap; 