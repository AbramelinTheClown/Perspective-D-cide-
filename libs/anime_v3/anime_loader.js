/* perspective_dcide/libs/anime_v3/anime_loader.js
   Lazy-loads the minified anime.js v3 bundle produced by lib_bundler.
   Caches the promise so subsequent glyphs reuse the same module.
*/
let animePromise;

function getLocalBlobUrl(file) {
  const base = window.location.origin + '/perspective_dcide/libs/anime_v3/dist/';
  return base + file;
}

export function loadAnimeV3() {
  if (animePromise) return animePromise;
  const blobUrl = getLocalBlobUrl('anime_v3.min.js');
  animePromise = import(/* webpackIgnore: true */ blobUrl).then((mod) => {
    return mod.default || window.anime || mod;
  });
  return animePromise;
} 