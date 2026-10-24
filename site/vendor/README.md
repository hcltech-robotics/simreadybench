`model-viewer.min.js` is @google/model-viewer 4.4.0, vendored from the npm release. Apache-2.0; see model-viewer-LICENSE.txt. Source: https://github.com/google/model-viewer/tree/v4.4.0

`three-0.186.1/comparison-three.min.js` bundles Three.js 0.186.1, its `GLTFLoader`
and `OrbitControls` for the object comparison viewer. MIT; see
`three-0.186.1/LICENSE.txt`. Source: https://github.com/mrdoob/three.js/tree/9b4a2ac29c63ccb43fd51c5661f2f873ac2c39b8

The exact npm package is https://registry.npmjs.org/three/-/three-0.186.1.tgz
(upstream git revision `9b4a2ac29c63ccb43fd51c5661f2f873ac2c39b8`). Its verified
package integrity is
`sha512-blFeqb49wRCSGUGj7gtpfnSGHy2lwDk94RhUmS1c/hTby70kvChbWpkJ4Pm1390LqzzvTmzgXKHPEafJwCb8jA==`.

The local ESM bundle was built with esbuild 0.27.0:

```sh
esbuild entry.js --bundle --minify --format=esm --target=es2020 \
  --legal-comments=inline --outfile=comparison-three.min.js
```

`entry.js` exports these classes/constants from the pinned package, plus
`GLTFLoader` and `OrbitControls` from its addon modules:

```js
export {
  Box3, BufferAttribute, BufferGeometry, Color, DirectionalLight, DoubleSide,
  HemisphereLight, MeshStandardMaterial, PerspectiveCamera, Points, PointsMaterial,
  Scene, Sphere, Spherical, SRGBColorSpace, Vector3, WebGLRenderer,
} from 'three';
export { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
export { OrbitControls } from 'three/addons/controls/OrbitControls.js';
```

The published bundle contains resolved dependencies and makes no runtime requests
to a package CDN.
