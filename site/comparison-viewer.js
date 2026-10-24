import {
  Box3, BufferAttribute, BufferGeometry, Color, DirectionalLight, DoubleSide,
  HemisphereLight, MeshStandardMaterial, PerspectiveCamera, Points, PointsMaterial,
  Scene, Spherical, SRGBColorSpace, Vector3, WebGLRenderer, GLTFLoader, OrbitControls,
} from './vendor/three-0.186.1/comparison-three.min.js';

const COLOUR_STOPS = ['#27516c', '#2a9d8f', '#e9c46a', '#e76f51'].map(value => new Color(value));
const BACKGROUND = '#f5f6f8';
const nonNegative = value => Number.isFinite(value) && value >= 0 ? value : 0;

function disposeObject(root) {
  if (!root) return;
  const geometries = new Set();
  const materials = new Set();
  const textures = new Set();
  root.traverse(object => {
    if (object.geometry) geometries.add(object.geometry);
    for (const material of [object.material].flat().filter(Boolean)) {
      materials.add(material);
      for (const value of Object.values(material)) {
        if (value?.isTexture) textures.add(value);
      }
    }
  });
  textures.forEach(texture => texture.dispose());
  materials.forEach(material => material.dispose());
  geometries.forEach(geometry => geometry.dispose());
  root.removeFromParent();
}

function neutraliseMaterials(root) {
  const materials = new Set();
  const textures = new Set();
  const replacement = new MeshStandardMaterial({
    color: '#a5adb8', roughness: 0.72, metalness: 0.02, side: DoubleSide,
  });
  root.traverse(object => {
    if (!object.isMesh) return;
    for (const material of [object.material].flat().filter(Boolean)) {
      materials.add(material);
      for (const value of Object.values(material)) {
        if (value?.isTexture) textures.add(value);
      }
    }
    object.material = replacement;
  });
  textures.forEach(texture => texture.dispose());
  materials.forEach(material => material.dispose());
}

function meshScene() {
  const scene = new Scene();
  scene.background = new Color(BACKGROUND);
  scene.add(new HemisphereLight('#ffffff', '#7a8290', 2.8));
  const key = new DirectionalLight('#ffffff', 3.0);
  key.position.set(3, 5, 4);
  scene.add(key);
  const fill = new DirectionalLight('#dce7f5', 1.1);
  fill.position.set(-4, 1, -2);
  scene.add(fill);
  return scene;
}

/** One world coordinate system, camera and canvas for both reconstruction meshes. */
export class ComparisonViewer {
  constructor(container, { onState = () => {}, onCameraChange = () => {} } = {}) {
    this.container = container;
    this.onState = onState;
    this.onCameraChange = onCameraChange;
    this.referenceScene = meshScene();
    this.candidateScene = meshScene();
    this.deltaScene = new Scene();
    this.deltaScene.background = new Color(BACKGROUND);
    this.camera = new PerspectiveCamera(35, 1, 0.001, 1000);
    this.camera.position.set(1, 0.58, 1);
    this.loader = new GLTFLoader();
    this.mode = 'swipe';
    this.split = 0.5;
    this.thresholdMetres = 0;
    this.generation = 0;
    this.loading = false;
    this.deltaLoading = false;
    this.disposed = false;
    this.contextLost = false;
    this.reference = null;
    this.candidate = null;
    this.deltaPoints = null;
    this.referenceSrc = null;
    this.candidateSrc = null;
    this.deltaSrc = null;
    this.deltaUrl = null;
    this.deltaMeta = {};
    this.errors = {};
    this.pointCount = 0;
    this.visiblePointCount = 0;
    this.maxDistanceMetres = 0;
    this.width = 1;
    this.height = 1;
    this.frame = null;
    this.controller = null;
    this.deltaPromise = null;
    this.pairRequest = null;
    this.reloadOnRestore = false;
    this.bounds = new Box3();

    try {
      this.renderer = new WebGLRenderer({ antialias: true, alpha: false });
      this.renderer.outputColorSpace = SRGBColorSpace;
      this.renderer.autoClear = false;
      this.renderer.setClearColor(BACKGROUND);
      this.canvas = this.renderer.domElement;
      this.canvas.tabIndex = 0;
      this.canvas.setAttribute('role', 'application');
      this.canvas.setAttribute('aria-label', '3D object comparison. Drag to rotate, scroll or pinch to zoom. Arrow keys rotate; plus and minus zoom; Home resets.');
      Object.assign(this.canvas.style, { display: 'block', width: '100%', height: '100%', touchAction: 'none' });
      container.appendChild(this.canvas);
      this.controls = new OrbitControls(this.camera, this.canvas);
      this.controls.enableDamping = false;
      this.controls.enablePan = true;
      this.controls.rotateSpeed = 0.8;
      this.controls.zoomSpeed = 0.85;
      this.controls.addEventListener('change', () => {
        this.requestRender();
        this.onCameraChange(this.getState());
      });
      this.onKeyDown = event => this.handleKey(event);
      this.canvas.addEventListener('keydown', this.onKeyDown);
      this.onContextLost = event => {
        event.preventDefault();
        this.contextLost = true;
        this.errors.webgl = '3D rendering context lost';
        this.emit();
      };
      this.onContextRestored = () => {
        this.contextLost = false;
        delete this.errors.webgl;
        if (this.reloadOnRestore && this.pairRequest) {
          this.reloadOnRestore = false;
          this.setPair(this.pairRequest);
          return;
        }
        this.requestRender();
        this.emit();
      };
      this.canvas.addEventListener('webglcontextlost', this.onContextLost);
      this.canvas.addEventListener('webglcontextrestored', this.onContextRestored);
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(container);
      this.resize();
    } catch {
      this.errors.webgl = '3D rendering unavailable';
    }
    queueMicrotask(() => this.emit());
  }

  getState() {
    const referenceDiagonalMetres = nonNegative(this.deltaMeta.reference_diagonal_m);
    const error = this.errors.webgl || this.errors.reference || this.errors.candidate
      || (this.mode === 'delta' ? this.errors.delta : null) || null;
    return {
      ready: !this.loading && !this.errors.webgl && Boolean(this.reference || this.candidate),
      referenceReady: Boolean(this.reference),
      candidateReady: Boolean(this.candidate),
      deltaReady: Boolean(this.deltaPoints),
      deltaAvailable: Boolean(this.deltaUrl && this.reference && this.candidate),
      loading: this.loading,
      deltaLoading: this.deltaLoading,
      mode: this.mode,
      split: this.split,
      thresholdMetres: this.thresholdMetres,
      referenceSrc: this.referenceSrc,
      candidateSrc: this.candidateSrc,
      deltaSrc: this.deltaSrc,
      cameraPosition: this.camera.position.toArray(),
      target: this.controls?.target.toArray() || [0, 0, 0],
      pointCount: this.pointCount,
      visiblePointCount: this.visiblePointCount,
      referencePointCount: nonNegative(this.deltaMeta.reference_point_count),
      referenceDiagonalMetres,
      colourMaxMetres: referenceDiagonalMetres * 0.05 || this.maxDistanceMetres || 0.001,
      maxDistanceMetres: this.maxDistanceMetres,
      p95DistanceMetres: nonNegative(this.deltaMeta.p95_distance_m),
      bounds: this.bounds.isEmpty() ? null : { min: this.bounds.min.toArray(), max: this.bounds.max.toArray() },
      viewport: { width: this.width, height: this.height },
      error,
      errors: { ...this.errors },
    };
  }

  emit() {
    if (!this.disposed) this.onState(this.getState());
  }

  async setPair({ referenceUrl, candidateUrl = null, deltaUrl = null, deltaMeta = {} }) {
    if (this.disposed) return this.getState();
    this.pairRequest = { referenceUrl, candidateUrl, deltaUrl, deltaMeta };
    const generation = ++this.generation;
    this.controller?.abort();
    this.controller = new AbortController();
    const signal = this.controller.signal;
    this.clearPair();
    this.errors = this.errors.webgl ? { webgl: this.errors.webgl } : {};
    this.deltaUrl = deltaUrl;
    this.deltaMeta = { ...deltaMeta };
    this.maxDistanceMetres = nonNegative(deltaMeta.max_distance_m);
    this.thresholdMetres = 0;
    this.loading = true;
    this.emit();
    this.requestRender();
    if (!this.renderer || this.errors.webgl) {
      this.reloadOnRestore = true;
      this.loading = false;
      this.emit();
      return this.getState();
    }

    const load = async (kind, url) => {
      if (!url) return;
      try {
        const response = await fetch(url, { signal });
        if (!response.ok) throw new Error('Mesh unavailable');
        const buffer = await response.arrayBuffer();
        if (generation !== this.generation || signal.aborted) return;
        const gltf = await this.loader.parseAsync(buffer, new URL('.', new URL(url, document.baseURI)).href);
        const model = gltf.scene;
        if (this.disposed || generation !== this.generation || signal.aborted) {
          disposeObject(model);
          return;
        }
        model.updateMatrixWorld(true);
        const bounds = new Box3().setFromObject(model);
        if (bounds.isEmpty() || !bounds.min.toArray().concat(bounds.max.toArray()).every(Number.isFinite)) {
          disposeObject(model);
          throw new Error('Mesh bounds unavailable');
        }
        neutraliseMaterials(model);
        this[kind] = model;
        this[`${kind}Src`] = url;
        this[`${kind}Scene`].add(model);
      } catch (error) {
        if (generation !== this.generation || signal.aborted || this.disposed) return;
        this.errors[kind] = kind === 'reference' ? 'Reference preview unavailable' : 'Reconstruction preview unavailable';
      }
    };
    await Promise.allSettled([load('reference', referenceUrl), load('candidate', candidateUrl)]);
    if (this.disposed || generation !== this.generation || signal.aborted) return this.getState();
    this.loading = false;
    this.fitPair();
    this.emit();
    this.requestRender();
    if (this.mode === 'delta') await this.loadDelta();
    return this.getState();
  }

  clearPair() {
    disposeObject(this.reference);
    disposeObject(this.candidate);
    disposeObject(this.deltaPoints);
    this.reference = null;
    this.candidate = null;
    this.deltaPoints = null;
    this.referenceSrc = null;
    this.candidateSrc = null;
    this.deltaSrc = null;
    this.deltaDistances = null;
    this.deltaIndices = null;
    this.deltaPromise = null;
    this.deltaLoading = false;
    this.pointCount = 0;
    this.visiblePointCount = 0;
    this.bounds.makeEmpty();
    this.renderer?.renderLists.dispose();
  }

  fitPair() {
    if (!this.controls) return;
    this.bounds.makeEmpty();
    for (const model of [this.reference, this.candidate]) {
      if (model) this.bounds.union(new Box3().setFromObject(model));
    }
    if (this.bounds.isEmpty()) return;
    const centre = this.bounds.getCenter(new Vector3());
    const diagonal = Math.max(this.bounds.getSize(new Vector3()).length(), 0.0001);
    const radius = diagonal * 0.5;
    const verticalFov = this.camera.fov * Math.PI / 180;
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * this.camera.aspect);
    const distance = radius / Math.sin(Math.min(verticalFov, horizontalFov) / 2) * 1.12;
    this.camera.position.copy(new Vector3(1, 0.58, 1).normalize().multiplyScalar(distance).add(centre));
    this.camera.near = Math.max(diagonal * 0.00001, 0.0000001);
    this.camera.far = Math.max(diagonal * 100, distance * 20);
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(centre);
    this.controls.minDistance = radius * 0.08;
    this.controls.maxDistance = distance * 10;
    this.controls.update();
    this.controls.saveState();
  }

  setMode(mode) {
    if (mode !== 'swipe' && mode !== 'delta') throw new RangeError('Unknown comparison mode');
    this.mode = mode;
    this.emit();
    this.requestRender();
    if (mode === 'delta') return this.loadDelta();
    return Promise.resolve(this.getState());
  }

  setSplit(fraction) {
    if (!Number.isFinite(fraction)) return;
    this.split = Math.max(0, Math.min(1, fraction));
    this.requestRender();
    this.emit();
  }

  setThreshold(distanceMetres) {
    this.thresholdMetres = nonNegative(distanceMetres);
    this.applyThreshold();
    this.requestRender();
    this.emit();
  }

  async loadDelta() {
    if (this.disposed || this.loading || !this.deltaUrl || !this.reference || !this.candidate) return this.getState();
    if (this.deltaPoints) return this.getState();
    if (this.deltaPromise) return this.deltaPromise;
    const generation = this.generation;
    const signal = this.controller.signal;
    const deltaUrl = this.deltaUrl;
    this.deltaLoading = true;
    delete this.errors.delta;
    this.emit();
    this.deltaPromise = (async () => {
      try {
        const response = await fetch(deltaUrl, { signal });
        if (!response.ok) throw new Error('Delta unavailable');
        const buffer = await response.arrayBuffer();
        if (this.disposed || generation !== this.generation || signal.aborted) return this.getState();
        if (buffer.byteLength % 16 !== 0) throw new Error('Invalid delta buffer');
        const count = buffer.byteLength / 16;
        if (count === 0 || count !== this.deltaMeta.point_count) throw new Error('Invalid delta count');
        const rows = new DataView(buffer);
        const positions = new Float32Array(count * 3);
        const colours = new Float32Array(count * 3);
        const distances = new Float32Array(count);
        const indices = new Uint32Array(count);
        const scale = this.getState().colourMaxMetres;
        const colour = new Color();
        let maxDistance = 0;
        for (let i = 0; i < count; i++) {
          const x = rows.getFloat32(i * 16, true);
          const y = rows.getFloat32(i * 16 + 4, true);
          const z = rows.getFloat32(i * 16 + 8, true);
          const distance = rows.getFloat32(i * 16 + 12, true);
          if (![x, y, z, distance].every(Number.isFinite) || distance < 0) throw new Error('Invalid delta point');
          positions.set([x, y, z], i * 3);
          distances[i] = distance;
          maxDistance = Math.max(maxDistance, distance);
          indices[i] = i;
          const ramp = Math.min(1, distance / scale) * (COLOUR_STOPS.length - 1);
          const segment = Math.min(COLOUR_STOPS.length - 2, Math.floor(ramp));
          colour.copy(COLOUR_STOPS[segment]).lerp(COLOUR_STOPS[segment + 1], ramp - segment);
          colours.set([colour.r, colour.g, colour.b], i * 3);
        }
        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new BufferAttribute(positions, 3));
        geometry.setAttribute('color', new BufferAttribute(colours, 3));
        geometry.setIndex(new BufferAttribute(indices, 1));
        geometry.computeBoundingSphere();
        const material = new PointsMaterial({ vertexColors: true, size: 2.5, sizeAttenuation: false });
        this.deltaPoints = new Points(geometry, material);
        this.deltaScene.add(this.deltaPoints);
        this.deltaDistances = distances;
        this.deltaIndices = indices;
        this.pointCount = count;
        this.maxDistanceMetres = maxDistance;
        this.deltaSrc = deltaUrl;
        this.applyThreshold();
      } catch {
        if (!this.disposed && generation === this.generation && !signal.aborted) {
          this.errors.delta = 'Mesh delta unavailable';
        }
      } finally {
        if (!this.disposed && generation === this.generation && !signal.aborted) {
          this.deltaLoading = false;
          this.deltaPromise = null;
          this.requestRender();
          this.emit();
        }
      }
      return this.getState();
    })();
    return this.deltaPromise;
  }

  applyThreshold() {
    if (!this.deltaPoints) return;
    let visible = 0;
    for (let i = 0; i < this.pointCount; i++) {
      if (this.deltaDistances[i] >= this.thresholdMetres) this.deltaIndices[visible++] = i;
    }
    this.deltaPoints.geometry.index.needsUpdate = true;
    this.deltaPoints.geometry.setDrawRange(0, visible);
    this.visiblePointCount = visible;
  }

  reset() {
    this.fitPair();
    this.requestRender();
    this.emit();
  }

  handleKey(event) {
    if (!this.controls || event.altKey || event.ctrlKey || event.metaKey) return;
    const spherical = new Spherical().setFromVector3(this.camera.position.clone().sub(this.controls.target));
    const step = Math.PI / 36;
    switch (event.key) {
      case 'ArrowLeft': spherical.theta -= step; break;
      case 'ArrowRight': spherical.theta += step; break;
      case 'ArrowUp': spherical.phi -= step; break;
      case 'ArrowDown': spherical.phi += step; break;
      case '+': case '=': spherical.radius *= 0.9; break;
      case '-': case '_': spherical.radius /= 0.9; break;
      case 'Home': event.preventDefault(); this.reset(); return;
      default: return;
    }
    event.preventDefault();
    spherical.makeSafe();
    spherical.radius = Math.max(this.controls.minDistance, Math.min(this.controls.maxDistance, spherical.radius));
    this.camera.position.copy(new Vector3().setFromSpherical(spherical).add(this.controls.target));
    this.controls.update();
    this.requestRender();
  }

  resize() {
    if (this.disposed || !this.renderer) return;
    const rect = this.container.getBoundingClientRect();
    this.width = Math.max(1, Math.round(rect.width));
    this.height = Math.max(1, Math.round(rect.height));
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(this.width, this.height, false);
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  requestRender() {
    if (this.disposed || !this.renderer || this.frame !== null) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      this.render();
    });
  }

  render() {
    if (this.disposed || this.contextLost || !this.renderer) return;
    const renderer = this.renderer;
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, this.width, this.height);
    renderer.clear();
    if (this.mode === 'delta' && this.deltaPoints) {
      renderer.render(this.deltaScene, this.camera);
      return;
    }
    if (!this.reference || !this.candidate) {
      renderer.render(this.reference ? this.referenceScene : this.candidateScene, this.camera);
      return;
    }
    // The full viewport is invariant: scissoring reveals each model through the same projection.
    const split = Math.round(this.width * this.split);
    renderer.setScissorTest(true);
    if (split > 0) {
      renderer.setScissor(0, 0, split, this.height);
      renderer.render(this.referenceScene, this.camera);
    }
    if (split < this.width) {
      renderer.setScissor(split, 0, this.width - split, this.height);
      renderer.clearDepth();
      renderer.render(this.candidateScene, this.camera);
    }
    renderer.setScissorTest(false);
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.generation++;
    this.controller?.abort();
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.resizeObserver?.disconnect();
    this.canvas?.removeEventListener('keydown', this.onKeyDown);
    this.canvas?.removeEventListener('webglcontextlost', this.onContextLost);
    this.canvas?.removeEventListener('webglcontextrestored', this.onContextRestored);
    this.controls?.dispose();
    this.clearPair();
    this.renderer?.dispose();
    this.renderer?.forceContextLoss();
    this.canvas?.remove();
  }
}
