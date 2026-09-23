import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import {
  CHAPTER_POSES,
  CHAPTER_STARTS,
  DURATION,
  LAYER_COUNT,
  TAU,
  ease,
  motionAt,
  random,
} from "./motion";

export interface PromoScene {
  configure: (running: boolean, reducedMotion: boolean) => void;
  seek: (chapter: number) => void;
  dispose: () => void;
}
interface Callbacks {
  onChapter: (chapter: number) => void;
  onProgress: (progress: number) => void;
  onContextLost: () => void;
}
const BEAD_COUNT = 260;
const mix = THREE.MathUtils.lerp;

function noise(x: number, y: number) {
  const ix = Math.floor(x),
    iy = Math.floor(y);
  const fx = ease(x - ix),
    fy = ease(y - iy);
  return mix(
    mix(random(ix + iy * 311), random(ix + 1 + iy * 311), fx),
    mix(random(ix + (iy + 1) * 311), random(ix + 1 + (iy + 1) * 311), fx),
    fy,
  );
}
function stoneTexture() {
  const size = 512;
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const index = (y * size + x) * 4;
      const broad = noise(x / 100, y / 100);
      const veins = noise(x / 70 + broad * 4, y / 9 + broad * 2);
      const pores = Math.pow(noise(x / 2.8, y / 2.8), 9);
      const fine = random(x + y * size);
      const shade = broad * 21 + veins * 19 + fine * 7 - pores * 130;
      data[index] = 173 + shade;
      data[index + 1] = 151 + shade;
      data[index + 2] = 121 + shade;
      data[index + 3] = 255;
    }
  }
  const texture = new THREE.DataTexture(data, size, size);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.needsUpdate = true;
  return texture;
}

// Large studio light cards give the glass broad, photographic reflections.
function studioEnvironment(renderer: THREE.WebGLRenderer) {
  const studio = new THREE.Scene();
  studio.background = new THREE.Color(0x24180e);
  const cards: THREE.Mesh[] = [];
  function card(
    w: number,
    h: number,
    position: number[],
    intensity: number,
    color: number,
  ) {
    const material = new THREE.MeshBasicMaterial({
      color,
      side: THREE.DoubleSide,
    });
    material.color.multiplyScalar(intensity);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), material);
    mesh.position.set(position[0], position[1], position[2]);
    mesh.lookAt(0, 0, 0);
    studio.add(mesh);
    cards.push(mesh);
  }
  card(6, 3, [1, 5, 2], 5, 0xffefd5);
  card(1.2, 5, [-4, 1, 3], 2.4, 0xffd8aa);
  card(2, 4, [4, 2, -3], 7, 0xffbf6b);
  card(5, 1, [0, -2, 4], 0.55, 0xffeedc);
  card(7, 2, [1.5, 4, -6], 3.5, 0xffe5bc);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const result = pmrem.fromScene(studio, 0.025, 0.1, 40);
  cards.forEach((mesh) => {
    mesh.geometry.dispose();
    (mesh.material as THREE.Material).dispose();
  });
  pmrem.dispose();
  return result;
}
function softShadowTexture() {
  const surface = document.createElement("canvas");
  surface.width = surface.height = 128;
  const context = surface.getContext("2d")!;
  const gradient = context.createRadialGradient(64, 64, 12, 64, 64, 64);
  gradient.addColorStop(0, "rgba(0,0,0,.55)");
  gradient.addColorStop(0.45, "rgba(0,0,0,.32)");
  gradient.addColorStop(1, "rgba(0,0,0,0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(surface);
}

export function createPromoScene(
  canvas: HTMLCanvasElement,
  callbacks: Callbacks,
): PromoScene {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.transmissionResolutionScale = 0.65;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#21130d");
  scene.fog = new THREE.FogExp2("#21130d", 0.072);
  const camera = new THREE.PerspectiveCamera(34, 2, 0.1, 60);
  const environment = studioEnvironment(renderer);
  scene.environment = environment.texture;
  scene.environmentIntensity = 0.9;
  scene.add(new THREE.HemisphereLight(0xffe6c6, 0x24130b, 0.8));
  const key = new THREE.DirectionalLight(0xffe1b4, 3.3);
  key.position.set(2, 6, 3);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xffb857, 4);
  rim.position.set(3, 4, -4);
  scene.add(rim);
  const pool = new THREE.SpotLight(0xffbc6b, 55, 24, 0.63, 1, 1.6);
  pool.position.set(5, 5, -2);
  pool.target.position.set(1.9, -1.8, 0);
  scene.add(pool, pool.target);
  const stone = stoneTexture();
  const floorMap = stone.clone();
  floorMap.repeat.set(14, 14);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(100, 100),
    new THREE.MeshStandardMaterial({
      color: 0x635448,
      roughness: 0.48,
      metalness: 0.12,
      map: floorMap,
      bumpMap: floorMap,
      bumpScale: 0.032,
      envMapIntensity: 0.3,
    }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -1.78;
  scene.add(floor);
  const sculpture = new THREE.Group();
  scene.add(sculpture);
  const shadowTexture = softShadowTexture();
  const shadow = new THREE.Mesh(
    new THREE.PlaneGeometry(4.8, 4.8),
    new THREE.MeshBasicMaterial({
      map: shadowTexture,
      transparent: true,
      depthWrite: false,
    }),
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = -1.77;
  sculpture.add(shadow);
  // A lathed, beveled profile avoids the sharp-edged cylinder silhouette.
  const plinthProfile = [
    [0, 0],
    [1.34, 0],
    [1.405, 0.012],
    [1.43, 0.04],
    [1.43, 0.245],
    [1.42, 0.273],
    [1.39, 0.295],
    [1.32, 0.3],
    [0, 0.3],
  ].map(([x, y]) => new THREE.Vector2(x, y));
  const plinth = new THREE.Mesh(
    new THREE.LatheGeometry(plinthProfile, 128),
    new THREE.MeshStandardMaterial({
      color: 0xf0dfc7,
      map: stone,
      bumpMap: stone,
      bumpScale: 0.035,
      roughness: 0.77,
      metalness: 0,
      envMapIntensity: 0.25,
    }),
  );
  plinth.position.y = -1.74;
  sculpture.add(plinth);
  const stack = new THREE.Group();
  sculpture.add(stack);
  const layerGeometry = new RoundedBoxGeometry(1.9, 0.11, 1.9, 4, 0.042);
  const layers = Array.from({ length: LAYER_COUNT }, () => {
    const face = new THREE.MeshPhysicalMaterial({
      color: 0xffe7b3,
      metalness: 0,
      roughness: 0.06,
      transmission: 0.94,
      thickness: 0.18,
      attenuationColor: new THREE.Color(0xe6a13e),
      attenuationDistance: 1.2,
      ior: 1.48,
      clearcoat: 1,
      clearcoatRoughness: 0.075,
      envMapIntensity: 1.3,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });
    const edge = face.clone();
    edge.color.set(0xffd995);
    edge.transmission = 0.65;
    edge.roughness = 0.13;
    edge.envMapIntensity = 1.8;
    const layer = new THREE.Mesh(layerGeometry, [
      edge,
      edge,
      face,
      face,
      edge,
      edge,
    ]);
    stack.add(layer);
    return layer;
  });
  const beads = new THREE.InstancedMesh(
    new THREE.SphereGeometry(1, 24, 16),
    new THREE.MeshPhysicalMaterial({
      color: 0xf2c27d,
      metalness: 0.7,
      roughness: 0.13,
      clearcoat: 1,
      clearcoatRoughness: 0.08,
      envMapIntensity: 1.2,
    }),
    BEAD_COUNT,
  );
  beads.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  beads.frustumCulled = false;
  sculpture.add(beads);
  const transform = new THREE.Object3D();
  const beadData = Array.from({ length: BEAD_COUNT }, (_, i) => ({
    offset: i / BEAD_COUNT,
    lane: (random(i + 17) - 0.5) * 0.5,
    angle: random(i + 81) * 0.43,
    size:
      i % 19 === 0
        ? 0.083 + random(i) * 0.045
        : 0.016 + random(i + 100) ** 2 * 0.04,
  }));
  // Refracted light is a soft, broken network rather than concentric rings.
  const causticMaterial = new THREE.ShaderMaterial({
    uniforms: { phase: { value: 0 }, strength: { value: 0 } },
    vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
    fragmentShader: `varying vec2 vUv; uniform float phase; uniform float strength;
      vec2 hash(vec2 p){ return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453); }
      void main(){
        vec2 uv=(vUv-.5)*2.; vec2 p=uv*5.; p+=.3*sin(p.yx*2.+phase)+.12*sin(p.yx*4.3-phase); vec2 cell=floor(p); vec2 f=fract(p);
        float first=8.; float second=8.;
        for(int y=-1;y<=1;y++){ for(int x=-1;x<=1;x++){
          vec2 g=vec2(float(x),float(y)); vec2 h=hash(cell+g);
          vec2 delta=g+.5+.28*sin(phase+6.2831*h)-f;
          float d=dot(delta,delta);
          if(d<first){second=first;first=d;} else if(d<second){second=d;}
        }}
        float line=pow(1.-smoothstep(.0,.17,second-first),3.);
        float fade=pow(max(0.,1.-length(uv)),2.);
        float value=(line*.7+.025)*fade*strength;
        gl_FragColor=vec4(1.,.62,.25,value);
      }`,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const caustic = new THREE.Mesh(
    new THREE.PlaneGeometry(6, 6),
    causticMaterial,
  );
  caustic.rotation.x = -Math.PI / 2;
  caustic.position.y = -1.755;
  sculpture.add(caustic);
  // MSAA must live on the composer's target; renderer antialiasing alone does
  // not smooth edges when rendering through postprocessing.
  const target = new THREE.WebGLRenderTarget(1, 1, {
    type: THREE.HalfFloatType,
    samples: Math.min(4, renderer.capabilities.maxSamples),
  });
  const composer = new EffectComposer(renderer, target);
  const bloom = new UnrealBloomPass(
    new THREE.Vector2(800, 400),
    0.2,
    0.65,
    1.7,
  );
  const output = new OutputPass();
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(bloom);
  composer.addPass(output);
  let elapsed = CHAPTER_POSES[0];
  let running = false,
    reduced = false,
    disposed = false,
    unavailable = false;
  let frame = 0,
    lastTime = 0,
    currentChapter = -1;
  let narrow = false,
    width = 0,
    height = 0;
  let navigation: { from: number; to: number; progress: number } | null = null;

  function draw() {
    const { time, phase, gather, settle, rotation, chapter } =
      motionAt(elapsed);
    const gap = mix(0.32, 0.155, settle);
    stack.rotation.y = rotation;
    for (let i = 0; i < LAYER_COUNT; i++) {
      const layer = layers[i];
      const reveal = ease((gather - i * 0.018) / (1 - i * 0.018));
      layer.position.y =
        -1.35 + i * gap + (1 - settle) * Math.sin(phase + i * 0.33) * 0.035;
      layer.rotation.y = (1 - settle) * (i - 4) * 0.035;
      layer.rotation.z = (1 - settle) * Math.sin(phase + i * 0.25) * 0.009;
      layer.scale.set(mix(0.65, 1, reveal), 1, mix(0.65, 1, reveal));
      layer.material[0].opacity = reveal * 0.9;
      layer.material[2].opacity = reveal * 0.43;
      layer.visible = reveal > 0.002;
    }
    beads.visible = gather < 0.999;
    if (beads.visible) {
      beadData.forEach((bead, i) => {
        const h = (bead.offset + time / DURATION) % 1;
        const angle = h * TAU * 2.15 + phase + bead.angle + gather * 1.2;
        const radius = mix(1.85 - h * 0.72 + bead.lane, 0.7, gather);
        const envelope = ease(h / 0.1) * ease((1 - h) / 0.13);
        const y = mix(
          -1.1 + h * 3.6,
          -1.35 + h * (LAYER_COUNT - 1) * gap,
          gather,
        );
        transform.position.set(
          Math.cos(angle) * radius,
          y,
          Math.sin(angle) * radius * 0.76,
        );
        transform.scale.setScalar(
          Math.max(0.00001, bead.size * envelope * (1 - ease(gather))),
        );
        transform.updateMatrix();
        beads.setMatrixAt(i, transform.matrix);
      });
      beads.instanceMatrix.needsUpdate = true;
    }
    causticMaterial.uniforms.phase.value = phase;
    causticMaterial.uniforms.strength.value = 0.12 + gather * 0.38;
    sculpture.position.set(narrow ? 0 : 1.9, narrow ? -0.2 : 0, 0);
    sculpture.scale.setScalar(narrow ? 0.86 : 1);
    pool.position.x = sculpture.position.x + 3.1;
    pool.target.position.x = sculpture.position.x;
    // One periodic orbit: the last frame meets the first in position and velocity.
    camera.position.set(
      0.15 + Math.sin(phase) * 0.1,
      narrow ? 2.6 : 2.5,
      narrow ? 9.7 : 8.8,
    );
    camera.lookAt(narrow ? 0 : 0.18, narrow ? 0.75 : 0.2, 0);
    if (chapter !== currentChapter) {
      currentChapter = chapter;
      callbacks.onChapter(chapter);
    }
    const end = CHAPTER_STARTS[chapter + 1] ?? DURATION;
    callbacks.onProgress(
      (time - CHAPTER_STARTS[chapter]) / (end - CHAPTER_STARTS[chapter]),
    );
    composer.render();
  }
  function tick(now: number) {
    if (disposed || unavailable || reduced || (!running && !navigation)) return;
    const delta = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;
    if (navigation) {
      navigation.progress = Math.min(1, navigation.progress + delta / 1.65);
      elapsed = mix(navigation.from, navigation.to, ease(navigation.progress));
      if (navigation.progress === 1) navigation = null;
    } else {
      elapsed += delta;
    }
    draw();
    // Render at the display's cadence, without the old 45 Hz skipping pattern.
    if (running || navigation) frame = requestAnimationFrame(tick);
  }
  function start() {
    cancelAnimationFrame(frame);
    lastTime = performance.now();
    if (!reduced && (running || navigation))
      frame = requestAnimationFrame(tick);
  }
  function resize() {
    const bounds = canvas.getBoundingClientRect();
    if (
      !bounds.width ||
      !bounds.height ||
      (width === bounds.width && height === bounds.height)
    )
      return;
    width = bounds.width;
    height = bounds.height;
    narrow = width < 600;
    camera.aspect = width / height;
    camera.fov = narrow ? 42 : 34;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    composer.setSize(width, height);
    draw();
  }
  const observer = new ResizeObserver(resize);
  observer.observe(canvas);
  resize();
  function contextLost(event: Event) {
    event.preventDefault();
    unavailable = true;
    running = false;
    cancelAnimationFrame(frame);
    callbacks.onContextLost();
  }
  canvas.addEventListener("webglcontextlost", contextLost);
  return {
    configure(active, reducedMotion) {
      if (disposed || unavailable) return;
      const changed = reduced !== reducedMotion;
      reduced = reducedMotion;
      running = active;
      if (!active) navigation = null;
      if (changed && reduced) {
        elapsed = CHAPTER_POSES[2];
        navigation = null;
        draw();
      }
      start();
    },
    seek(chapter) {
      if (disposed || unavailable) return;
      const to = CHAPTER_POSES[chapter] ?? CHAPTER_POSES[0];
      if (reduced) {
        elapsed = to;
        draw();
      } else {
        const from = ((elapsed % DURATION) + DURATION) % DURATION;
        let distance = to - from;
        if (distance > DURATION / 2) distance -= DURATION;
        if (distance < -DURATION / 2) distance += DURATION;
        navigation = { from, to: from + distance, progress: 0 };
        start();
      }
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      canvas.removeEventListener("webglcontextlost", contextLost);
      const geometries = new Set<THREE.BufferGeometry>();
      const materials = new Set<THREE.Material>();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          geometries.add(object.geometry);
          (Array.isArray(object.material)
            ? object.material
            : [object.material]
          ).forEach((material) => materials.add(material));
        }
      });
      geometries.forEach((geometry) => geometry.dispose());
      materials.forEach((material) => material.dispose());
      stone.dispose();
      floorMap.dispose();
      shadowTexture.dispose();
      environment.dispose();
      bloom.dispose();
      output.dispose();
      composer.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
}
