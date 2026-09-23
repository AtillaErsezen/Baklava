export const DURATION = 24;
export const LAYER_COUNT = 9;
export const CHAPTER_STARTS = [0, 7.8, 15.2];
export const CHAPTER_POSES = [2.4, 10.5, 17.8];
export const TAU = Math.PI * 2;

// Quintic easing keeps both velocity and acceleration continuous at each join.
export function ease(value: number) {
  const t = Math.max(0, Math.min(1, value));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

export function motionAt(elapsed: number) {
  const time = ((elapsed % DURATION) + DURATION) % DURATION;
  const phase = (time / DURATION) * TAU;
  return {
    time,
    phase,
    gather: ease((time - 3.8) / 4.6) * (1 - ease((time - 20.6) / 3.4)),
    settle: ease((time - 12.3) / 4) * (1 - ease((time - 20.3) / 3.2)),
    rotation: 0.42 + Math.sin(phase) * 0.12,
    chapter: time < CHAPTER_STARTS[1] ? 0 : time < CHAPTER_STARTS[2] ? 1 : 2,
  };
}

// Seeded variation avoids visual changes on resize or when a scene is replayed.
export function random(seed: number) {
  const n = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return n - Math.floor(n);
}
