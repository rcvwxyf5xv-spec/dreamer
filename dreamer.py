#!/usr/bin/env python3
"""
Laya-dreamed flow field.

A dreamer: particles drift through a continuously
evolving curl-noise vector field, the way smoke or a dream-state visual
would move, rather than converging onto a static geometric attractor (which
is what made this look like a generic macOS/screensaver wallpaper).

Laya directly picks the dream's *motif* -- a real, visually distinct
archetype (spiral / waves / tendrils / pulse / fracture), not just a slider
nudge -- plus the dream's intensity, coherence (lucid vs chaotic), recall
(how long trails linger, like memory), and the odds of an associative leap:
a sudden, mid-stream shift to a new vision, the way dreams jump topic
without warning. Between Laya's ~1/sec decisions the field keeps flowing
and drifting on its own via continuously advancing phases, so it is never
static and never repeats.

Memory stays flat forever: a fixed number of particles persist across
frames (no history, no per-frame growth) inside a fixed toroidal domain
(positions wrap, they are never discarded or reallocated), and the pixel
accumulation buffer is a single fixed-size array that decays each frame
(the "recall" rate) rather than accumulating without bound.
"""
import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

# If numpy isn't importable (e.g. the script was launched with the system
# python instead of the project venv), re-exec with the venv interpreter so
# `python dreamer.py` just works no matter how it's invoked.
try:
    import numpy as np
except ImportError:
    venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python3"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)
    raise

LAYA_MODEL = "aac6fef/laya-mlx"


def load_laya_agent():
    """Load the Laya model, skipping the network entirely if already cached.

    huggingface_hub caches the model under ~/.cache/huggingface/hub once
    downloaded; it is never re-downloaded on later runs. By default it still
    does a quick local cache-validity check on every load (harmless, <0.5s,
    no re-download) -- if a cached snapshot is present we set HF_HUB_OFFLINE
    so it skips even that and never depends on network access at all.
    """
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_cache_dir = cache_root / f"models--{LAYA_MODEL.replace('/', '--')}"
    if model_cache_dir.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import laya_mlx as laya
    return laya.load(LAYA_MODEL)

MOTIFS = {
    "spiral": "a swirling vortex, everything pulled into a slow rotating whirlpool",
    "waves": "gentle flowing waves drifting steadily across the whole vision",
    "tendrils": "delicate branching filaments reaching and curling outward",
    "pulse": "concentric pulses radiating outward and inward like a heartbeat",
    "fracture": "a fractured, turbulent vision pulling in conflicting directions",
}

DREAM_QUESTIONS = {
    "motif": {
        "type": "choice",
        "instructions": "What is this dream a vision of right now?",
        "criteria": MOTIFS,
    },
    "intensity": {
        "type": "score",
        "instructions": "How intensely is this dream moving right now?",
        "criteria": ["still", "drifting", "flowing", "surging", "overwhelming"],
    },
    "coherence": {
        "type": "score",
        "instructions": "Is this a lucid, coherent dream or a hazy, fractured one?",
        "criteria": ["fractured", "hazy", "dreamlike", "vivid", "lucid"],
    },
    "palette": {
        "type": "choice",
        "instructions": "What color does this dream feel like right now?",
        "criteria": ["fiery", "oceanic", "electric", "organic", "void"],
    },
    "recall": {
        "type": "noul",
        "instructions": "Should this dream linger, leaving long fading trails like a strong memory, rather than passing by quickly?",
    },
    "leap": {
        "type": "noul",
        "instructions": "Should the dream suddenly leap to an entirely new vision right now, the way dreams jump topic without warning?",
    },
}

PALETTES = {
    "fiery":   [(10, 0, 0), (200, 30, 0), (255, 140, 0), (255, 230, 120)],
    "oceanic": [(0, 5, 20), (0, 60, 120), (0, 160, 200), (150, 240, 255)],
    "electric":[(5, 0, 20), (100, 0, 200), (180, 60, 255), (230, 200, 255)],
    "organic": [(2, 10, 2), (20, 90, 30), (120, 200, 60), (230, 255, 180)],
    "void":    [(0, 0, 0), (30, 30, 45), (70, 70, 110), (180, 180, 220)],
}
PALETTE_ARRAYS = {name: np.array(stops, dtype=np.float64) for name, stops in PALETTES.items()}
PALETTE_FADE_RATE = 1.0 / 180.0  # ~6s crossfade at 30fps


def palette_lookup(stops, t):
    n = len(stops)
    t = np.clip(t, 0.0, 0.999) * (n - 1)
    i0 = t.astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = (t - i0)[..., None]
    c0 = stops[i0]
    c1 = stops[i1]
    return c0 * (1 - frac) + c1 * frac


def smoothstep(t):
    return t * t * (3 - 2 * t)


N_LAYERS = 5
DOMAIN = 3.0
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))


def build_layer_targets(motif, coherence, intensity, rng):
    """Roll a fresh set of N_LAYERS wave-layer parameters for the given
    dream motif. Each motif is a genuinely different field archetype (not
    just a parameter nudge): they differ in projection mode (linear vs
    radial), spatial frequency range, and how rotational vs translational
    the flow is -- so choosing a motif visibly changes what kind of thing
    is on screen, the way choosing what to dream about would."""
    base_angle = rng.uniform(0, 2 * math.pi)
    spread = 0.25 + (1.0 - coherence) * 1.6

    freq_range = {
        "spiral": (1.5, 3.5), "waves": (0.8, 2.2), "tendrils": (5.0, 11.0),
        "pulse": (2.5, 5.5), "fracture": (4.0, 13.0),
    }[motif]
    curl_center = {
        "spiral": 0.9, "waves": 0.05, "tendrils": 0.6, "pulse": 0.35, "fracture": 0.0,
    }[motif]
    curl_spread = {
        "spiral": 0.15, "waves": 0.15, "tendrils": 0.3, "pulse": 0.4, "fracture": 1.2,
    }[motif]
    radial = motif == "pulse"

    layers = []
    for i in range(N_LAYERS):
        angle = base_angle + i * GOLDEN_ANGLE * spread + rng.uniform(-0.3, 0.3) * (1.0 - coherence)
        freq = rng.uniform(*freq_range)
        sign = 1.0 if rng.random() > 0.5 else -1.0
        curl = float(np.clip(curl_center * sign if motif == "fracture" else curl_center
                              + rng.uniform(-curl_spread, curl_spread), -1.0, 1.0))
        amp = (0.5 + rng.uniform(-0.15, 0.15)) * (0.5 + intensity)
        phase_speed = rng.uniform(0.15, 0.5) * (1.0 if rng.random() > 0.5 else -1.0) * (0.4 + intensity)
        layers.append(dict(angle=angle, freq=freq, curl=curl, amp=amp,
                            phase_speed=phase_speed, radial=radial))
    return layers


class Layer:
    """One flow-field wave component. Shape parameters continuously ease
    toward Laya-set targets while phase advances every frame forever, so
    the field is always moving even between decisions."""

    __slots__ = ("angle", "freq", "curl", "amp", "phase", "phase_speed", "radial",
                 "target_angle", "target_freq", "target_curl", "target_amp", "target_radial")

    def __init__(self, params, rng):
        self.phase = rng.uniform(0, 2 * math.pi)
        self.apply_target(params)
        self.angle, self.freq, self.curl, self.amp = (
            self.target_angle, self.target_freq, self.target_curl, self.target_amp)
        self.phase_speed = params["phase_speed"]
        self.radial = params["radial"]

    def apply_target(self, params):
        self.target_angle = params["angle"]
        self.target_freq = params["freq"]
        self.target_curl = params["curl"]
        self.target_amp = params["amp"]
        self.target_radial = params["radial"]
        self.phase_speed = params["phase_speed"]

    def ease(self, rate, dt):
        self.angle += (self.target_angle - self.angle) * rate
        self.freq += (self.target_freq - self.freq) * rate
        self.curl += (self.target_curl - self.curl) * rate
        self.amp += (self.target_amp - self.amp) * rate
        self.radial = self.target_radial
        self.phase += self.phase_speed * dt


N_POINTS = 30000


class DreamField:
    """Persistent, fixed-memory particle flow field. Laya picks the dream's
    motif/intensity/coherence/palette/recall/leap; the field itself flows
    and drifts continuously between those decisions and never repeats."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.px = self.rng.uniform(-DOMAIN, DOMAIN, N_POINTS)
        self.py = self.rng.uniform(-DOMAIN, DOMAIN, N_POINTS)
        self.pc = self.rng.uniform(0, 1, N_POINTS)

        self.motif = "waves"
        self.intensity = 0.3
        self.coherence = 0.6
        self.target_intensity = self.intensity
        self.target_coherence = self.coherence
        self.target_recall = 0.5
        self.recall = 0.5

        layer_params = build_layer_targets(self.motif, self.coherence, self.intensity, self.rng)
        self.layers = [Layer(p, self.rng) for p in layer_params]

        self.palette_name = "electric"
        self.palette_from = PALETTE_ARRAYS["electric"]
        self.palette_to = PALETTE_ARRAYS["electric"]
        self.palette_blend = 1.0

        self.buffer_shape = None
        self.density = None
        self.color_sum = None
        self.frame = 0
        self._lock = threading.Lock()

    def current_palette_stops(self):
        return self.palette_from * (1 - self.palette_blend) + self.palette_to * self.palette_blend

    def set_palette_target(self, name):
        if name == self.palette_name:
            return
        self.palette_from = self.current_palette_stops()
        self.palette_to = PALETTE_ARRAYS[name]
        self.palette_blend = 0.0
        self.palette_name = name

    def describe_state(self):
        # Deliberately do not restate the current motif/palette here: feeding
        # a choice back as "current state" anchors Laya into reconfirming it
        # with ever-growing confidence every cycle, so it never changes again.
        return (
            f"A dream flowing with intensity {self.intensity:.2f} and coherence {self.coherence:.2f}, "
            f"at moment {self.frame} of an unbroken, ongoing dream. "
            f"It is time for the dream to continue evolving or shift."
        )

    def mutate_from_laya(self, answers):
        # Only set *targets* here. step() continuously eases live values
        # toward them, so a new Laya decision redirects the ongoing drift
        # instead of ever cutting to a new frame instantly.
        with self._lock:
            new_motif = self._sample_choice(answers["motif"]["probabilities"])
            self.target_intensity = answers["intensity"]["score"] / 4.0
            self.target_coherence = answers["coherence"]["score"] / 4.0
            self.target_recall = answers["recall"]["noul"]
            leap_prob = answers["leap"]["noul"]
            self.set_palette_target(self._sample_choice(answers["palette"]["probabilities"]))

            motif_changed = new_motif != self.motif
            leaping = motif_changed or (self.rng.random() < leap_prob * 0.6)

            if leaping:
                self.motif = new_motif
                targets = build_layer_targets(self.motif, self.target_coherence,
                                               self.target_intensity, self.rng)
                for layer, params in zip(self.layers, targets):
                    layer.apply_target(params)
            else:
                # Still dreaming the same vision: gently re-roll each layer's
                # target within the current motif so detail keeps shifting
                # even when the overall theme hasn't changed.
                targets = build_layer_targets(self.motif, self.target_coherence,
                                               self.target_intensity, self.rng)
                for layer, params in zip(self.layers, targets):
                    layer.target_angle = layer.angle + self.rng.uniform(-0.6, 0.6)
                    layer.target_freq = params["freq"]
                    layer.target_curl = (layer.curl + params["curl"]) / 2
                    layer.target_amp = params["amp"]
                    layer.target_radial = params["radial"]
                    layer.phase_speed = params["phase_speed"]

    def _sample_choice(self, probabilities):
        # Sample proportionally to Laya's own probabilities instead of always
        # taking the argmax, so a real-but-non-dominant preference (e.g. 20%)
        # still gets a genuine chance to land instead of being permanently
        # outvoted by whatever choice happened to win first.
        options = list(probabilities.keys())
        weights = np.array(list(probabilities.values()), dtype=np.float64)
        weights = weights / weights.sum()
        return self.rng.choice(options, p=weights)

    def step(self, n_iters=1):
        with self._lock:
            meta_rate = 0.04
            self.intensity += (self.target_intensity - self.intensity) * meta_rate
            self.coherence += (self.target_coherence - self.coherence) * meta_rate
            self.recall += (self.target_recall - self.recall) * meta_rate
            self.palette_blend = min(1.0, self.palette_blend + PALETTE_FADE_RATE)

            shape_rate = 0.02 + 0.05 * self.intensity
            dt = 0.02 * (0.4 + self.intensity)

            for _ in range(n_iters):
                for layer in self.layers:
                    layer.ease(shape_rate, dt)

                # Every term summed into vx/vy below is a tangential
                # (rotational) field around some direction -- mathematically
                # divergence-free, so particle density can neither pile up
                # into sinks nor thin out into voids (Liouville: flow along
                # a divergence-free field preserves area). Just as important
                # on a *wrapped* domain: the wavevector (nx, ny) is rounded
                # to integers, which forces the wave to complete a whole
                # number of cycles exactly across the domain width -- so its
                # value/slope match up perfectly at the wrap seam. A
                # non-integer wavevector (an earlier version of this used a
                # raw continuous angle/freq, and a hypot()-based radial term
                # that isn't even wrap-periodic) is discontinuous right at
                # the seam, and that seam acts like a comb that herds nearly
                # every particle onto it over time -- exactly the "converges
                # to almost nothing happening" symptom.
                vx = np.zeros(N_POINTS)
                vy = np.zeros(N_POINTS)
                for layer in self.layers:
                    nx = round(layer.freq * math.cos(layer.angle))
                    ny = round(layer.freq * math.sin(layer.angle))
                    if nx == 0 and ny == 0:
                        nx = 1
                    norm = math.hypot(nx, ny)
                    dirx, diry = nx / norm, ny / norm
                    k = math.pi / DOMAIN
                    proj = (self.px * nx + self.py * ny) * k
                    wave = np.sin(proj + layer.phase)
                    sign = 1.0 if layer.curl >= 0 else -1.0
                    vx += sign * layer.amp * (-diry * wave)
                    vy += sign * layer.amp * (dirx * wave)

                self.px += vx * dt
                self.py += vy * dt
                # Toroidal wrap keeps the domain -- and memory footprint --
                # fixed forever; particles never leave and are never
                # discarded/reallocated, they just drift back around.
                self.px = ((self.px + DOMAIN) % (2 * DOMAIN)) - DOMAIN
                self.py = ((self.py + DOMAIN) % (2 * DOMAIN)) - DOMAIN

                drift = self.rng.uniform(-0.004, 0.004, N_POINTS)
                self.pc = (self.pc + drift) % 1.0

            self.frame += 1

    def _ensure_buffers(self, width, height):
        if self.buffer_shape != (height, width):
            self.buffer_shape = (height, width)
            self.density = np.zeros((height, width), dtype=np.float64)
            self.color_sum = np.zeros((height, width), dtype=np.float64)

    def render(self, width, height):
        self._ensure_buffers(width, height)

        sx = (self.px + DOMAIN) / (2 * DOMAIN) * width
        sy = (self.py + DOMAIN) / (2 * DOMAIN) * height
        ix = np.clip(sx.astype(np.int64), 0, width - 1)
        iy = np.clip(sy.astype(np.int64), 0, height - 1)
        c = self.pc

        # "Recall" controls how long trails linger -- a strong memory fades
        # slowly, a fleeting one fades fast -- but always via decay of a
        # fixed-size buffer, never by accumulating unbounded history.
        fade = 0.80 + 0.19 * self.recall
        self.density *= fade
        self.color_sum *= fade
        np.add.at(self.density, (iy, ix), 1.0)
        np.add.at(self.color_sum, (iy, ix), c)

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_hue = np.where(self.density > 1e-9, self.color_sum / np.maximum(self.density, 1e-9), 0.0)

        brightness = np.log1p(self.density) / math.log1p(max(self.density.max(), 1.0))
        return brightness, avg_hue


class LayaBrain:
    def __init__(self):
        self.agent = load_laya_agent()
        self.busy = False

    def query_async(self, system, on_done):
        def work():
            try:
                state = system.describe_state()
                result = self.agent.predict(state, DREAM_QUESTIONS)
                on_done(result["answers"])
            finally:
                self.busy = False
        self.busy = True
        threading.Thread(target=work, daemon=True).start()


def render_rgb(system, width, height):
    brightness, hue = system.render(width, height)
    rgb = palette_lookup(system.current_palette_stops(), hue)
    rgb = rgb * brightness[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


ASCII_RAMP = " .:-=+*#%@"


def render_ascii(system, cols, rows):
    brightness, hue = system.render(cols, rows)
    idx = np.clip((brightness * (len(ASCII_RAMP) - 1)), 0, len(ASCII_RAMP) - 1).astype(int)
    return idx, hue, brightness


def run_gui(size, fps_cap, borderless=False):
    import pygame
    from pygame._sdl2.video import Window

    pygame.init()
    flags = pygame.RESIZABLE | (pygame.NOFRAME if borderless else 0)
    screen = pygame.display.set_mode((size, size), flags)
    pygame.display.set_caption("Laya Dream Field")
    window = Window.from_display_module()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    print("Loading Laya model...")
    brain = LayaBrain()
    system = DreamField()
    print("Model loaded. Rendering...")

    query_interval = 1.0
    last_query = 0.0

    dragging = False
    drag_zone_frac = 1.0 / 20.0
    show_text = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                flags = pygame.RESIZABLE | (pygame.NOFRAME if window.borderless else 0)
                screen = pygame.display.set_mode((event.w, event.h), flags)
                window = Window.from_display_module()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_b:
                    window.borderless = not window.borderless
                elif event.key == pygame.K_t:
                    show_text = not show_text
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                w, h = screen.get_size()
                if window.borderless and event.pos[1] <= h * drag_zone_frac:
                    dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                wx, wy = window.position
                window.position = (wx + event.rel[0], wy + event.rel[1])

        w, h = screen.get_size()

        system.step(n_iters=3)
        rgb = render_rgb(system, w, h)

        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        if show_text:
            info = font.render(
                f"motif: {system.motif}  intensity: {system.intensity:.2f}  coherence: {system.coherence:.2f}  "
                f"palette: {system.palette_name}  [b] titlebar  [t] hide text",
                True, (255, 255, 255),
            )
            screen.blit(info, (5, 5))
        pygame.display.flip()

        now = time.time()
        if not brain.busy and now - last_query > query_interval:
            last_query = now
            brain.query_async(system, system.mutate_from_laya)

        del rgb
        clock.tick(fps_cap)

    pygame.quit()


def run_terminal(fps_cap):
    import curses

    def main(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(int(1000 / fps_cap))

        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            for i, cid in enumerate([curses.COLOR_RED, curses.COLOR_BLUE, curses.COLOR_MAGENTA,
                                      curses.COLOR_GREEN, curses.COLOR_WHITE], start=1):
                curses.init_pair(i, cid, -1)

        stdscr.addstr(0, 0, "Loading Laya model...")
        stdscr.refresh()

        brain = LayaBrain()
        system = DreamField()

        PALETTE_COLOR = {"fiery": 1, "oceanic": 2, "electric": 3, "organic": 4, "void": 5}
        query_interval = 1.0
        last_query = 0.0
        show_text = True

        while True:
            ch = stdscr.getch()
            if ch == ord("q") or ch == 27:
                break
            elif ch == ord("t"):
                show_text = not show_text

            h, w = stdscr.getmaxyx()
            rows, cols = (h - 2, w) if show_text else (h, w)
            if rows < 3 or cols < 3:
                continue

            system.step(n_iters=3)
            idx, hue, brightness = render_ascii(system, cols, rows)
            color_pair = PALETTE_COLOR.get(system.palette_name, 3)

            stdscr.erase()
            for row_idx in range(rows):
                line = "".join(ASCII_RAMP[v] for v in idx[row_idx])
                try:
                    stdscr.addnstr(row_idx, 0, line, w - 1, curses.color_pair(color_pair))
                except curses.error:
                    pass

            if show_text:
                status = (
                    f" {system.motif} intensity:{system.intensity:.2f} coherence:{system.coherence:.2f} "
                    f"palette:{system.palette_name} [q]uit [t] hide text "
                )
                try:
                    stdscr.addnstr(h - 1, 0, status, w - 1, curses.A_REVERSE)
                except curses.error:
                    pass
            stdscr.refresh()

            now = time.time()
            if not brain.busy and now - last_query > query_interval:
                last_query = now
                brain.query_async(system, system.mutate_from_laya)

            del idx, hue, brightness

    curses.wrapper(main)


def main():
    parser = argparse.ArgumentParser(
        description="Laya-dreamed flow field: a real-time evolving visual driven by live Laya model decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python dreamer.py                # GUI window\n"
            "  python dreamer.py --borderless   # GUI window with no titlebar (drag top strip to move, 'b' to toggle)\n"
            "  python dreamer.py --terminal     # text-art in terminal"
        ),
    )
    parser.add_argument("--terminal", "-t", action="store_true", help="run in terminal text-art mode (curses)")
    parser.add_argument("--size", "-s", type=int, default=512, help="initial window size in pixels (GUI mode, default: 512)")
    parser.add_argument("--fps", type=int, default=30, help="target FPS cap (default: 30)")
    parser.add_argument("--borderless", action="store_true", help="start GUI window without a titlebar (drag the top 1/20th to move it, press 'b' to toggle anytime)")
    args = parser.parse_args()

    if args.terminal:
        run_terminal(args.fps)
    else:
        run_gui(args.size, args.fps, borderless=args.borderless)


if __name__ == "__main__":
    main()
