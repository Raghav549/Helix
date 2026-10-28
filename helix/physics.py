"""Small, deterministic, physically informed scene renderer.

Pinhole camera, explicit sphere/box intersections, hard shadow rays, Lambertian
surface response, inverse-square point lighting and a normalized Blinn-Phong
specular approximation. This is a bounded research instrument, not a general
path tracer or a claim of photorealism. Images and ground truth labels are
suitable for controlled synthetic experiments; no external assets are used.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import re
import numpy as np
from PIL import Image

COLORS = {
    'red': (0.8, 0.055, 0.035),
    'green': (0.035, 0.65, 0.07),
    'blue': (0.035, 0.09, 0.8),
    'yellow': (0.8, 0.6, 0.025),
    'white': (0.75, 0.75, 0.75),
}


@dataclass(frozen=True)
class Scene:
    shape: str = 'sphere'
    color: str = 'red'
    position: str = 'center'
    material: str = 'matte'
    light: str = 'left'
    fov: float = 45.0
    size: int = 64

    def __post_init__(self):
        if self.shape not in ('sphere', 'cube') or self.color not in COLORS:
            raise ValueError('supported shapes: sphere, cube; colors: ' + ', '.join(COLORS))
        if self.position not in ('left', 'center', 'right'):
            raise ValueError('position must be left, center, or right')
        if self.material not in ('matte', 'glossy') or self.light not in ('left', 'right'):
            raise ValueError('invalid material or light direction')
        if not 20 <= self.fov <= 80 or not 16 <= self.size <= 512:
            raise ValueError('fov must be 20..80 degrees and size 16..512 pixels')


def scene_from_prompt(prompt: str, size: int = 64) -> Scene:
    """Interpret a disclosed, small controlled vocabulary; unknown words are ignored."""
    if not isinstance(prompt, str) or len(prompt) > 2048:
        raise ValueError('prompt too long')
    text = prompt.lower()
    tokens = set(re.findall(r'[a-z]+', text))
    color = next((c for c in COLORS if c in tokens), 'red')
    shape = 'cube' if tokens & {'cube', 'box'} else 'sphere'
    position = 'center'
    # Distinguish object placement from lighting placement.
    if re.search(r'(?:on|to|at|positioned|placed)\s+(?:the\s+)?left', text): position = 'left'
    if re.search(r'(?:on|to|at|positioned|placed)\s+(?:the\s+)?right', text): position = 'right'
    light = 'right' if re.search(r'(?:light|lit|lighting)[^.]{0,30}\bright\b', text) else 'left'
    material = 'glossy' if tokens & {'glossy', 'shiny', 'polished'} else 'matte'
    fov = 65.0 if tokens & {'wide', 'wide-angle'} else 30.0 if tokens & {'telephoto'} else 45.0
    return Scene(shape, color, position, material, light, fov, size)


def _normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _object_intersection(origin, direction, center, shape):
    if shape == 'sphere':
        offset = origin - center
        b = np.sum(offset * direction, axis=-1)
        c = np.sum(offset * offset, axis=-1) - .65 ** 2
        disc = b*b - c
        root = np.sqrt(np.maximum(disc, 0))
        near, far = -b-root, -b+root
        t = np.where(near > 1e-4, near, far)
        return np.where((disc >= 0) & (t > 1e-4), t, np.inf)
    inv = np.divide(1.0, direction, out=np.full_like(direction, 1e12), where=np.abs(direction) > 1e-12)
    a = (center - .65 - origin) * inv
    b = (center + .65 - origin) * inv
    tmin = np.max(np.minimum(a, b), axis=-1)
    tmax = np.min(np.maximum(a, b), axis=-1)
    t = np.where(tmin > 1e-4, tmin, tmax)
    return np.where((tmax >= np.maximum(tmin, 0)) & (t > 1e-4), t, np.inf)


def render_scene(scene: Scene) -> tuple[Image.Image, dict]:
    n = scene.size
    eye = np.array([0., 1.6, 4.8])
    target = np.array([0., .55, 0.])
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, np.array([0., 1., 0.])))
    up = _normalize(np.cross(right, forward))
    scale = np.tan(np.deg2rad(scene.fov) / 2)
    xx, yy = np.meshgrid((np.arange(n) + .5) / n * 2 - 1,
                         1 - (np.arange(n) + .5) / n * 2)
    rays = _normalize(forward + xx[..., None] * right * scale + yy[..., None] * up * scale)
    origins = np.broadcast_to(eye, rays.shape)
    x = {'left': -.85, 'center': 0., 'right': .85}[scene.position]
    center = np.array([x, .65, 0.])
    obj_t = _object_intersection(origins, rays, center, scene.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        floor_t = -origins[..., 1] / rays[..., 1]
    floor_t = np.where((rays[..., 1] < -1e-8) & (floor_t > 0), floor_t, np.inf)
    is_obj = obj_t < floor_t
    t = np.minimum(obj_t, floor_t)
    visible = np.isfinite(t)
    safe_t = np.where(visible, t, 0.)
    points = origins + rays * safe_t[..., None]
    normals = np.zeros_like(rays)
    normals[..., 1] = 1.
    if scene.shape == 'sphere':
        object_normal = _normalize(points - center)
    else:
        relative = points - center
        axis = np.argmax(np.abs(relative), axis=-1)
        object_normal = np.zeros_like(relative)
        np.put_along_axis(object_normal, axis[..., None],
                          np.take_along_axis(np.sign(relative), axis[..., None], axis=-1), axis=-1)
    normals = np.where(is_obj[..., None], object_normal, normals)
    albedo = np.where(is_obj[..., None], np.array(COLORS[scene.color]), np.array([.36, .36, .36]))
    lamp = np.array([-3. if scene.light == 'left' else 3., 4.5, 3.])
    to_light = lamp - points
    distance = np.maximum(np.linalg.norm(to_light, axis=-1), 1e-6)
    light_dir = to_light / distance[..., None]
    shadow_t = _object_intersection(points + normals * .002, light_dir, center, scene.shape)
    shadow = (shadow_t < distance - .002) & visible
    diffuse = np.maximum(np.sum(normals * light_dir, axis=-1), 0.)
    radiance = 18. / (distance * distance)
    direct = np.where(shadow, 0., diffuse * radiance)
    view = _normalize(eye - points)
    half = _normalize(light_dir + view)
    shininess = 64. if scene.material == 'glossy' else 8.
    spec = np.maximum(np.sum(normals * half, axis=-1), 0.) ** shininess
    spec *= np.where(is_obj & (scene.material == 'glossy'), .45, .035) * radiance
    spec = np.where(shadow, 0., spec)
    linear = albedo * (.16 + direct[..., None]) + spec[..., None]
    rgb = np.clip(linear, 0., 1.) ** (1. / 2.2)
    sky = np.broadcast_to(np.array([.66, .75, .86]), rgb.shape)
    rgb = np.where(visible[..., None], rgb, sky)
    pixels = np.round(np.clip(rgb, 0., 1.) * 255).astype(np.uint8)
    info = {'scene': asdict(scene), 'renderer': 'HELIX pinhole/direct-light v1',
            'object_pixels': int(is_obj.sum()), 'shadow_pixels': int((shadow & ~is_obj).sum()),
            'depth_min': float(np.min(t[visible])) if visible.any() else None,
            'limitations': 'single primitive, direct light and hard shadows; no global illumination'}
    return Image.fromarray(pixels, 'RGB'), info
