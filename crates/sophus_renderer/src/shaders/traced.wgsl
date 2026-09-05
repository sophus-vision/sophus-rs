// Primitives which are intersected with the ray through each pixel, rather than rasterized.
//
// The distortion pass already works out the exact ray of every pixel of the distorted image, so
// tracing a primitive against it costs one intersection and needs nothing else: no intermediate
// to render into, no resampling, and no upper bound on the field of view. A sphere traced this
// way is round at any distance and under any distortion.

struct TracedEllipsoid {
    // in *camera* coordinates - the cpu puts them there, so the shader has nothing to transform
    center: vec3<f32>,
    // encloses the ellipsoid, for a cheap reject before the full intersection
    bounding_radius: f32,
    // rows of the map taking the ellipsoid to the unit sphere, padded to four
    to_unit_sphere: mat3x4<f32>,
    color: vec4<f32>,
    // 1 to draw as edges, whatever the rest of the scene does
    wireframe: f32,
    padding0: f32,
    padding1: f32,
    padding2: f32,
};

// A sphere is an ellipsoid whose shape is a scaled identity, so there is one intersection routine
// rather than two. It costs two matrix-vector products more than a dedicated sphere would, and
// saves a second code path.
fn to_unit_sphere_space(ellipsoid: TracedEllipsoid, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(ellipsoid.to_unit_sphere[0].xyz, v),
        dot(ellipsoid.to_unit_sphere[1].xyz, v),
        dot(ellipsoid.to_unit_sphere[2].xyz, v),
    );
}

// The transpose of the same map, which is what carries a normal back out of that space.
fn from_unit_sphere_space_normal(ellipsoid: TracedEllipsoid, v: vec3<f32>) -> vec3<f32> {
    return ellipsoid.to_unit_sphere[0].xyz * v.x
        + ellipsoid.to_unit_sphere[1].xyz * v.y
        + ellipsoid.to_unit_sphere[2].xyz * v.z;
}

// How far inside the silhouette this ray passes, as a fraction of the ellipsoid's own size: one
// through the middle, zero on the silhouette, negative outside.
//
// This is the margin the coverage is measured from. It cannot be a distance in world units - for
// an anisotropic ellipsoid there is no single such distance - so it is measured in the space where
// the ellipsoid is the unit sphere, and turned into pixels by comparing it with the margin of the
// neighbouring pixel's ray.
fn silhouette_margin(
    ellipsoid: TracedEllipsoid,
    to_center: vec3<f32>,
    direction: vec3<f32>,
) -> f32 {
    let d = to_unit_sphere_space(ellipsoid, direction);
    let o = to_unit_sphere_space(ellipsoid, to_center);
    return 1.0 - length(cross(o, d)) / length(d);
}

// A flat piece of a plane: the whole plane, an ellipse of it, or a rectangle of it. The surface
// is the same in every case and only the test applied to the hit point differs, so there is one
// intersection - a single division - and three predicates.
struct TracedPlanar {
    center: vec3<f32>,
    // 0 unbounded, 1 elliptical, 2 rectangular
    bound: f32,
    // rows taking a point of the plane to the coordinates the bound is tested in
    to_unit_shape: mat2x4<f32>,
    normal: vec3<f32>,
    line_width: f32,
    color: vec4<f32>,
    pattern_scale: f32,
    // 0 plain, 1 grid, 2 checker
    pattern: f32,
    // 1 to draw as edges, whatever the rest of the scene does
    wireframe: f32,
    padding0: f32,
    // the other colour of a checkerboard
    pattern_color: vec4<f32>,
};

// A capsule: every point within `radius` of the segment. Its ends are hemispheres, so two of them
// meeting at a joint show no seam - and there are no caps to clip against.
struct TracedCapsule {
    // `from` and `to` would read better, but `from` is a reserved word in wgsl
    p0: vec3<f32>,
    radius: f32,
    p1: vec3<f32>,
    // 1 to draw as edges, whatever the rest of the scene does
    wireframe: f32,
    color: vec4<f32>,
    // 1 to cut the ends flat - a cylinder - rather than rounding them over
    flat_ends: f32,
    padding0: f32,
    padding1: f32,
    padding2: f32,
};

// A cone, from its apex along its axis. The curved surface only: a disk closes it off.
struct TracedCone {
    apex: vec3<f32>,
    height: f32,
    axis: vec3<f32>,
    radius: f32,
    color: vec4<f32>,
    // 1 to draw as edges, whatever the rest of the scene does
    wireframe: f32,
    padding0: f32,
    padding1: f32,
    padding2: f32,
};

struct TracedPrimitives {
    ellipsoid_count: u32,
    planar_count: u32,
    capsule_count: u32,
    cone_count: u32,
    // the light the meshes are shaded by, so that traced and rasterized surfaces sit in one scene
    light_in_camera: vec4<f32>,
    ellipsoids: array<TracedEllipsoid, 256>,
    planars: array<TracedPlanar, 512>,
    capsules: array<TracedCapsule, 512>,
    cones: array<TracedCone, 512>,
};

@group(2) @binding(0) var<storage, read> traced: TracedPrimitives;

struct TracedPixel {
    rgb: vec3<f32>,
    inverse_distance: f32,
};

// The rays of the next pixel across and the next one down, which is how a traced silhouette is
// antialiased: there are no derivatives in a compute shader, so the size of a pixel has to be
// found by asking for one.
//
// Both, not just one. A surface raking away from the viewer moves far more under a step down the
// image than across it - near the horizon of a ground plane, a hundred times more - and a
// footprint taken along one axis says nothing about the other.
struct NeighbouringRays {
    across: vec3<f32>,
    down: vec3<f32>,
};

fn neighbouring_rays(
    uv_distorted: vec2<f32>,
    image_size: vec2<f32>,
    view_port_size: vec2<u32>,
) -> NeighbouringRays {
    let step = image_size / vec2<f32>(view_port_size);
    var out: NeighbouringRays;
    out.across = normalize(distorted_to_ray(uv_distorted + vec2<f32>(step.x, 0.0), camera));
    out.down = normalize(distorted_to_ray(uv_distorted + vec2<f32>(0.0, step.y), camera));
    return out;
}

// How far inside its bound the point lies, as a fraction of the shape's own size: one at the
// centre, zero on the edge, negative outside. The coordinates are the ones the bound is unit in,
// so an ellipse is the unit circle and a rectangle the unit square.
fn planar_margin(planar: TracedPlanar, hit: vec3<f32>) -> f32 {
    let to_center = hit - planar.center;
    let local = vec2<f32>(
        dot(planar.to_unit_shape[0].xyz, to_center),
        dot(planar.to_unit_shape[1].xyz, to_center),
    );
    if (planar.bound > 1.5) {
        return 1.0 - max(abs(local.x), abs(local.y));
    }
    return 1.0 - length(local);
}

// Metric coordinates in the plane, which the grid is ruled in. The rows taking a point to the
// coordinates its bound is unit in are the dual basis of the axes, so normalizing them gives the
// axes back whenever those are orthogonal - and a grid on a skewed pair of axes is skewed, which
// is what it should be.
fn planar_metric(planar: TracedPlanar, hit: vec3<f32>) -> vec2<f32> {
    let to_center = hit - planar.center;
    return vec2<f32>(
        dot(normalize(planar.to_unit_shape[0].xyz), to_center),
        dot(normalize(planar.to_unit_shape[1].xyz), to_center),
    );
}

// What is left of a grid where no ruling can be read: haze. Faint enough to read as a surface
// which is not quite there, and neutral, so that the far field of a ground plane goes quiet
// rather than staying a saturated wash of whatever colour the lines are.
const GRID_HAZE_ALPHA: f32 = 0.22;
const GRID_HAZE_RGB: vec3<f32> = vec3<f32>(0.55, 0.55, 0.55);

// How far a pixel reaches along the plane against how far away it is, between which the haze
// comes in. A property of the geometry rather than of the zoom - so the haze stays where it is as
// the camera pulls back, where anything keyed on the footprint alone would close in and swallow
// the plane.
const GRID_HAZE_FROM: f32 = 0.1;
const GRID_HAZE_TO: f32 = 0.5;

// The finest ruling drawn: its cells are never closer together than this on screen.
const GRID_MIN_PIXELS: f32 = 12.0;

// What the ruling steps by, so that a heavier line falls every fifth one - close enough together
// to count without counting far.
const GRID_STEP: f32 = 5.0;

// `1 / log2(GRID_STEP)`, which turns a ratio of spacings into steps of the ruling.
const GRID_STEP_LOG2: f32 = 0.4306766;

// Over how many steps a ruling fades in, which is a factor of two of its spacing.
const GRID_FADE: f32 = 0.4306766;

// How much wider a line is drawn one step above the finest ruling, so that the steps can be told
// apart and the eye has something to count.
const GRID_THICK: f32 = 2.0;

// How many steps are ruled at once. Three, not two, because the lines of a fourth would be a
// subset of the third's: nothing new appears when the ruling steps, so the step cannot be seen.
const GRID_STEPS: u32 = 3u;

// How much of a pixel one line of the grid covers, filtered over the pixel's footprint.
//
// Near the camera this is a line of `half_width` with an antialiased edge. As the plane rakes away
// and one pixel comes to span a whole period, it becomes the *average* of the line over that
// period - which is what stops a ground grid from turning to noise at the horizon, where a
// point-sampled one alternates between drawing a line and missing it entirely.
fn grid_line_coverage(x: f32, spacing: f32, footprint: f32, line_width: f32) -> f32 {
    let half_width = 0.5 * line_width * footprint;
    // Lines fall on the half cells rather than on the multiples, which is what stops any one of
    // them from being special. Zero is a multiple of every spacing there is, so a ruling anchored
    // on the multiples draws the two lines through the origin at every step at once: they come
    // out heavier than their neighbours and run all the way to the horizon, while everything
    // around them has long since thinned away. Half cells nest just as well - `GRID_STEP` is odd,
    // so every half cell of a coarse ruling is a half cell of a fine one - and no line then
    // belongs to every step.
    let to_line = abs(fract(x / spacing) - 0.5) * spacing;
    let sharp = clamp((half_width - to_line) / max(footprint, 1e-12) + 0.5, 0.0, 1.0);
    // what the line averages to once a pixel spans a period or more
    let average = clamp(2.0 * half_width / spacing, 0.0, 1.0);
    return mix(sharp, average, clamp(footprint / spacing, 0.0, 1.0));
}

// One axis of the grid: three steps of the spacing asked for, drawn at once.
//
// A single ruling can only be read over about two decades of zoom - closer than that and there is
// one line on the screen, further and the lines merge into a sheet, which says nothing about
// scale, the only reason to rule a grid. So the ruling steps instead, keyed on the pixel's own
// footprint: each step is faded in as its cells grow large enough to read and drawn wider the
// coarser it is, and the whole thing then holds at any distance and any zoom. It also holds
// *within* one image, which is what a ground plane needs: the ruling at the horizon is several
// steps coarser than the ruling underfoot.
//
// `finest` bounds how far it will subdivide, in steps of the spacing asked for.
fn grid_axis_coverage(
    x: f32,
    base_spacing: f32,
    footprint: f32,
    line_width: f32,
    finest: f32,
) -> f32 {
    // the step whose cells come out `GRID_MIN_PIXELS` apart on screen, which is the finest one
    // this pixel can hold - fractional, since it is what everything else is measured against
    let readable = log2(max(footprint, 1e-12) * GRID_MIN_PIXELS / base_spacing) * GRID_STEP_LOG2;
    let level = max(floor(readable) + 1.0, finest);
    // Widths are judged against the finest ruling that *could* be read, so that the steps keep
    // their weights as the ruling steps. A grid which will not subdivide past the spacing asked
    // for has nothing finer to be judged against, so it is judged against itself.
    let reference = min(readable, level);

    var covered = 0.0;
    for (var d = 0u; d < GRID_STEPS; d = d + 1u) {
        let this_level = level + f32(d);
        let spacing = base_spacing * pow(GRID_STEP, this_level);
        let weight = clamp((this_level - readable) / GRID_FADE, 0.0, 1.0);
        let width =
            line_width * mix(1.0, GRID_THICK, clamp(this_level - reference - 1.0, 0.0, 1.0));
        covered = max(covered, weight * grid_line_coverage(x, spacing, footprint, width));
    }
    return covered;
}

// Distance from the ray to the segment, which is what a capsule is measured from: the surface is
// the points at `radius` from it. Both ends are clamped, so this covers the hemispherical caps as
// well as the body, and it is the margin the coverage is taken from.
fn ray_segment_distance(start: vec3<f32>, along: vec3<f32>, direction: vec3<f32>) -> f32 {
    // minimise |t * direction - (from + s * along)|, with `direction` a unit vector
    let to_start = dot(direction, start);
    let base = to_start * direction - start;
    let slope = dot(along, direction) * direction - along;
    let slope2 = dot(slope, slope);
    var s = 0.0;
    if (slope2 > 1e-12) {
        s = clamp(-dot(base, slope) / slope2, 0.0, 1.0);
    }
    let closest = start + s * along;
    let t = max(dot(direction, closest), 0.0);
    return length(t * direction - closest);
}

// The discriminant of the cone's quadratic for a given ray, which is what the coverage is
// measured from - it is smooth, and vanishes exactly on the silhouette.
fn cone_discriminant(cone: TracedCone, cos2: f32, direction: vec3<f32>) -> f32 {
    let m = dot(direction, cone.axis);
    let n = dot(cone.apex, cone.axis);
    let p = dot(direction, cone.apex);
    let q = dot(cone.apex, cone.apex);
    let a = m * m - cos2;
    let b = cos2 * p - m * n;
    let c = n * n - cos2 * q;
    return b * b - a * c;
}

// A checkerboard, filtered analytically over the pixel's footprint.
//
// The checker alternates by a square wave, whose average over an interval is the difference of its
// antiderivative - a triangle wave - at the ends. Integrating once more and taking the *second*
// difference convolves the pattern with a tent instead of a box, which is the box convolved with
// itself: it falls away at the edge of the footprint rather than stopping, so the pattern stops
// shimmering as the camera moves.
//
// `checker_primitive` is that second antiderivative, from integrating the triangle wave: over one
// period it is a run of parabolas, which `d * (2 |d| - 1)` traces out, and the `0.5 x` carries the
// mean. Filtering a procedural pattern by integrating it in closed form is Inigo Quilez's, see
// https://iquilezles.org/articles/filterableprocedurals
fn checker_primitive(x: vec2<f32>) -> vec2<f32> {
    let d = fract(0.5 * x) - vec2<f32>(0.5, 0.5);
    return 0.5 * x + d * (2.0 * abs(d) - vec2<f32>(1.0, 1.0));
}

fn checker_coverage(local: vec2<f32>, size: f32, footprint: vec2<f32>) -> f32 {
    let x = local / size;
    // never a kernel of no width, which two coincident neighbours would ask for
    let w = max(footprint / size, vec2<f32>(1e-4, 1e-4));
    let i = (checker_primitive(x + w) - 2.0 * checker_primitive(x) + checker_primitive(x - w))
        / (w * w);
    return 0.5 + 0.5 * i.x * i.y;
}

// A traced primitive drawn as a wireframe is drawn as its silhouette - the outline of the shape
// itself, rather than of any tessellation of it, which is the one place a traced primitive has an
// edge at all.
//
// `inside` is how far inside that silhouette the pixel lies, in pixels, which every primitive
// here already works out for its own antialiasing.
fn silhouette_band(inside: f32) -> f32 {
    return clamp(inside + 0.5, 0.0, 1.0)
        - clamp(inside - WIREFRAME_WIDTH + 0.5, 0.0, 1.0);
}

fn coverage_from_inside(inside: f32, entity_wireframe: f32) -> f32 {
    if (draws_as_wireframe(pinhole.wireframe, entity_wireframe)) {
        return silhouette_band(inside);
    }
    return clamp(inside + 0.5, 0.0, 1.0);
}

// Composites the traced primitives over what was rasterized, by distance along the ray.
//
// `direction` must be a unit vector in the camera frame, `neighbour` the unit ray of the next
// pixel across, and `inverse_distance` what the rasterized scene left here - the last is directly
// comparable with what the tracer solves for, since the distance image measures the same thing.
fn composite_traced(
    rgb: vec3<f32>,
    inverse_distance: f32,
    direction: vec3<f32>,
    neighbours: NeighbouringRays,
) -> TracedPixel {
    var out: TracedPixel;
    out.rgb = rgb;
    out.inverse_distance = inverse_distance;

    for (var i = 0u; i < traced.ellipsoid_count; i = i + 1u) {
        let ellipsoid = traced.ellipsoids[i];
        let center = ellipsoid.center;

        // how far along the ray the centre lies, and how far the ray passes from it
        let along = dot(center, direction);
        if (along <= 0.0) {
            // behind the camera - an ellipsoid the camera sits inside is not drawn
            continue;
        }
        if (length(cross(center, direction)) > ellipsoid.bounding_radius) {
            // the ray misses the enclosing sphere, so it misses the ellipsoid
            continue;
        }

        // Coverage of the pixel, from how far inside the silhouette the ray passes against how
        // much that changes from one pixel to the next. The rasterized surfaces get this from
        // multisampling; a traced one has to work it out, since the distortion pass writes one
        // sample per pixel and a compute shader has no derivatives.
        let margin = silhouette_margin(ellipsoid, center, direction);
        let margin_step = max(
            abs(margin - silhouette_margin(ellipsoid, center, neighbours.across)),
            abs(margin - silhouette_margin(ellipsoid, center, neighbours.down)));
        let coverage = coverage_from_inside(margin / max(margin_step, 1e-12), ellipsoid.wireframe);
        if (coverage <= 0.0) {
            continue;
        }

        // Distance to the surface, solved where the ellipsoid is the unit sphere. `t` carries over
        // unchanged because the direction is *not* normalized there - both ends of the ray go
        // through the same linear map.
        //
        // The discriminant is `|d|^2 - |o x d|^2` rather than `(o.d)^2 - |d|^2 (|o|^2 - 1)`. They
        // are the same number by Lagrange's identity, but the second subtracts two values of order
        // `|o|^2` to reach one of order one: for 2-5 cm semi-axes at 300 m it comes out negative,
        // and the ellipsoid disappears.
        let d = to_unit_sphere_space(ellipsoid, direction);
        let o = to_unit_sphere_space(ellipsoid, -center);
        let a = dot(d, d);
        let b = dot(o, d);
        let cr = cross(o, d);
        let discriminant = a - dot(cr, cr);

        var t = -b / a;
        if (discriminant > 0.0) {
            let root = sqrt(discriminant);
            t = (-b - root) / a;
            if (t <= 0.0) {
                // the camera is inside the ellipsoid: take the surface behind it
                t = (-b + root) / a;
            }
        }
        if (t <= 0.0) {
            continue;
        }

        let ellipsoid_inverse_depth = 1.0 / t;
        if (ellipsoid_inverse_depth <= out.inverse_distance) {
            // something rasterized, or a nearer primitive, is already in front of this one
            continue;
        }

        // the gradient of `|M(x - c)|^2` is `2 M^T M (x - c)`, so the normal comes back through
        // the transpose of the map, not the map
        let normal = normalize(from_unit_sphere_space_normal(ellipsoid, o + t * d));
        let shaded = shade(normal, traced.light_in_camera.xyz, ellipsoid.color.rgb);
        let alpha = ellipsoid.color.a * coverage;
        out.rgb = mix(out.rgb, shaded, alpha);
        if (alpha > 0.5) {
            // the pixel belongs to the ellipsoid, so the depth does too
            out.inverse_distance = ellipsoid_inverse_depth;
        }
    }

    for (var i = 0u; i < traced.planar_count; i = i + 1u) {
        let planar = traced.planars[i];

        // one division, and the plane is behind the camera or edge on when it does not divide
        let facing = dot(planar.normal, direction);
        if (abs(facing) < 1e-9) {
            continue;
        }
        let t = dot(planar.normal, planar.center) / facing;
        if (t <= 0.0) {
            continue;
        }

        let planar_inverse_depth = 1.0 / t;
        if (planar_inverse_depth <= out.inverse_distance) {
            continue;
        }

        // How far inside the bound the hit lies, and how much that changes from one pixel to the
        // next - the same margin rule the ellipsoid uses. The neighbouring ray is intersected
        // too, which for a plane raking away from the viewer is what keeps the edge from
        // crawling.
        // Where the neighbouring pixels' rays meet the same plane, which gives both the margin's
        // gradient and the footprint the grid is filtered over. Both neighbours: a plane raking
        // away from the viewer moves far further under a step down the image than across it.
        let hit = t * direction;
        var across_hit = hit;
        var down_hit = hit;
        var has_across = false;
        var has_down = false;
        let across_facing = dot(planar.normal, neighbours.across);
        if (abs(across_facing) > 1e-9) {
            let across_t = dot(planar.normal, planar.center) / across_facing;
            if (across_t > 0.0) {
                across_hit = across_t * neighbours.across;
                has_across = true;
            }
        }
        let down_facing = dot(planar.normal, neighbours.down);
        if (abs(down_facing) > 1e-9) {
            let down_t = dot(planar.normal, planar.center) / down_facing;
            if (down_t > 0.0) {
                down_hit = down_t * neighbours.down;
                has_down = true;
            }
        }

        var coverage = 1.0;
        if (planar.bound > 0.5) {
            let margin = planar_margin(planar, hit);
            var across_margin = -1.0;
            var down_margin = -1.0;
            var margin_step = 0.0;
            if (has_across) {
                across_margin = planar_margin(planar, across_hit);
                margin_step = abs(margin - across_margin);
            }
            if (has_down) {
                down_margin = planar_margin(planar, down_hit);
                margin_step = max(margin_step, abs(margin - down_margin));
            }

            // A pixel can only be partly covered if it, or a pixel beside it, is inside the
            // bound. Without that: as a ray comes to run parallel to the plane its hit races off
            // to infinity, so the margin and its step are both enormous and their ratio lands
            // somewhere in the middle - which draws the shape, faintly, along the whole circle of
            // directions parallel to its plane. On a disk the size of an arrow head that is a
            // hairline across the sky.
            if (margin <= 0.0 && across_margin <= 0.0 && down_margin <= 0.0) {
                continue;
            }

            let inside = margin / max(margin_step, 1e-12);
            coverage = coverage_from_inside(inside, planar.wireframe);
            if (!draws_as_wireframe(pinhole.wireframe, planar.wireframe)
                && planar.pattern < 0.5
                && planar.line_width > 0.0) {
                // an outline: the shape less the shape shrunk by the width, so a ring
                coverage = coverage - clamp(inside - planar.line_width + 0.5, 0.0, 1.0);
            }
            if (coverage <= 0.0) {
                continue;
            }
        }

        var pattern_rgb = planar.color.rgb;
        if (planar.pattern > 0.5) {
            let local = planar_metric(planar, hit);
            // One footprint per grid axis, each the larger of what a step across the image and a
            // step down it do to that axis. They differ enormously on a raking plane: near the
            // horizon of a ground plane one row down is many grid periods along the axis running
            // away from the viewer, while one column across is a fraction of one.
            var footprint = vec2<f32>(planar.pattern_scale, planar.pattern_scale);
            if (has_across || has_down) {
                footprint = vec2<f32>(0.0, 0.0);
                if (has_across) {
                    footprint = abs(planar_metric(planar, across_hit) - local);
                }
                if (has_down) {
                    footprint = max(footprint, abs(planar_metric(planar, down_hit) - local));
                }
            }
            // A checkerboard has no edges to draw, so as a wireframe it is ruled instead: the
            // same squares, as lines. An unbounded plane has no silhouette either, so this is all
            // there is to draw of a ground plane.
            let as_grid = planar.pattern < 1.5 || draws_as_wireframe(pinhole.wireframe, planar.wireframe);
            if (as_grid) {
                // a grid: only the lines are drawn, so the scene shows through between them
                let line_width = select(planar.line_width, WIREFRAME_WIDTH,
                    draws_as_wireframe(pinhole.wireframe, planar.wireframe) && planar.pattern > 1.5);
                // An unbounded plane subdivides as far as the camera cares to go; a bounded
                // one is never ruled finer than it was asked to be, since there its spacing is a
                // property of the thing being drawn rather than of the view.
                let finest = select(-40.0, 0.0, planar.bound > 0.5);
                let on_grid = max(
                    grid_axis_coverage(
                        local.x, planar.pattern_scale, footprint.x, line_width, finest),
                    grid_axis_coverage(
                        local.y, planar.pattern_scale, footprint.y, line_width, finest));

                // Towards the horizon a pixel comes to span a great deal of the plane, and no
                // ruling can be read there however coarse it is. What is left is the plane
                // itself, as haze: it runs to its horizon whether or not anything can be said
                // about its scale there, and drawing nothing at all instead ends it early, which
                // gives a ruled plane a horizon of its own.
                //
                // Cross-faded rather than taken whichever is larger: the lines are on their way
                // out exactly where the haze is on its way in, and the larger of the two dips
                // between them - a dark ring around the horizon.
                let slant = max(footprint.x, footprint.y) / t;
                let hazy = clamp(
                    (slant - GRID_HAZE_FROM) / (GRID_HAZE_TO - GRID_HAZE_FROM), 0.0, 1.0);
                coverage = coverage * mix(on_grid, GRID_HAZE_ALPHA, hazy);
                pattern_rgb = mix(pattern_rgb, GRID_HAZE_RGB, hazy);
                if (coverage <= 0.0) {
                    continue;
                }
            } else {
                // A checker: the plane is drawn throughout, in two shades of its colour. The
                // filter is never narrower than one view-port pixel of the plane, so that a
                // boundary between squares has somewhere to be antialiased over.
                let other = checker_coverage(local, planar.pattern_scale, footprint);
                pattern_rgb = mix(planar.color.rgb, planar.pattern_color.rgb, other);
            }
        }

        let shaded = shade(planar.normal, traced.light_in_camera.xyz, pattern_rgb);
        let alpha = planar.color.a * coverage;
        out.rgb = mix(out.rgb, shaded, alpha);
        if (alpha > 0.5) {
            out.inverse_distance = planar_inverse_depth;
        }
    }

    for (var i = 0u; i < traced.capsule_count; i = i + 1u) {
        let capsule = traced.capsules[i];
        let along = capsule.p1 - capsule.p0;

        // Coverage first, since it also says whether the ray comes near the capsule at all. The
        // margin is how far inside the surface the ray passes, as a fraction of the radius.
        let margin =
            1.0 - ray_segment_distance(capsule.p0, along, direction) / capsule.radius;
        let margin_step = max(
            abs(margin - (1.0
                - ray_segment_distance(capsule.p0, along, neighbours.across) / capsule.radius)),
            abs(margin - (1.0
                - ray_segment_distance(capsule.p0, along, neighbours.down) / capsule.radius)));
        let coverage = coverage_from_inside(margin / max(margin_step, 1e-12), capsule.wireframe);
        if (coverage <= 0.0) {
            continue;
        }

        // The body, solved where the cross products keep it well conditioned: `u x v` against
        // `r |along| |v|` is a difference of comparable small quantities, where the textbook
        // `baba*oaoa - baoa^2` subtracts two of order `|from|^2` - and misses a two centimetre
        // capsule at 300 m outright.
        let u = cross(along, -capsule.p0);
        let v = cross(along, direction);
        let a = dot(v, v);
        let b = dot(u, v);
        let uv = cross(u, v);
        let along2 = dot(along, along);
        let h = capsule.radius * capsule.radius * along2 * a - dot(uv, uv);

        var t = -1.0;
        if (a > 1e-12 && h >= 0.0) {
            let body_t = (-b - sqrt(h)) / a;
            // within the segment, rather than on the infinite cylinder it lies on
            let s = dot(along, body_t * direction - capsule.p0);
            if (body_t > 0.0 && s >= 0.0 && s <= along2) {
                t = body_t;
            }
        }
        // Cut flat, the body *is* the whole of it: the clip on the body above is the cylinder's
        // two end planes, and what would round it over is skipped. The ends are then open, and
        // the disks which close them are traced in their own right - which is also what gives the
        // rim an antialiased edge, since the clip along the axis is a hard one.
        if (t <= 0.0 && capsule.flat_ends < 0.5) {
            // the hemispherical ends, each of them the sphere case
            for (var which_end = 0u; which_end < 2u; which_end = which_end + 1u) {
                let center = select(capsule.p0, capsule.p1, which_end == 1u);
                let center_along = dot(center, direction);
                if (center_along <= 0.0) {
                    continue;
                }
                let perpendicular = length(cross(center, direction));
                let discriminant =
                    capsule.radius * capsule.radius - perpendicular * perpendicular;
                if (discriminant < 0.0) {
                    continue;
                }
                let cap_t = center_along - sqrt(discriminant);
                if (cap_t > 0.0 && (t <= 0.0 || cap_t < t)) {
                    t = cap_t;
                }
            }
        }
        if (t <= 0.0) {
            continue;
        }

        let capsule_inverse_depth = 1.0 / t;
        if (capsule_inverse_depth <= out.inverse_distance) {
            continue;
        }

        // the nearest point of the segment to the hit, which the normal points away from
        let hit = t * direction;
        let s = clamp(dot(hit - capsule.p0, along) / along2, 0.0, 1.0);
        let normal = normalize(hit - (capsule.p0 + s * along));
        let shaded = shade(normal, traced.light_in_camera.xyz, capsule.color.rgb);
        let alpha = capsule.color.a * coverage;
        out.rgb = mix(out.rgb, shaded, alpha);
        if (alpha > 0.5) {
            out.inverse_distance = capsule_inverse_depth;
        }
    }

    for (var i = 0u; i < traced.cone_count; i = i + 1u) {
        let cone = traced.cones[i];

        // `(w . axis)^2 = cos(half angle)^2 |w|^2` for `w` from the apex to the surface, which is
        // a quadratic in `t`. Unlike the ellipsoid and the capsule, nothing here cancels as the
        // cone gets small and distant: its size enters only through the angle.
        let cos2 = cone.height * cone.height
            / (cone.height * cone.height + cone.radius * cone.radius);
        let m = dot(direction, cone.axis);
        let n = dot(cone.apex, cone.axis);
        let p = dot(direction, cone.apex);
        let q = dot(cone.apex, cone.apex);
        let a = m * m - cos2;
        let b = cos2 * p - m * n;
        let c = n * n - cos2 * q;
        let discriminant = b * b - a * c;
        if (discriminant < 0.0) {
            continue;
        }
        let root = sqrt(discriminant);

        // The roots of `a t^2 + 2 b t + c`, paired through `c` rather than both divided by `a`.
        //
        // `a` is `m^2 - cos^2` and vanishes for a ray running parallel to the cone's surface,
        // where one root goes to infinity and the other stays finite. Dividing both by `a` turns
        // the finite one into garbage of the same enormous size as the other, and garbage that
        // large sometimes lands back inside the height - which drew the cone as a hairline across
        // the sky, along the locus of directions making exactly the cone's own angle with its
        // axis. Paired through `c`, the finite root stays finite and the other stays out of
        // range.
        let pair = -(b + select(-root, root, b >= 0.0));
        var roots = vec2<f32>(-1.0, -1.0);
        if (abs(a) > 1e-9) {
            roots.x = pair / a;
        }
        if (abs(pair) > 1e-9) {
            roots.y = c / pair;
        }

        // the nearer root which is on the cone rather than on the mirror image of it behind the
        // apex, and within the height
        var t = -1.0;
        for (var which = 0u; which < 2u; which = which + 1u) {
            let candidate = select(roots.x, roots.y, which == 1u);
            if (candidate <= 0.0) {
                continue;
            }
            let up = dot(candidate * direction - cone.apex, cone.axis);
            if (up < 0.0 || up > cone.height) {
                continue;
            }
            if (t <= 0.0 || candidate < t) {
                t = candidate;
            }
        }
        if (t <= 0.0) {
            continue;
        }

        let cone_inverse_depth = 1.0 / t;
        if (cone_inverse_depth <= out.inverse_distance) {
            continue;
        }

        // the discriminant is smooth and vanishes on the silhouette, so it serves as the margin -
        // the rule only needs the zero to be in the right place, the scale coming from the step
        let margin_step = max(
            abs(discriminant - cone_discriminant(cone, cos2, neighbours.across)),
            abs(discriminant - cone_discriminant(cone, cos2, neighbours.down)));
        let coverage = coverage_from_inside(discriminant / max(margin_step, 1e-12), cone.wireframe);
        if (coverage <= 0.0) {
            continue;
        }

        let w = t * direction - cone.apex;
        // the surface normal of a cone: along the surface, turned out by the half angle
        let radial = w - dot(w, cone.axis) * cone.axis;
        let normal = normalize(
            normalize(radial) * cone.height - cone.axis * cone.radius);
        let shaded = shade(normal, traced.light_in_camera.xyz, cone.color.rgb);
        let alpha = cone.color.a * coverage;
        out.rgb = mix(out.rgb, shaded, alpha);
        if (alpha > 0.5) {
            out.inverse_distance = cone_inverse_depth;
        }
    }

    return out;
}
