mod pixel_ellipse;
mod pixel_line;
mod pixel_point;
mod scene_overlay;

use eframe::wgpu;
pub use pixel_ellipse::*;
pub use pixel_line::*;
pub use pixel_point::*;
use sophus_autodiff::linalg::{
    MatF64,
    SVec,
    VecF64,
};

use crate::{
    RenderContext,
    camera::RenderIntrinsics,
    pipeline_builder::{
        EllipseVertex2,
        PipelineBuilder,
        TargetTexture,
    },
    pixel_renderer::{
        pixel_ellipse::{
            PIVOT_ELLIPSES,
            ellipse_vertex,
        },
        pixel_line::PixelLineRenderer,
        pixel_point::PixelPointRenderer,
        scene_overlay::SceneOverlayRenderer,
    },
    prelude::*,
    renderables::{
        Color,
        Ellipse2,
    },
    textures::DepthTextures,
    types::{
        PivotGesture,
        ScenePivotMarker,
    },
    uniform_buffers::{
        OVERLAY_POSE_SLOT,
        VertexShaderUniformBuffers,
    },
};

/// How big the pivot marker is drawn, across, in image pixels.
const PIVOT_RADIUS_PIXELS: f64 = 26.0;

/// How wide its rings are, in view-port pixels, and how thick its bars are.
const PIVOT_LINE_WIDTH: f32 = 2.0;
const PIVOT_BAR_HALF_WIDTH_PIXELS: f64 = 1.8;

/// How far open a ring is drawn, against its own length.
const PIVOT_RING_APERTURE: f64 = 0.34;

/// The shortest a bar is drawn, so that the axis into the screen is still there to see.
const PIVOT_MIN_BAR_PIXELS: f64 = 9.0;

/// How big the dot at the middle of it is, in image pixels.
const PIVOT_CENTRE_RADIUS_PIXELS: f64 = 3.0;

/// Renderer for pixel data
pub struct PixelRenderer {
    pub(crate) line_renderer: PixelLineRenderer,
    pub(crate) point_renderer: PixelPointRenderer,
    pub(crate) ellipse_renderer: PixelEllipseRenderer,
    /// the scene's own lines and points, which are drawn here rather than into the intermediate
    pub(crate) scene_overlay: SceneOverlayRenderer,
    pub(crate) pixel_pipeline_builder: PipelineBuilder,
}

impl PixelRenderer {
    /// Create a new pixel renderer
    pub fn new(render_context: &RenderContext, uniforms: Arc<VertexShaderUniformBuffers>) -> Self {
        let pixel_pipeline_builder = PipelineBuilder::new_pixel(
            render_context,
            Arc::new(TargetTexture {
                rgba_output_format: wgpu::TextureFormat::Rgba8Unorm,
            }),
            uniforms.clone(),
        );

        Self {
            line_renderer: PixelLineRenderer::new(render_context, &pixel_pipeline_builder),
            point_renderer: PixelPointRenderer::new(render_context, &pixel_pipeline_builder),
            ellipse_renderer: PixelEllipseRenderer::new(render_context, &pixel_pipeline_builder),
            scene_overlay: SceneOverlayRenderer::new(render_context, &pixel_pipeline_builder),
            pixel_pipeline_builder,
        }
    }

    /// The pivot of an interaction, drawn as what that interaction does.
    ///
    /// Every gesture works in the camera's *current* frame - a drag turns the view about the
    /// camera's x and y, a sideways scroll about its z, a drag along the ground translates along
    /// its x and y - so the marker is drawn on those axes, worked out afresh each frame. It shows
    /// the frame the gesture acts in rather than a thing in the scene, so it neither turns with
    /// the view nor snaps back to anything.
    ///
    /// A turn is drawn as the ring it turns along and a slide as a bar along the way it slides,
    /// so the glyph says what the pointer will do next as well as where it will do it from. They
    /// are circles and segments *in the scene*, projected, so their size and their foreshortening
    /// are the camera's own - which a glyph drawn in the image could not be.
    pub(crate) fn show_interaction_marker(
        &self,
        context: &RenderContext,
        marker: &Option<ScenePivotMarker>,
        intrinsics: &RenderIntrinsics,
    ) {
        let Some(marker) = marker else {
            *self.ellipse_renderer.show_interaction_marker.lock() = false;
            return;
        };

        let pixel = VecF64::<2>::new(marker.u as f64, marker.v as f64);
        let direction = intrinsics.cam_unproj_to_unit_vector(&pixel);
        let pivot = direction * marker.distance as f64;
        let center = intrinsics.cam_proj(&pivot);
        let offset_at = |offset: VecF64<3>| intrinsics.cam_proj(&(pivot + offset)) - center;

        // How big a metre is here, in pixels - measured by projecting a step across the view
        // rather than read off a focal length, so the marker holds its size on screen under any
        // camera model. Along the ray a step measures nothing, hence "across".
        let mut unit = VecF64::<3>::zeros();
        unit[direction.iamin()] = 1.0;
        let across = direction.cross(&unit).normalize();
        let step = 1e-3 * marker.distance as f64;
        let pixels_per_metre = offset_at(across * step).norm() / step.max(1e-12);
        let radius = PIVOT_RADIUS_PIXELS / pixels_per_metre.max(1e-9);

        let axis = |i: usize| {
            let mut axis = VecF64::<3>::zeros();
            axis[i] = radius;
            axis
        };
        // A bar along `i`, as long as the ball is wide and a couple of pixels thick.
        //
        // The axis into the screen projects to a stub, and to nothing at all at the principal
        // point, so it is given a floor on its length. Which way it points is still what the
        // projection says - away from where the camera looks, the way things move as they come
        // towards you.
        let bar = |i: usize| {
            let projected = offset_at(axis(i));
            let along = match projected.norm() > PIVOT_MIN_BAR_PIXELS {
                true => projected,
                false => match projected.norm() > 1e-6 {
                    true => projected.normalize() * PIVOT_MIN_BAR_PIXELS,
                    false => VecF64::<2>::new(0.0, PIVOT_MIN_BAR_PIXELS),
                },
            };
            let across = VecF64::<2>::new(-along.y, along.x);
            MatF64::<2, 2>::from_columns(&[along, across.normalize() * PIVOT_BAR_HALF_WIDTH_PIXELS])
        };

        // The ring a turn about `i` runs along, which is the circle spanned by the other two axes.
        //
        // About the axis the camera looks along, that circle faces the camera and is drawn as the
        // projection makes it - it is also the silhouette of the ball. About either of the others
        // the circle is seen exactly edge on, since its plane holds the direction the camera looks,
        // so the projection of it is a segment and there is no honest ellipse to draw. Those are
        // opened to a fixed aperture about the screen's own axes instead: which way the drag goes
        // and that something turns is all they are for. Opening up the true projection would tilt
        // them towards the principal point, since that is the way a step along the view axis moves
        // a pixel, and they would come out skewed rather than foreshortened.
        let ring_about = |i: usize| {
            let spanning = [offset_at(axis((i + 1) % 3)), offset_at(axis((i + 2) % 3))];
            if i == 2 {
                return MatF64::<2, 2>::from_columns(&spanning);
            }
            // the one of the two which is not the axis into the screen, and so is the way the
            // edge-on circle lies
            let along = match spanning[0].norm() >= spanning[1].norm() {
                true => spanning[0],
                false => spanning[1],
            };
            let across = VecF64::<2>::new(-along.y, along.x);
            MatF64::<2, 2>::from_columns(&[
                along,
                across.normalize() * (PIVOT_RING_APERTURE * along.norm()),
            ])
        };

        // The whole marker is drawn whatever is happening: a ring about each axis, for the turns,
        // and a bar along each, for the slides. What the gesture is decides which of them is lit.
        // Grey says the thing is there; a colour says the pointer is working on it.
        let idle = Color {
            r: 0.62,
            g: 0.62,
            b: 0.66,
            a: 1.0,
        };
        let red = Color {
            r: 0.90,
            g: 0.22,
            b: 0.22,
            a: 1.0,
        };
        let green = Color {
            r: 0.16,
            g: 0.72,
            b: 0.26,
            a: 1.0,
        };
        let blue = Color {
            r: 0.26,
            g: 0.44,
            b: 0.95,
            a: 1.0,
        };
        // Turning about the axis a vertical drag works on is red, as x is; about the one a
        // horizontal drag works on green, as y is; about the axis into the screen blue, as z is.
        // The bars take the same three colours along the axes they slide on.
        let turns = match marker.gesture {
            PivotGesture::Orbit => [red, green, idle],
            PivotGesture::Roll => [idle, idle, blue],
            _ => [idle; 3],
        };
        let slides = match marker.gesture {
            PivotGesture::Pan => [red, green, idle],
            PivotGesture::Zoom => [idle, idle, blue],
            _ => [idle; 3],
        };

        // Rings are outlines; bars and the middle are filled, which is a line width of zero. A
        // view which cannot be turned across the screen is drawn without the two rings which
        // stand for that - what is left is the ring it can be rolled along and the three ways it
        // can be moved.
        let first_ring = match marker.can_orbit {
            true => 0,
            false => 2,
        };
        let mut glyphs: Vec<(MatF64<2, 2>, f32, Color)> = (first_ring..3)
            .map(|i| (ring_about(i), PIVOT_LINE_WIDTH, turns[i]))
            .chain((0..3).map(|i| (bar(i), 0.0, slides[i])))
            .collect();
        // and the middle of it, in the colour the marker arrives with - how far away the pivot is,
        // mapped the way the depth view maps it. The one thing here which says something about the
        // point rather than about the gesture.
        glyphs.push((
            MatF64::<2, 2>::identity() * PIVOT_CENTRE_RADIUS_PIXELS,
            0.0,
            marker.color,
        ));

        // Grey behind each of them, a little larger and a little wider, so that the marker reads
        // against whatever it is held over.
        let grey = Color {
            r: 0.16,
            g: 0.16,
            b: 0.18,
            a: 0.9,
        };
        let center = SVec::<f32, 2>::new(center.x as f32, center.y as f32);
        let ellipse = |shape: MatF64<2, 2>, line_width: f32, color: Color| Ellipse2 {
            center,
            shape,
            line_width,
            color,
        };
        let halos = glyphs.iter().map(|(shape, line_width, _)| {
            ellipse(
                shape * 1.10,
                match *line_width > 0.0 {
                    true => line_width + 2.4,
                    false => 0.0,
                },
                grey,
            )
        });
        let rings = glyphs
            .iter()
            .map(|(shape, line_width, color)| ellipse(*shape, *line_width, *color));

        // Anything left degenerate after all that has no interior to draw, and the draw call is a
        // fixed six instances - so the rest is filled up with nothing.
        let nothing = EllipseVertex2 {
            _center: [0.0, 0.0],
            _half_extent: [0.0, 0.0],
            _to_unit_circle: [0.0; 4],
            _color: [0.0; 4],
            _line_width: 0.0,
            _padding: [0.0; 3],
        };
        let mut vertex_data = halos
            .chain(rings)
            .filter_map(|ellipse| ellipse_vertex(&ellipse))
            .collect::<Vec<_>>();
        vertex_data.resize(PIVOT_ELLIPSES, nothing);

        context.wgpu_queue.write_buffer(
            &self.ellipse_renderer.interaction_vertex_buffer,
            0,
            bytemuck::cast_slice(&vertex_data),
        );
        *self.ellipse_renderer.show_interaction_marker.lock() = true;
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        render_context: &RenderContext,
        command_encoder: &'rp mut wgpu::CommandEncoder,
        texture_view: &'rp wgpu::TextureView,
        depth: &DepthTextures,
    ) {
        // The scene's own lines and points, in a pass of their own: they write the scene's
        // inverse distance as well as the colour, which the 2d renderables over them do not.
        //
        // A pass cannot read the texture it writes, so the depth is copied first and the copy is
        // what they are occluded against. Both only when there is something to draw.
        let scene_depth = &depth.main_render_ndc_z_texture;
        if !self.scene_overlay.is_empty() {
            command_encoder.copy_texture_to_texture(
                scene_depth.final_texture.as_image_copy(),
                scene_depth.read_copy_texture.as_image_copy(),
                scene_depth.final_texture.size(),
            );
            let depth_bind_group =
                render_context
                    .wgpu_device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("scene overlay depth bind group"),
                        layout: &self.scene_overlay.depth_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &scene_depth.read_copy_texture_view,
                            ),
                        }],
                    });

            let mut overlay_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene overlay pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: texture_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &scene_depth.final_texture_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // They are held in world coordinates, so they are drawn from the slot which holds the
            // camera's own pose rather than any entity's.
            let uniforms = &self.pixel_pipeline_builder.uniforms;
            overlay_pass.set_bind_group(
                0,
                &uniforms.render_bind_group,
                &[OVERLAY_POSE_SLOT * uniforms.camera_from_entity_pose_buffer.slot_stride as u32],
            );
            overlay_pass.set_bind_group(1, &depth_bind_group, &[]);
            self.scene_overlay.paint(&mut overlay_pass);
        }

        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        // 2d renderables have no entity pose; bind the first (identity) pose slot
        render_pass.set_bind_group(
            0,
            &self.pixel_pipeline_builder.uniforms.render_bind_group,
            &[0],
        );

        self.line_renderer.paint(&mut render_pass);
        self.ellipse_renderer.paint(&mut render_pass);
        self.point_renderer.paint(&mut render_pass);
    }
}
