use alloc::{
    collections::BTreeMap,
    format,
    string::String,
    vec,
    vec::Vec,
};

use crossbeam_channel::Receiver;
use eframe::egui;
use egui_plot::{
    LineStyle,
    PlotUi,
    VLine,
};
use linked_hash_map::LinkedHashMap;
use sophus_image::{
    ArcImageF32,
    ImageSize,
};
use sophus_lie::prelude::IsAffineGroup;
use sophus_renderer::{
    HasAspectRatio,
    RenderContext,
    camera::RenderIntrinsics,
};

use crate::{
    WindowParams,
    interactions::ViewportScale,
    layout::{
        WindowArea,
        WindowPlacement,
        show_image,
    },
    packets::{
        CurveVec,
        CurveVecWithConf,
        LineType,
        Packet,
    },
    views::{
        ActiveViewInfo,
        GraphType,
        ImageView,
        PlotView,
        SceneView,
        View,
        ViewMode,
    },
};

extern crate alloc;

/// Viewer top-level struct.
pub struct ViewerBase {
    context: RenderContext,
    views: LinkedHashMap<String, View>,
    message_recv: Receiver<Vec<Packet>>,
    view_mode: ViewMode,
    backface_culling: bool,
    wireframe: bool,
    responses: BTreeMap<String, ResponseStruct>,
    active_view: String,
    active_view_info: Option<ActiveViewInfo>,
    floating_windows: bool,
    show_title_bars: bool,
}

pub(crate) struct ResponseStruct {
    pub(crate) ui_response: egui::Response,
    pub(crate) inverse_distance_image: Option<ArcImageF32>,
    pub(crate) scales: ViewportScale,
    pub(crate) view_port_size: ImageSize,
    pub(crate) view_disabled: bool,
}

/// Configuration for a simple viewer.
pub struct ViewerBaseConfig {
    /// Message receiver.
    pub message_recv: Receiver<Vec<Packet>>,
}

impl ViewerBase {
    /// Create a new viewer.
    pub fn new(render_state: RenderContext, config: ViewerBaseConfig) -> ViewerBase {
        ViewerBase {
            context: render_state.clone(),
            views: LinkedHashMap::new(),
            message_recv: config.message_recv,
            view_mode: ViewMode::default(),
            backface_culling: true,
            wireframe: false,
            responses: BTreeMap::new(),
            active_view_info: None,
            active_view: Default::default(),
            floating_windows: false,
            show_title_bars: false,
        }
    }

    /// Update the data.
    pub fn update_data(&mut self, _ctx: &egui::Context, _frame: &eframe::Frame) {
        Self::process_simple_packets(&mut self.views, &self.context, &self.message_recv);
    }

    /// Process events.
    pub fn process_events(&mut self) {
        for (view_label, view) in self.views.iter_mut() {
            let mut view_port_size = ImageSize::default();
            match view {
                View::Scene(view) => {
                    if let Some(response) = self.responses.get(view_label) {
                        if let Some(inverse_distance_image) = &response.inverse_distance_image {
                            view.interaction.process_event(
                                &mut self.active_view,
                                &view.intrinsics(),
                                view.locked_to_birds_eye_orientation,
                                &response.ui_response,
                                &response.scales,
                                response.view_port_size,
                                Some(inverse_distance_image.clone()),
                            );
                        }
                        view_port_size = response.view_port_size
                    }
                }
                View::Image(view) => {
                    if let Some(response) = self.responses.get(view_label) {
                        // Note: the in-plane interaction of an image view does not use a
                        // z-buffer, so - unlike for a scene view - there is nothing to wait for.
                        view.interaction.process_event(
                            &mut self.active_view,
                            &view.intrinsics(),
                            true,
                            &response.ui_response,
                            &response.scales,
                            response.view_port_size,
                            None,
                        );
                        view_port_size = response.view_port_size
                    }
                }
                View::Plot(_) => {}
            }

            if view.interaction().is_active() && &self.active_view == view_label {
                self.active_view_info = Some(ActiveViewInfo {
                    active_view: view_label.clone(),
                    scene_from_camera: view.interaction().scene_from_camera(),
                    camera_properties: Some(view.camera_propterties()),
                    // is_active, so marker is guaranteed to be Some
                    pivot: view.interaction().marker().unwrap(),
                    view_type: view.view_type(),
                    view_port_size,
                    locked_to_birds_eye_orientation: view.locked_to_birds_eye_orientation(),
                    zoom2d: view.interaction().zoom2d(),
                });
            }
        }
        self.responses.clear();
    }

    /// Update bottom status bar
    pub fn update_top_bar(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
            egui::CollapsingHeader::new("Settings").show(ui, |ui| {
                ui.horizontal(|ui| {
                    // the view modes are alternatives to one another, so they share a combo box;
                    // wireframe and back-face culling are independent of which is shown, and of
                    // each other
                    egui::ComboBox::from_id_salt("view mode")
                        .selected_text(match self.view_mode {
                            ViewMode::Color => "color",
                            ViewMode::Depth => "depth",
                            ViewMode::FrustumPlanes => "frustum planes",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.view_mode, ViewMode::Color, "color");
                            ui.selectable_value(&mut self.view_mode, ViewMode::Depth, "depth");
                            ui.selectable_value(
                                &mut self.view_mode,
                                ViewMode::FrustumPlanes,
                                "frustum planes",
                            )
                            .on_hover_text(
                                "tint each pixel by the frustum of the intermediate it was \
                                 rendered into",
                            );
                        });
                    ui.checkbox(&mut self.wireframe, "wireframe").on_hover_text(
                        "draw the whole scene as edges - each renderable can also be a \
                             wireframe on its own",
                    );
                    ui.checkbox(&mut self.backface_culling, "backface culling");
                });
            });

            let help_button_response = ui.button("❓");

            egui::Popup::from_toggle_button_response(&help_button_response).show(|ui| {
                ui.set_width(250.0);
                ui.label("PAN UP/DOWN + LEFT/RIGHT");
                ui.label("mouse: left-button drag");
                ui.label("touchpad: one finger drag");
                ui.label("");
                ui.label("ROTATE UP/DOWN + LEFT/RIGHT*");
                ui.label("mouse: right-button drag");
                ui.label("touchpad: two finger drag** / shift + drag");
                ui.label("");
                ui.label("ZOOM");
                ui.label("mouse: scroll-wheel");
                ui.label("touchpad: two finger vertical scroll");
                ui.label("");
                ui.label("ROTATE IN-PLANE");
                ui.label("mouse: shift + scroll-wheel");
                ui.label("touchpad: two finger horizontal scroll");
                ui.label("");
                ui.label("RESET THE VIEW");
                ui.label("mouse: double click");
                ui.label("image views: back to the whole image");
                ui.label("scene views: back to the starting camera");
                ui.label("");
                ui.label("* Disabled if locked to birds-eye orientation.");
                ui.label("** Does not work on all touchpads.");
            });
        });
    }

    /// Update the left panel.
    pub fn update_left_panel(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        for (view_label, view) in self.views.iter_mut() {
            ui.checkbox(view.enabled_mut(), view_label);
        }
        ui.separator();
        ui.checkbox(&mut self.floating_windows, "floating windows");
        ui.checkbox(&mut self.show_title_bars, "show title bars");
        ui.separator();
    }

    /// Update bottom status bar
    pub fn update_bottom_status_bar(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        match self.active_view_info.as_ref() {
            Some(view_info) => {
                if let Some(camera_properties) = view_info.camera_properties.as_ref() {
                    ui.horizontal_wrapped(|ui| {
                        // The pivot is the point an interaction turns about: a pixel, and how far
                        // away along the ray through it. An image view has no scene behind it,
                        // hence no distance to show.
                        //
                        // Given as inverse distance first, which is what the buffer behind it
                        // holds and the unit a point too far away to have a useful distance is
                        // parameterised in, with the metres it comes to beside it.
                        let pivot = match view_info.pivot.distance.is_finite() {
                            true => format!(
                                "pivot: ({:0.1}, {:0.1}) at {:0.4} 1/m ({:0.3} m)",
                                view_info.pivot.u,
                                view_info.pivot.v,
                                1.0 / (view_info.pivot.distance as f64).max(1e-9),
                                view_info.pivot.distance,
                            ),
                            false => {
                                format!(
                                    "pivot: ({:0.1}, {:0.1})",
                                    view_info.pivot.u, view_info.pivot.v
                                )
                            }
                        };
                        ui.label(format!(
                            "{}: {}, view-port: {} x {}, image: {} x {}, clip: [{}, {}], {pivot}",
                            view_info.view_type,
                            view_info.active_view,
                            view_info.view_port_size.width,
                            view_info.view_port_size.height,
                            camera_properties.intrinsics.image_size().width,
                            camera_properties.intrinsics.image_size().height,
                            camera_properties.clipping_planes.near,
                            camera_properties.clipping_planes.far,
                        ));

                        // An image view is always seen from the identity pose, so the camera
                        // pose says nothing - what moves is the 2d zoom and pan.
                        if view_info.view_type == "Image" {
                            let zoom = view_info.zoom2d;
                            ui.label(format!(
                                "ZOOM: {:0.2}x, pan: ({:0.1}, {:0.1}) image px",
                                zoom.scaling[0],
                                -zoom.translation[0] / zoom.scaling[0],
                                -zoom.translation[1] / zoom.scaling[1],
                            ));
                        } else {
                            // the rotation as a rotation vector - the log of the rotation -
                            // which reads as an axis scaled by the angle in radians, rather
                            // than as four quaternion components
                            let rotation_vector = view_info.scene_from_camera.rotation().log();

                            ui.label(format!(
                                "CAMERA position: ({:0.3}, {:0.3}, {:0.3}), rotation-vector: \
                                ({:0.4}, {:0.4}, {:0.4})",
                                view_info.scene_from_camera.translation()[0],
                                view_info.scene_from_camera.translation()[1],
                                view_info.scene_from_camera.translation()[2],
                                rotation_vector[0],
                                rotation_vector[1],
                                rotation_vector[2],
                            ));

                            // A view locked to the bird's eye orientation cannot be rotated,
                            // which is a property of the view rather than of the packet which
                            // created it - so it is toggled here, on whichever view is active.
                            let mut locked = view_info.locked_to_birds_eye_orientation;
                            if ui
                                .checkbox(&mut locked, "bird's eye")
                                .on_hover_text(
                                    "lock this view to the bird's eye orientation - unlock it to \
                                     orbit",
                                )
                                .changed()
                                && let Some(View::Scene(scene_view)) =
                                    self.views.get_mut(&view_info.active_view)
                            {
                                scene_view.locked_to_birds_eye_orientation = locked;
                                // Locking is what the flag says, but turning the view to look
                                // down is what is being asked for: the lock alone leaves a view
                                // pointing wherever it already was, and unable to be turned.
                                if locked {
                                    scene_view.interaction.look_straight_down();
                                }
                            }

                            // The displayed pose is rounded, which is not enough to reproduce a
                            // view - so hand out the full precision, as something which can be
                            // pasted straight into code.
                            if ui
                                .button("COPY CAMERA")
                                .on_hover_text("copy `scene_from_camera` to the clipboard, as Rust")
                                .clicked()
                            {
                                let translation = view_info.scene_from_camera.translation();
                                let image_size = camera_properties.intrinsics.image_size();
                                let intrinsics = match &camera_properties.intrinsics {
                                    RenderIntrinsics::Pinhole(pinhole) => format!(
                                        "DynCameraF64::new_pinhole(\n        \
                                         VecF64::from_array({:?}),\n        \
                                         ImageSize::new({}, {}),\n    )",
                                        pinhole.params().as_slice(),
                                        image_size.width,
                                        image_size.height,
                                    ),
                                    RenderIntrinsics::UnifiedExtended(unified) => format!(
                                        "DynCameraF64::new_enhanced_unified(\n        \
                                         VecF64::from_array({:?}),\n        \
                                         ImageSize::new({}, {}),\n    )",
                                        unified.params().as_slice(),
                                        image_size.width,
                                        image_size.height,
                                    ),
                                };
                                ui.ctx().copy_text(format!(
                                    "// {}\n\
                                     // pivot: {:0.4} 1/m ({:0.3} m)\n\
                                     let camera = {intrinsics};\n\
                                     let clipping_planes = ClippingPlanes {{\n    \
                                     near: {:?},\n    far: {:?},\n}};\n\
                                     let scene_from_camera = \
                                     Isometry3::from_rotation_and_translation(\n    \
                                     Rotation3::exp(VecF64::<3>::new({:?}, {:?}, {:?})),\n    \
                                     VecF64::<3>::new({:?}, {:?}, {:?}),\n);",
                                    view_info.active_view,
                                    1.0 / (view_info.pivot.distance as f64).max(1e-9),
                                    view_info.pivot.distance,
                                    camera_properties.clipping_planes.near,
                                    camera_properties.clipping_planes.far,
                                    rotation_vector[0],
                                    rotation_vector[1],
                                    rotation_vector[2],
                                    translation[0],
                                    translation[1],
                                    translation[2],
                                ));
                            }
                        }
                    });
                } else {
                    ui.label(format!(
                        "{}: {}, view-port: {} x {}",
                        view_info.view_type,
                        view_info.active_view,
                        view_info.view_port_size.width,
                        view_info.view_port_size.height,
                    ));
                }
            }
            None => {
                ui.label("view: n/a");
            }
        }
    }

    /// Update the central panel.
    pub fn update_central_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.scope(|ui0| {
            if self.views.is_empty() {
                return;
            }

            ui0.horizontal_wrapped(|ui| {
                let mut boxes = vec![];
                for (view_label, view) in self.views.iter_mut() {
                    if !view.enabled() {
                        continue;
                    }
                    let view_aspect_ratio = view.aspect_ratio();
                    boxes.push(WindowArea {
                        view_label: view_label.clone(),
                        width_by_height_ratio: view_aspect_ratio,
                    });
                }

                let rects = WindowArea::flow_layout(ui, &boxes, self.show_title_bars);

                for placement in rects.iter() {
                    let view = self.views.get_mut(&placement.view_label).unwrap();

                    match view {
                        View::Scene(view) => {
                            let response = view.render(
                                ctx,
                                self.context.clone(),
                                placement,
                                WindowParams {
                                    view_mode: self.view_mode,
                                    backface_culling: self.backface_culling,
                                    wireframe: self.wireframe,
                                    floating_windows: self.floating_windows,
                                    show_title_bars: self.show_title_bars,
                                },
                            );

                            if let Some(response) = response {
                                if response.view_disabled {
                                    view.enabled = false;
                                }
                                self.responses
                                    .insert(placement.view_label.clone(), response);
                            }
                        }
                        View::Image(view) => {
                            let view_port_size = placement.viewport_size();

                            let render_result = view
                                .renderer
                                .render_params(
                                    &view_port_size,
                                    &view.interaction.scene_from_camera(),
                                )
                                .zoom(view.interaction.zoom2d())
                                .interaction(view.interaction.marker())
                                .backface_culling(self.backface_culling)
                                .wireframe(self.wireframe)
                                .render();

                            let (ui_response, view_disabled) = show_image(
                                ctx,
                                render_result.rgba_egui_tex_id,
                                placement,
                                self.floating_windows,
                                self.show_title_bars,
                            );

                            if view_disabled {
                                view.enabled = false;
                            }

                            self.responses.insert(
                                placement.view_label.clone(),
                                ResponseStruct {
                                    scales: ViewportScale::from_image_size_and_viewport_rect(
                                        view.intrinsics().image_size(),
                                        ui_response.rect,
                                    ),
                                    ui_response,
                                    inverse_distance_image: None,
                                    view_port_size,
                                    view_disabled,
                                },
                            );
                        }
                        View::Plot(view) => {
                            Self::show_plot(
                                ctx,
                                view,
                                placement,
                                self.floating_windows,
                                self.show_title_bars,
                            );
                        }
                    }
                }
            });
        });
    }

    fn show_plot(
        ctx: &egui::Context,
        view: &mut PlotView,
        placement: &WindowPlacement,
        floating_windows: bool,
        show_title_bars: bool,
    ) {
        let mut enabled = true;

        fn color_cnv(color: sophus_renderer::renderables::Color) -> egui::Color32 {
            egui::Color32::from_rgb(
                (color.r * 255.0).clamp(0.0, 255.0) as u8,
                (color.g * 255.0).clamp(0.0, 255.0) as u8,
                (color.b * 255.0).clamp(0.0, 255.0) as u8,
            )
        }

        fn add_vline(plot_ui: &mut PlotUi, name: &str, x: f64) {
            plot_ui
                .add(VLine::new(name.to_owned(), x).color(egui::Color32::from_rgb(255, 255, 255)));
        }

        fn show_vec<const N: usize>(curve_name: &str, g: &CurveVec<N>, plot_ui: &mut PlotUi) {
            if let Some(v_line) = &g.v_line {
                add_vline(plot_ui, &v_line.name, v_line.x);
            }

            let mut points: Vec<Vec<egui_plot::PlotPoint>> = vec![Vec::new(); N];
            for (x, y) in &g.data {
                for i in 0..N {
                    points[i].push(egui_plot::PlotPoint::new(*x, y[i]));
                }
            }

            match g.style.line_type {
                LineType::LineStrip => {
                    for (i, p) in points.iter().enumerate().take(N) {
                        let plot_points = egui_plot::PlotPoints::Owned(p.clone());
                        plot_ui.line(
                            egui_plot::Line::new(format!("{curve_name}-{i}"), plot_points)
                                .color(color_cnv(g.style.colors[i])),
                        );
                    }
                }
                LineType::Points => {
                    // note: use points() for point style
                    for (i, p) in points.iter().enumerate().take(N) {
                        let plot_points = egui_plot::PlotPoints::Owned(p.clone());
                        plot_ui.points(
                            egui_plot::Points::new(format!("{curve_name}-{i}"), plot_points)
                                .color(color_cnv(g.style.colors[i])),
                        );
                    }
                }
            }
        }

        fn show_vec_conf<const N: usize>(
            curve_name: &str,
            g: &CurveVecWithConf<N>,
            plot_ui: &mut PlotUi,
        ) {
            if let Some(v_line) = &g.v_line {
                add_vline(plot_ui, &v_line.name, v_line.x);
            }

            let mut mid: Vec<Vec<egui_plot::PlotPoint>> = vec![Vec::new(); N];
            let mut up: Vec<Vec<egui_plot::PlotPoint>> = vec![Vec::new(); N];
            let mut dn: Vec<Vec<egui_plot::PlotPoint>> = vec![Vec::new(); N];

            for (x, (y, e)) in &g.data {
                for i in 0..N {
                    mid[i].push(egui_plot::PlotPoint::new(*x, y[i]));
                    up[i].push(egui_plot::PlotPoint::new(*x, y[i] + e[i]));
                    dn[i].push(egui_plot::PlotPoint::new(*x, y[i] - e[i]));
                }
            }

            let mut draw = |series: Vec<Vec<egui_plot::PlotPoint>>, style: LineStyle| {
                for (i, p) in series.iter().enumerate().take(N) {
                    let plot_points = egui_plot::PlotPoints::Owned(p.clone());
                    plot_ui.line(
                        egui_plot::Line::new(format!("{curve_name}-{i}"), plot_points)
                            .color(color_cnv(g.style.colors[i]))
                            .style(style),
                    );
                }
            };

            draw(mid, LineStyle::Solid);
            let dash = LineStyle::dashed_dense();
            draw(up, dash);
            draw(dn, dash);
        }

        // one place to render every curve
        let mut render_curves = |plot_ui: &mut PlotUi| {
            for (curve_name, graph_data) in &mut view.curves {
                if !graph_data.show_graph {
                    continue;
                }
                match &graph_data.curve {
                    GraphType::Scalar(g) => {
                        if let Some(v_line) = &g.v_line {
                            add_vline(plot_ui, &v_line.name, v_line.x);
                        }

                        let points: Vec<_> = g
                            .data
                            .iter()
                            .map(|(x, y)| egui_plot::PlotPoint::new(*x, *y))
                            .collect();
                        let plot_points = egui_plot::PlotPoints::Owned(points);

                        match g.style.line_type {
                            LineType::LineStrip => {
                                plot_ui.line(
                                    egui_plot::Line::new(curve_name, plot_points)
                                        .color(color_cnv(g.style.color)),
                                );
                            }
                            LineType::Points => {
                                plot_ui.points(
                                    egui_plot::Points::new(curve_name, plot_points)
                                        .color(color_cnv(g.style.color)),
                                );
                            }
                        }
                    }
                    GraphType::Vec2(g) => show_vec(curve_name, g, plot_ui),
                    GraphType::Vec3(g) => show_vec(curve_name, g, plot_ui),
                    GraphType::Vec2Conf(g) => show_vec_conf(curve_name, g, plot_ui),
                    GraphType::Vec3Conf(g) => show_vec_conf(curve_name, g, plot_ui),
                }
            }
        };

        // shared plot builder with optional size for the non-floating window
        let plot_size = if floating_windows {
            None
        } else {
            Some((placement.content_size.x, placement.content_size.y))
        };

        // single window path with conditional tweaks
        let mut win = egui::Window::new(placement.view_label.clone())
            .collapsible(false)
            .open(&mut enabled);

        win = if floating_windows {
            win.resizable(true).title_bar(true)
        } else {
            win.title_bar(show_title_bars)
                .fixed_pos(placement.rect.min)
                .fixed_size(egui::Vec2::new(
                    placement.rect.width(),
                    placement.rect.height(),
                ))
        };

        win.show(ctx, |ui| {
            let mut plot = egui_plot::Plot::new(placement.view_label.clone())
                .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop));

            if let Some((w, h)) = plot_size {
                plot = plot.width(w).height(h);
            }

            plot.show(ui, |plot_ui| {
                render_curves(plot_ui);
            });
        });

        if !enabled {
            view.enabled = false;
        }
    }

    pub(crate) fn process_simple_packets(
        views: &mut LinkedHashMap<String, View>,
        context: &RenderContext,
        message_recv: &Receiver<Vec<Packet>>,
    ) {
        loop {
            let maybe_stream = message_recv.try_recv();
            if maybe_stream.is_err() {
                break;
            }
            let stream = maybe_stream.unwrap();
            for packet in stream {
                match packet {
                    Packet::Scene(packet) => SceneView::update(views, packet, context),
                    Packet::Image(packet) => ImageView::update(views, packet, context),
                    Packet::Plot(packets) => {
                        for packet in packets {
                            PlotView::update(views, packet)
                        }
                    }
                }
            }
        }
    }
}
