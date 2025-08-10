use alloc::{
    collections::BTreeMap,
    format,
    string::String,
    vec,
    vec::Vec,
};

use eframe::egui;
use egui_plot::{
    LineStyle,
    PlotUi,
    VLine,
};
use linked_hash_map::LinkedHashMap;
use sophus_autodiff::prelude::HasParams;
use sophus_image::{
    ArcImageF32,
    ImageSize,
};
use sophus_lie::prelude::IsAffineGroup;
use sophus_renderer::{
    HasAspectRatio,
    RenderContext,
};
use thingbuf::mpsc::blocking::Receiver;

use crate::{
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
    },
};

extern crate alloc;

/// Viewer top-level struct.
pub struct ViewerBase {
    context: RenderContext,
    views: LinkedHashMap<String, View>,
    message_recv: Receiver<Vec<Packet>>,
    show_depth: bool,
    backface_culling: bool,
    responses: BTreeMap<String, ResponseStruct>,
    active_view: String,
    active_view_info: Option<ActiveViewInfo>,
    floating_windows: bool,
    show_title_bars: bool,
}

pub(crate) struct ResponseStruct {
    pub(crate) ui_response: egui::Response,
    pub(crate) z_image: Option<ArcImageF32>,
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
            show_depth: false,
            backface_culling: false,
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
                        if response.z_image.is_some() {
                            view.interaction.process_event(
                                &mut self.active_view,
                                &view.intrinsics(),
                                view.locked_to_birds_eye_orientation,
                                &response.ui_response,
                                &response.scales,
                                response.view_port_size,
                                Some(response.z_image.as_ref().unwrap().clone()),
                            );
                        }
                        view_port_size = response.view_port_size
                    }
                }
                View::Image(view) => {
                    if let Some(response) = self.responses.get(view_label) {
                        if response.z_image.is_some() {
                            view.interaction.process_event(
                                &mut self.active_view,
                                &view.intrinsics(),
                                true,
                                &response.ui_response,
                                &response.scales,
                                response.view_port_size,
                                None,
                            );
                        }
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
                    scene_focus: view.interaction().marker().unwrap(),
                    view_type: view.view_type(),
                    view_port_size,
                    locked_to_birds_eye_orientation: view.locked_to_birds_eye_orientation(),
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
                    ui.checkbox(&mut self.show_depth, "show depth");
                    ui.checkbox(&mut self.backface_culling, "backface culling");
                });
            });

            let help_button_response = ui.button("❓");

            let popup_id = ui.make_persistent_id("help");
            if help_button_response.clicked() {
                ui.memory_mut(|mem| mem.toggle_popup(popup_id));
            }
            let below = egui::AboveOrBelow::Below;
            let close_on_click_outside = egui::popup::PopupCloseBehavior::CloseOnClickOutside;
            egui::popup::popup_above_or_below_widget(
                ui,
                popup_id,
                &help_button_response,
                below,
                close_on_click_outside,
                |ui| {
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
                    ui.label("* Disabled if locked to birds-eye orientation.");
                    ui.label("** Does not work on all touchpads.");
                },
            );
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
                        ui.label(format!(
                            "{}: {}, view-port: {} x {}, image: {} x {}, clip: [{}, {}], \
                            focus uv: {:0.1} {:0.1}, ndc-z: {:0.3}, metric-z: {:0.3}",
                            view_info.view_type,
                            view_info.active_view,
                            view_info.view_port_size.width,
                            view_info.view_port_size.height,
                            camera_properties.intrinsics.image_size().width,
                            camera_properties.intrinsics.image_size().height,
                            camera_properties.clipping_planes.near,
                            camera_properties.clipping_planes.far,
                            view_info.scene_focus.u,
                            view_info.scene_focus.v,
                            view_info.scene_focus.ndc_z,
                            camera_properties
                                .clipping_planes
                                .metric_z_from_ndc_z(view_info.scene_focus.ndc_z as f64),
                        ));

                        let scene_from_camera_orientation = view_info.scene_from_camera.rotation();
                        let scene_from_camera_quaternion = scene_from_camera_orientation.params();

                        ui.label(format!(
                            "CAMERA position: ({:0.3}, {:0.3}, {:0.3}), quaternion: {:0.4}, \
                            ({:0.4}, {:0.4}, {:0.4}), bird's eye view: {}",
                            view_info.scene_from_camera.translation()[0],
                            view_info.scene_from_camera.translation()[1],
                            view_info.scene_from_camera.translation()[2],
                            scene_from_camera_quaternion[0],
                            scene_from_camera_quaternion[1],
                            scene_from_camera_quaternion[2],
                            scene_from_camera_quaternion[3],
                            view_info.locked_to_birds_eye_orientation
                        ));
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
                                self.show_depth,
                                self.backface_culling,
                                self.context.clone(),
                                placement,
                                self.floating_windows,
                                self.show_title_bars,
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
                                    ui_response,
                                    scales: ViewportScale::from_image_size_and_viewport_size(
                                        view.intrinsics().image_size(),
                                        placement,
                                    ),
                                    z_image: None,
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
        let bar_size = if show_title_bars {
            WindowArea::BAR_SIZE
        } else {
            0.0
        };
        let plot_size = if floating_windows {
            None
        } else {
            Some((
                placement.rect.width() - WindowArea::BORDER,
                placement.rect.height() - WindowArea::BORDER - bar_size,
            ))
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
