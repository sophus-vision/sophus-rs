use alloc::{
    collections::BTreeMap,
    format,
    string::String,
    vec,
    vec::Vec,
};

use eframe::egui::{
    self,
    Ui,
};
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
    renderables::Color,
};
use thingbuf::mpsc::blocking::Receiver;

use crate::{
    interactions::ViewportScale,
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
        ViewportSize,
        get_adjusted_view_size,
        get_max_size,
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
}

pub(crate) struct ResponseStruct {
    pub(crate) ui_response: egui::Response,
    pub(crate) z_image: Option<ArcImageF32>,
    pub(crate) scales: ViewportScale,
    pub(crate) view_port_size: ImageSize,
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
        }
    }

    /// Update the data.
    pub fn update_data(&mut self) {
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
    pub fn update_central_panel(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        ui.scope(|ui0| {
            if self.views.is_empty() {
                return;
            }
            let maybe_max_size = get_max_size(
                &self.views,
                0.99 * ui0.available_width(),
                0.99 * ui0.available_height(),
            );
            if maybe_max_size.is_none() {
                return;
            }
            let (max_width, max_height) = maybe_max_size.unwrap();

            ui0.horizontal_wrapped(|ui| {
                // for loop to draw all the views, but not the plots yet
                for (view_label, view) in self.views.iter_mut() {
                    if !view.enabled() {
                        continue;
                    }
                    let view_aspect_ratio = view.aspect_ratio();
                    let adjusted_size =
                        get_adjusted_view_size(view_aspect_ratio, max_width, max_height);
                    match view {
                        View::Scene(view) => {
                            let response = view.render(
                                ui,
                                self.show_depth,
                                self.backface_culling,
                                self.context.clone(),
                                adjusted_size,
                            );

                            if let Some(response) = response {
                                self.responses.insert(view_label.to_owned(), response);
                            }
                        }
                        View::Image(view) => {
                            let render_result = view
                                .renderer
                                .render_params(
                                    &adjusted_size.image_size(),
                                    &view.interaction.scene_from_camera(),
                                )
                                .zoom(view.interaction.zoom2d())
                                .interaction(view.interaction.marker())
                                .backface_culling(self.backface_culling)
                                .render();

                            let ui_response = ui.add(
                                egui::Image::new(egui::load::SizedTexture {
                                    size: egui::Vec2::new(
                                        adjusted_size.width,
                                        adjusted_size.height,
                                    ),
                                    id: render_result.rgba_egui_tex_id,
                                })
                                .fit_to_exact_size(egui::Vec2 {
                                    x: adjusted_size.width,
                                    y: adjusted_size.height,
                                })
                                .sense(egui::Sense::click_and_drag()),
                            );

                            self.responses.insert(
                                view_label.clone(),
                                ResponseStruct {
                                    ui_response,
                                    scales: ViewportScale::from_image_size_and_viewport_size(
                                        view.intrinsics().image_size(),
                                        adjusted_size,
                                    ),
                                    z_image: None,
                                    view_port_size: adjusted_size.image_size(),
                                },
                            );
                        }
                        View::Plot(_) => {
                            // no-op
                        }
                    }
                }

                // for loop to show all the plots
                for (view_label, view) in self.views.iter_mut() {
                    if !view.enabled() {
                        continue;
                    }

                    let view_aspect_ratio = view.aspect_ratio();
                    let adjusted_size =
                        get_adjusted_view_size(view_aspect_ratio, max_width, max_height);
                    match view {
                        View::Scene(_) => {
                            // no-op
                        }
                        View::Image(_) => {
                            // no-op
                        }
                        View::Plot(view) => {
                            Self::show_plot(ui, view, adjusted_size, view_label.clone());
                        }
                    }
                }
            });
        });
    }

    fn show_plot(ui: &mut Ui, view: &mut PlotView, adjusted_size: ViewportSize, plot_name: String) {
        let plot = egui_plot::Plot::new(plot_name)
            .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
            .height(adjusted_size.height)
            .width(adjusted_size.width);

        fn color_cnv(color: sophus_renderer::renderables::Color) -> egui::Color32 {
            egui::Color32::from_rgb(
                (color.r * 255.0).clamp(0.0, 255.0) as u8,
                (color.g * 255.0).clamp(0.0, 255.0) as u8,
                (color.b * 255.0).clamp(0.0, 255.0) as u8,
            )
        }
        fn show_vec<const N: usize>(curve_name: &str, g: &CurveVec<N>, plot_ui: &mut PlotUi) {
            if let Some(v_line) = &g.v_line {
                plot_ui.add(
                    VLine::new(v_line.name.clone(), v_line.x)
                        .color(egui::Color32::from_rgb(255, 255, 255)),
                );
            }
            let mut points = vec![];
            for _ in 0..N {
                points.push(Vec::new());
            }

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
                    for (i, p) in points.iter().enumerate().take(N) {
                        let plot_points = egui_plot::PlotPoints::Owned(p.clone());
                        plot_ui.line(
                            egui_plot::Line::new(format!("{curve_name}-{i}"), plot_points)
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
                plot_ui.add(
                    VLine::new(v_line.name.clone(), v_line.x)
                        .color(egui::Color32::from_rgb(255, 255, 255)),
                );
            }
            let mut points = vec![];
            let mut up_points = vec![];
            let mut down_points = vec![];
            for _ in 0..N {
                points.push(Vec::new());
                up_points.push(Vec::new());
                down_points.push(Vec::new());
            }

            for (x, (y, e)) in &g.data {
                for i in 0..N {
                    points[i].push(egui_plot::PlotPoint::new(*x, y[i]));
                    up_points[i].push(egui_plot::PlotPoint::new(*x, y[i] + e[i]));
                    down_points[i].push(egui_plot::PlotPoint::new(*x, y[i] - e[i]));
                }
            }

            let mut plot_points =
                |points: Vec<Vec<egui_plot::PlotPoint>>, color: [Color; N], style: LineStyle| {
                    for (i, p) in points.iter().enumerate().take(N) {
                        let plot_points = egui_plot::PlotPoints::Owned(p.clone());
                        plot_ui.line(
                            egui_plot::Line::new(format!("{curve_name}-{i}"), plot_points)
                                .color(color_cnv(color[i]))
                                .style(style),
                        );
                    }
                };

            plot_points(points, g.style.colors, LineStyle::Solid);
            plot_points(up_points, g.style.colors, LineStyle::dashed_dense());
            plot_points(down_points, g.style.colors, LineStyle::dashed_dense());
        }

        ui.add_sized(
            [adjusted_size.width, adjusted_size.height],
            |ui: &mut egui::Ui| {
                plot.show(ui, |plot_ui| {
                    for (curve_name, graph_data) in &mut view.curves {
                        if !graph_data.show_graph {
                            continue;
                        }

                        match &graph_data.curve {
                            GraphType::Scalar(g) => {
                                if let Some(v_line) = &g.v_line {
                                    plot_ui.add(
                                        VLine::new(v_line.name.clone(), v_line.x)
                                            .color(egui::Color32::from_rgb(255, 255, 255)),
                                    );
                                }
                                let mut points = vec![];

                                for (x, y) in &g.data {
                                    points.push(egui_plot::PlotPoint::new(*x, *y));
                                }

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
                            GraphType::Vec2(g) => {
                                show_vec(curve_name, g, plot_ui);
                            }
                            GraphType::Vec3(g) => {
                                show_vec(curve_name, g, plot_ui);
                            }
                            GraphType::Vec2Conf(g) => {
                                show_vec_conf(curve_name, g, plot_ui);
                            }
                            GraphType::Vec3Conf(g) => {
                                show_vec_conf(curve_name, g, plot_ui);
                            }
                        }
                    }
                })
                .response
            },
        );
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
