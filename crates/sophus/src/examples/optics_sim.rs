use std::sync::Arc;

use log::warn;
use sophus_autodiff::linalg::VecF64;
use sophus_geo::Circle;
use sophus_image::{
    ImageSize,
    IsIntensityArcImage,
    MutImageF32,
};
use sophus_lie::Isometry3F64;
use sophus_opt::{
    nlls::{
        CostFn,
        CostTerms,
        OptParams,
        optimize_nlls,
    },
    prelude::{
        IsImageView,
        IsMutImageView,
    },
    variables::{
        VarBuilder,
        VarFamily,
        VarKind,
    },
};
use sophus_renderer::{
    camera::{
        RenderCamera,
        RenderCameraProperties,
    },
    renderables::{
        Color,
        ImageFrame,
    },
};
use sophus_viewer::packets::{
    ImageViewPacket,
    Packet,
    append_to_scene_packet,
    create_scene_packet,
    delete_image_packet,
    delete_scene_packet,
};
use thingbuf::mpsc::blocking::Sender;

use crate::examples::optics_sim::{
    aperture_stop::ApertureStop,
    convex_lens::{
        BiConvexLens2,
        BiConvexLens2F64,
    },
    cost::ChiefRayCost,
    detector::Detector,
    element::Element,
    light_path::LightPath,
    scene_point::ScenePoints,
};

/// Aperture stop in an optical system.
pub mod aperture_stop;
/// BiConvex lens in an optical system.
pub mod convex_lens;
/// Cost terms for the optimization process.
pub mod cost;
/// Detector in an optical system.
pub mod detector;
/// Elements in an optical system.
pub mod element;
/// The light path of rays through the optical system.
pub mod light_path;
/// Scene points in the optical system.
pub mod scene_point;

/// Refines the angle of the chief ray using a nonlinear least squares optimization.
pub fn refine_chief_ray_angle(
    angle: f64,
    scene_point: VecF64<2>,
    lens: Arc<BiConvexLens2F64>,
    target_point: VecF64<2>,
) -> f64 {
    let family: VarFamily<VecF64<1>> = VarFamily::new(VarKind::Free, vec![VecF64::<1>::new(angle)]);
    let variables = VarBuilder::new().add_family("angle", family).build();
    let mut chief_ray_cost = CostTerms::new(["angle"], vec![]);
    chief_ray_cost.collection.push(ChiefRayCost {
        scene_point,
        aperture: target_point,
        entity_indices: [0],
    });
    let solution = optimize_nlls(
        variables,
        vec![CostFn::new_boxed(lens, chief_ray_cost)],
        OptParams {
            num_iterations: 100,
            initial_lm_damping: 1.0,
            parallelize: true,
            solver: Default::default(),
        },
    )
    .unwrap();
    solution.variables.get_members::<VecF64<1>>("angle")[0][0]
}

/// Optics elements.
pub struct OpticsElements {
    /// Lenses.
    pub lens: Arc<BiConvexLens2<f64, 0, 0>>,
    /// Detector - aka sensor plane.
    pub detector: Detector,
    /// Camera aperture.
    pub aperture: ApertureStop,
    /// Scene points.
    pub scene_points: ScenePoints,
}

impl Default for OpticsElements {
    fn default() -> Self {
        Self::new()
    }
}

impl OpticsElements {
    /// Create set of optics elements for the sim.
    pub fn new() -> Self {
        OpticsElements {
            detector: Detector { x: 0.5 },
            lens: Arc::new(BiConvexLens2::new(
                Circle {
                    center: VecF64::<2>::new(0.560, 0.0),
                    radius: 0.600,
                },
                Circle {
                    center: VecF64::<2>::new(-0.560, 0.0),
                    radius: 0.600,
                },
                1.5,
            )),
            scene_points: ScenePoints {
                p: [VecF64::<2>::new(-2.5, 0.1), VecF64::<2>::new(-2.0, -0.1)],
            },
            aperture: ApertureStop {
                x: 0.2,
                radius: 0.02,
            },
        }
    }
}

/// Optics sim widget
pub struct OpticsSimWidget {
    /// Optics elements.
    pub elements: OpticsElements,
    message_send: Sender<Vec<Packet>>,
}

impl Drop for OpticsSimWidget {
    fn drop(&mut self) {
        match self.message_send.send(vec![
            delete_scene_packet("scene"),
            delete_image_packet("image"),
        ]) {
            Ok(_) => {}
            Err(_) => {
                warn!("Failed to send delete packets, viewer might not be running.");
            }
        }
    }
}

impl OpticsSimWidget {
    /// Create a new optics sim.
    pub fn new(
        message_send: Sender<std::vec::Vec<sophus_viewer::packets::Packet>>,
    ) -> OpticsSimWidget {
        let packets = vec![create_scene_packet(
            "scene",
            RenderCamera {
                scene_from_camera: Isometry3F64::from_translation(VecF64::<3>::new(
                    -0.5, 0.0, -3.0,
                )),
                properties: RenderCameraProperties::default_from(ImageSize {
                    width: 640,
                    height: 480,
                }),
            },
            true,
        )];
        message_send.send(packets).unwrap();

        OpticsSimWidget {
            elements: OpticsElements::new(),
            message_send,
        }
    }

    /// Update the optics sim.
    pub fn send_update(&mut self) {
        let mut light_path = vec![];

        let mut image = MutImageF32::from_image_size(ImageSize {
            width: 15,
            height: 50,
        });

        for i in 0..2 {
            let top_angle = refine_chief_ray_angle(
                0.0,
                self.elements.scene_points.p[i],
                self.elements.lens.clone(),
                VecF64::<2>::new(self.elements.aperture.x, self.elements.aperture.radius),
            );
            let center_angle = refine_chief_ray_angle(
                0.0,
                self.elements.scene_points.p[i],
                self.elements.lens.clone(),
                VecF64::<2>::new(self.elements.aperture.x, 0.0),
            );
            let bottom_angle = refine_chief_ray_angle(
                0.0,
                self.elements.scene_points.p[i],
                self.elements.lens.clone(),
                VecF64::<2>::new(self.elements.aperture.x, -self.elements.aperture.radius),
            );

            for angle in [
                ("top", top_angle),
                ("center", center_angle),
                ("bottom", bottom_angle),
            ] {
                light_path.push(LightPath::from(
                    format!("path_{}_{}", i, angle.0),
                    self.elements.scene_points.p[i],
                    angle.1,
                    self.elements.detector.clone(),
                    self.elements.aperture.clone(),
                    angle.0 == "center",
                    if i == 0 { Color::red() } else { Color::blue() },
                ));
            }

            let angle_range = top_angle - bottom_angle;
            for j in 0..100 {
                let angle = bottom_angle + j as f64 * 0.01 * angle_range;

                let mut path = LightPath::from(
                    format!("path_{i}_{j}"),
                    self.elements.scene_points.p[i],
                    angle,
                    self.elements.detector.clone(),
                    self.elements.aperture.clone(),
                    false,
                    if i == 0 { Color::red() } else { Color::blue() },
                );

                path.propagate(&self.elements.lens);

                if let Some(point) = path.image_point {
                    let pixel = (point * 90.0 + 25.0).round() as usize;

                    if (0..image.image_size().height).contains(&pixel) {
                        *image.mut_pixel(0, pixel) += 0.1 * angle_range as f32;
                    }
                }
            }
        }

        for path in light_path.iter_mut() {
            path.propagate(&self.elements.lens);
        }

        let mut renderables3d = vec![
            self.elements.detector.to_renderable3(),
            self.elements.scene_points.to_renderable3(),
            self.elements.lens.to_renderable3(),
            self.elements.aperture.to_renderable3(),
        ];

        for path in light_path {
            renderables3d.push(path.to_renderable3());
        }

        let packets = vec![
            append_to_scene_packet("scene", renderables3d),
            Packet::Image(ImageViewPacket {
                view_label: "image".to_owned(),
                frame: Some(ImageFrame::from_image(&image.to_shared().to_rgba())),
                scene_renderables: vec![],
                pixel_renderables: vec![],
                delete: false,
            }),
        ];

        self.message_send.send(packets).unwrap();
    }
}
