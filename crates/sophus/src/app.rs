use eframe::egui;
use sophus_autodiff::points::example_points;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct TemplateApp {
    // Example stuff:
    label: String,

    #[serde(skip)] // This how you opt-out of serialization of a field
    value: f32,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            // Example stuff:
            label: "Hello World!!!!".to_owned(),
            value: 2.7,
        }
    }
}

use sophus_autodiff::{
    dual::DualVector,
    linalg::{
        MatF64,
        VecF64,
    },
};
use sophus_lie::{
    Isometry2,
    Isometry2F64,
    Rotation2F64,
};
use sophus_opt::{
    nlls::{
        CostFn,
        CostTerms,
        EvaluatedCostTerm,
        OptParams,
        optimize_nlls,
    },
    prelude::*,
    robust_kernel,
    variables::{
        VarBuilder,
        VarFamily,
        VarKind,
    },
};

/// We want to fit the isometry `T ∈ SE(2)` to a prior distribution
/// `N(E(T), W⁻¹)`, where `E(T)` is the prior mean and `W⁻¹` is the prior
/// covariance matrix.

/// (1) First we define the residual cost term.
#[derive(Clone, Debug)]
pub struct Isometry2PriorCostTerm {
    /// Prior mean, `E(T)` of type [Isometry2F64].
    pub isometry_prior_mean: Isometry2F64,
    /// `W`, which is the inverse of the prior covariance matrix.
    pub isometry_prior_precision: MatF64<3, 3>,
    /// We only have one variable, so this will be `[0]`.
    pub entity_indices: [usize; 1],
}

impl Isometry2PriorCostTerm {
    /// (2) Then we define  residual function for the cost term:
    ///
    /// `g(T) = log[T * E(T)⁻¹]`
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        isometry: Isometry2<Scalar, 1, DM, DN>,
        isometry_prior_mean: Isometry2<Scalar, 1, DM, DN>,
    ) -> Scalar::Vector<3> {
        (isometry * isometry_prior_mean.inverse()).log()
    }
}

/// (3) Implement the `HasResidualFn` trait for the cost term.
impl HasResidualFn<3, 1, (), Isometry2F64> for Isometry2PriorCostTerm {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: Isometry2F64,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<robust_kernel::RobustKernel>,
    ) -> EvaluatedCostTerm<3, 1> {
        let isometry: Isometry2F64 = args;

        let residual = Self::residual(isometry, self.isometry_prior_mean);
        let dx_res_fn = |x: DualVector<3, 3, 1>| -> DualVector<3, 3, 1> {
            Self::residual(
                Isometry2::exp(x) * isometry.to_dual_c(),
                self.isometry_prior_mean.to_dual_c(),
            )
        };

        (|| dx_res_fn(DualVector::var(VecF64::<3>::zeros())).jacobian(),).make(
            idx,
            var_kinds,
            residual,
            robust_kernel,
            Some(self.isometry_prior_precision),
        )
    }
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        let points = example_points::<f64, 4, 1, 0, 0>();

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }

        Default::default()
    }
}

impl eframe::App for TemplateApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or
        // `Area`. For inspiration and more examples, go to https://emilk.github.io/egui

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::menu::bar(ui, |ui| {
                // NOTE: no File->Quit on web pages!
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }

                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            ui.heading("eframe template");

            let prior_world_from_robot = Isometry2F64::from_translation(VecF64::<2>::new(1.0, 2.0));

            // (4) Define the cost terms, by specifying the residual function
            // `g(T) = Isometry2PriorCostTerm` as well as providing the prior distribution.
            const POSE: &str = "poses";
            let obs_pose_a_from_pose_b_poses = CostTerms::new(
                [POSE],
                vec![Isometry2PriorCostTerm {
                    isometry_prior_mean: prior_world_from_robot,
                    isometry_prior_precision: MatF64::<3, 3>::identity(),
                    entity_indices: [0],
                }],
            );

            // (5) Define the decision variables. In this case, we only have one variable,
            // and we initialize it with the identity transformation.
            let est_world_from_robot = Isometry2F64::identity();
            let variables = VarBuilder::new()
                .add_family(
                    POSE,
                    VarFamily::new(VarKind::Free, vec![est_world_from_robot]),
                )
                .build();

            // (6) Perform the non-linear least squares optimization.
            let solution = optimize_nlls(
                variables,
                vec![CostFn::new_boxed((), obs_pose_a_from_pose_b_poses.clone())],
                OptParams::default(),
            )
            .unwrap();

            // (7) Retrieve the refined transformation and compare it with the prior one.
            let refined_world_from_robot = solution.variables.get_members::<Isometry2F64>(POSE)[0];

            ui.label(format!("Write something: {:?}", refined_world_from_robot));

            ui.horizontal(|ui| {
                ui.label("Write something: ");
                ui.text_edit_singleline(&mut self.label);
            });

            ui.add(egui::Slider::new(&mut self.value, 0.0..=10.0).text("value"));
            if ui.button("Increment").clicked() {
                self.value += 1.0;
            }

            ui.separator();

            ui.add(egui::github_link_file!(
                "https://github.com/emilk/eframe_template/blob/main/",
                "Source code."
            ));

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                powered_by_egui_and_eframe(ui);
                egui::warn_if_debug_build(ui);
            });
        });
    }
}

fn powered_by_egui_and_eframe(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label("Powered by ");
        ui.hyperlink_to("egui", "https://github.com/emilk/egui");
        ui.label(" and ");
        ui.hyperlink_to(
            "eframe",
            "https://github.com/emilk/egui/tree/master/crates/eframe",
        );
        ui.label(".");
    });
}
