use eframe::egui;
use sophus_lie::Isometry3F64;

/// plugin
pub trait IsUiPlugin: std::marker::Sized {
    /// update left panel
    fn update_left_panel(&mut self, _left_ui: &mut egui::Ui, _ctx: &egui::Context) {}

    /// view response
    fn process_view2d_response(&mut self, _view_name: &str, _response: &egui::Response) {}

    /// view 3d response
    fn process_view3d_response(
        &mut self,
        _view_name: &str,
        _response: &egui::Response,
        _scene_from_camera: &Isometry3F64,
    ) {
    }
}

/// Null plugin
pub struct NullPlugin {}

impl IsUiPlugin for NullPlugin {}
