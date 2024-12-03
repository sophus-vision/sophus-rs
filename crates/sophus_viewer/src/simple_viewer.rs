use crate::packets::Packet;
use crate::preludes::*;
use crate::viewer_base::ViewerBase;
use crate::viewer_base::ViewerBaseConfig;
use sophus_renderer::RenderContext;
use eframe::egui;
use thingbuf::mpsc::blocking::Receiver;

/// Simple viewer
pub struct SimpleViewer {
    base: ViewerBase,
}

impl SimpleViewer {
    /// Create a new simple viewer
    pub fn new(
        render_state: RenderContext,
        message_recv: Receiver<Vec<Packet>>,
    ) -> Box<SimpleViewer> {
        Box::new(SimpleViewer {
            base: ViewerBase::new(render_state, ViewerBaseConfig { message_recv }),
        })
    }
}

impl eframe::App for SimpleViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.base.update_data();

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            self.base.update_top_bar(ui, ctx);
        });

        egui::SidePanel::left("left").show(ctx, |ui| {
            self.base.update_left_panel(ui, ctx);
        });

        egui::TopBottomPanel::bottom("bottom")
            .max_height(32.0)
            .show(ctx, |ui| {
                self.base.update_bottom_status_bar(ui, ctx);
            });

        // central pane must be always created last
        egui::CentralPanel::default().show(ctx, |ui| {
            self.base.update_central_panel(ui, ctx);
        });

        self.base.process_events();

        ctx.request_repaint();
    }
}
