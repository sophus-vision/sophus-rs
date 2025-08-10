use eframe::egui;
use sophus_image::ImageSize;

/// A rectangle representing the position and size of a window.
#[derive(Clone, Debug)]
pub struct WindowPlacement {
    /// The label of the window, used for identification.
    pub view_label: String,
    /// The rectangle representing the position and size of the window.
    pub rect: egui::Rect,
}

impl WindowPlacement {
    pub(crate) fn viewport_size(&self) -> ImageSize {
        ImageSize {
            width: self.rect.width().ceil() as usize,
            height: self.rect.height().ceil() as usize,
        }
    }
}

/// A window occupying a portion of the available space.
#[derive(Clone, Debug)]
pub struct WindowArea {
    /// The label of the window, used for identification.
    pub view_label: String,
    /// aspect ratio of the window, i.e. width / height
    pub width_by_height_ratio: f32,
}

impl WindowArea {
    /// The border size around the window.
    pub const BORDER: f32 = 14.0;
    /// The size of the bar at the top of the window.
    pub const BAR_SIZE: f32 = 34.0;

    fn size_from_height(&self, height: f32) -> egui::Vec2 {
        egui::Vec2::new(self.width_by_height_ratio * height, height)
    }

    fn height(boxes: &[WindowArea], total_width: f32) -> f32 {
        let width_by_height_sum = boxes
            .iter()
            .map(|b| b.width_by_height_ratio as f64)
            .sum::<f64>() as f32;
        total_width / width_by_height_sum
    }

    fn greedy_flow_layout_from_height(
        boxes: &[WindowArea],
        h: f32,
        clip: egui::Rect,
    ) -> Option<Vec<WindowPlacement>> {
        // Start at the top-left.
        let mut x_offset = clip.min.x;
        let mut y_offset = clip.min.y;
        let mut res = vec![];

        for b in boxes.iter() {
            let size = b.size_from_height(h);
            let w = size.x;

            // If the next box would overflow the right edge , wrap to a new row.
            if x_offset + w + Self::BORDER > clip.max.x {
                x_offset = clip.min.x;
                y_offset += h + Self::BORDER + Self::BAR_SIZE;
            }

            // If placing the box would overflow the bottom edge, return None.
            if y_offset + h + Self::BORDER + Self::BAR_SIZE > clip.max.y
                || x_offset + Self::BORDER + w > clip.max.x
            {
                return None;
            }

            res.push(WindowPlacement {
                view_label: b.view_label.clone(),
                rect: egui::Rect::from_min_size(
                    egui::Pos2::new(x_offset, y_offset),
                    egui::Vec2::new(w + Self::BORDER, h + Self::BORDER + Self::BAR_SIZE),
                ),
            });

            x_offset += w + Self::BORDER;
        }

        Some(res)
    }

    /// Calculates the layout of the boxes in a way that they fit into the available space.
    pub fn flow_layout(boxes: &[WindowArea], ui: &egui::Ui) -> Vec<WindowPlacement> {
        if boxes.is_empty() {
            return vec![];
        }

        // Finds the largest height that still allows all boxes to fit into ui.clip_rect() by
        // binary searching:

        // Available packing area.
        let rect = ui.clip_rect();

        let total_width = rect.size().x;

        // Lower and upper bounds for the common height of the boxes.
        let mut lower = (0.95 * WindowArea::height(boxes, total_width))
            .min(ui.clip_rect().size().y)
            - Self::BORDER * 2.0
            - Self::BAR_SIZE;
        println!("boxes {}", boxes.len());
        let mut upper = rect.size().y;

        // We assume that lower is always a valid height, and upper is never valid.
        assert!(WindowArea::greedy_flow_layout_from_height(boxes, lower, rect).is_some());
        assert!(WindowArea::greedy_flow_layout_from_height(boxes, upper, rect).is_none());

        const MAX_ITERATIONS: usize = 20;

        for _i in 0..MAX_ITERATIONS {
            let trial_height = lower + (upper - lower) * 0.5;

            let boxes = WindowArea::greedy_flow_layout_from_height(boxes, trial_height, rect);
            if boxes.is_none() {
                // If the boxes do not fit, set upper to trial_height.
                upper = trial_height;
            } else {
                // If the boxes fit, set lower to trial_height.
                lower = trial_height;
            }
        }
        WindowArea::greedy_flow_layout_from_height(boxes, lower, rect).unwrap()
    }
}

pub(crate) fn show_image(
    ctx: &egui::Context,
    egui_texture: egui::TextureId,
    placement: &WindowPlacement,
) -> (egui::Response, bool) {
    let mut enabled = true;

    let r = egui::Window::new(placement.view_label.clone())
        .movable(false)
        .title_bar(true)
        .open(&mut enabled)
        .collapsible(false)
        .fixed_pos(placement.rect.min)
        .fixed_size(egui::Vec2::new(
            placement.rect.width(),
            placement.rect.height(),
        ))
        .show(ctx, |ui| {
            ui.add(
                egui::Image::new(egui::load::SizedTexture {
                    size: placement.rect.size(),
                    id: egui_texture,
                })
                .corner_radius(5)
                .maintain_aspect_ratio(false)
                .fit_to_exact_size(egui::Vec2::new(
                    placement.rect.width() - WindowArea::BORDER,
                    placement.rect.height() - WindowArea::BORDER - WindowArea::BAR_SIZE,
                )),
            )
        })
        .unwrap()
        .inner
        .unwrap();

    (r, !enabled)
}
