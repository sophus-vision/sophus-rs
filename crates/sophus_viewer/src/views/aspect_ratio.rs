use linked_hash_map::LinkedHashMap;
use sophus_image::ImageSize;

use crate::views::View;

pub(crate) trait HasAspectRatio {
    fn aspect_ratio(&self) -> f32;
}

pub(crate) fn get_median_aspect_ratio_and_num(
    views: &LinkedHashMap<String, View>,
) -> Option<(f32, usize)> {
    let mut aspect_ratios = std::vec::Vec::with_capacity(views.len());
    for (_label, widget) in views.iter() {
        if widget.enabled() {
            aspect_ratios.push(widget.aspect_ratio());
        }
    }
    aspect_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = aspect_ratios.len();
    if n == 0 {
        return None;
    }
    if n % 2 == 1 {
        Some((aspect_ratios[n / 2], n))
    } else {
        Some((
            0.5 * aspect_ratios[n / 2] + 0.5 * aspect_ratios[n / 2 - 1],
            n,
        ))
    }
}

pub(crate) fn get_max_size(
    views: &LinkedHashMap<String, View>,
    available_width: f32,
    available_height: f32,
) -> Option<(f32, f32)> {
    if let Some((median_aspect_ratio, n)) = get_median_aspect_ratio_and_num(views) {
        let mut max_width = 0.0;
        let mut max_height = 0.0;

        for num_cols in 1..=n {
            let num_rows: f32 = ((n as f32) / (num_cols as f32)).ceil();

            let w: f32 = available_width / (num_cols as f32);
            let h = (w / median_aspect_ratio).min(available_height / num_rows);
            let w = median_aspect_ratio * h;
            if w > max_width {
                max_width = w;
                max_height = h;
            }
        }

        return Some((max_width, max_height));
    }
    None
}

#[derive(Clone, Copy)]
pub(crate) struct ViewportSize {
    pub(crate) width: f32,
    pub(crate) height: f32,
}

impl ViewportSize {
    pub(crate) fn image_size(&self) -> ImageSize {
        ImageSize {
            width: self.width.ceil() as usize,
            height: self.height.ceil() as usize,
        }
    }
}

pub(crate) fn get_adjusted_view_size(
    view_aspect_ratio: f32,
    max_width: f32,
    max_height: f32,
) -> ViewportSize {
    let width = max_width.min(max_height * view_aspect_ratio);
    let height = max_height.min(max_width / view_aspect_ratio);
    ViewportSize { width, height }
}
