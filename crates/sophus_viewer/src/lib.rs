#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, feature(doc_auto_cfg))]

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

mod interactions;

pub use interactions::{
    inplane_interaction::*,
    orbit_interaction::*,
    *,
};

/// The view packets.
pub mod packets;
/// Views.
pub mod views;
/// sophus_viewer prelude.
///
/// It is recommended to import this prelude when working with `sophus_viewer types:
///
/// ```
/// use sophus_viewer::prelude::*;
/// ```
///
/// or
///
/// ```ignore
/// use sophus::prelude::*;
/// ```
///
/// to import all preludes when using the `sophus` umbrella crate.
pub mod prelude {
    pub(crate) use alloc::{
        boxed::Box,
        collections::{
            btree_map::BTreeMap,
            vec_deque::VecDeque,
        },
        string::{
            String,
            ToString,
        },
        vec::Vec,
    };

    pub use sophus_autodiff::prelude::*;
    pub use sophus_image::prelude::*;
    pub use sophus_lie::prelude::*;
    pub use sophus_opt::prelude::*;

    extern crate alloc;
}

mod simple_viewer;
mod viewer_base;

pub use simple_viewer::*;
pub use viewer_base::*;
pub use views::*;

/// eframe native options - recommended for use with the sophus
pub fn recommened_eframe_native_options() -> eframe::NativeOptions {
    eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([850.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,
        multisampling: sophus_renderer::SOPHUS_RENDER_MULTISAMPLE_COUNT as u16,
        ..Default::default()
    }
}
