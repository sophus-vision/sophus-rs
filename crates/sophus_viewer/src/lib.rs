#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![no_std]
#![cfg_attr(nightly, feature(doc_auto_cfg))]
//! Simple viewer for 2D and 3D visualizations.

/// Interactions
pub mod interactions;
/// The view packets.
pub mod packets;
/// eframea app impl
pub mod simple_viewer;
/// Viewer base
pub mod viewer_base;
/// The view struct.
pub mod views;

/// eframe native options - recommended for use with the sophus
pub fn recommened_eframe_native_options() -> eframe::NativeOptions {
    eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([850.0, 480.0]),
        renderer: eframe::Renderer::Wgpu,
        multisampling: sophus_renderer::types::DOG_MULTISAMPLE_COUNT as u16,
        ..Default::default()
    }
}

/// prelude
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
