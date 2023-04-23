use std::fmt::Display;

use crate::image::dyn_arc_image::IntensityImage;
use crate::image::layout::ImageSize;
use linked_hash_map::LinkedHashMap;
use notan::egui::{self, *};
use notan::prelude::*;

use super::micro_widgets::{
    Button, EnumStringRepr, MicroWidget, MicroWidgetType, Number, RangedVar, Var, VarType,
};

pub struct ContextParams {
    pub title: String,
    pub window_size: ImageSize,
}

pub struct Shared<T: Sized> {
    inner: std::sync::Arc<std::sync::Mutex<T>>,
}

#[uniform]
#[derive(Copy, Clone)]
pub struct VBlock {
    viewport_size: [f32; 2],
}

#[uniform]
#[derive(Copy, Clone)]
pub struct FBlock {
    pub color1: [f32; 4],
    pub color2: [f32; 4],
    pub checksize: i32,
}

pub struct SharedContextState {
    pub params: ContextParams,
    pub panel_widgets: LinkedHashMap<String, MicroWidgetType>,

    layout: Option<IntensityImage>,
    pub vblock: VBlock,
    pub fblock: FBlock,
}

impl<T> Shared<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Shared<T> {
    pub fn new(t: T) -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(t)),
        }
    }

    pub fn lock(&self) -> std::sync::MutexGuard<T> {
        self.inner.lock().unwrap()
    }
}

// make this copyable
pub struct Context {
    pub shared_state: Shared<SharedContextState>,
}

impl Context {
    pub fn from_params(params: ContextParams) -> Self {
        Context {
            shared_state: Shared::new(SharedContextState {
                params,
                panel_widgets: LinkedHashMap::new(),
                layout: Option::None,
                vblock: VBlock {
                    viewport_size: [320.0, 320.0],
                },
                fblock: FBlock {
                    color1: [1.0, 0.0, 0.0, 1.0],
                    color2: [0.0, 1.0, 0.0, 1.0],
                    checksize: 10,
                },
            }),
        }
    }

    pub fn set_layout(&mut self, img: IntensityImage) {
        self.shared_state.lock().layout = Some(img);
    }

    pub fn run_loop(self, callback: Box<dyn Fn(Shared<SharedContextState>)>) -> Result<(), String> {
        let win = WindowConfig::default()
            .lazy_loop(false)
            .vsync(true)
            .high_dpi(true);

        let c = self.shared_state.clone();
        let setup = |gfx: &mut Graphics| State::new(gfx, c, callback);

        notan::init_with(setup)
            .add_config(win)
            .add_config(EguiConfig)
            .draw(draw)
            .build()
    }

    pub fn add_panel_widget(&mut self, name: &str, widget: MicroWidgetType) {
        self.shared_state
            .lock()
            .panel_widgets
            .insert(name.to_string(), widget);
    }

    pub fn add_var<T: Number>(&mut self, name: &str, value: T) {
        self.add_panel_widget(name, MicroWidgetType::Var(T::from_var(Var { value })));
    }

    pub fn add_ranged_var<T: Number>(&mut self, name: &str, value: T, min: T, max: T) {
        self.add_panel_widget(
            name,
            MicroWidgetType::RangedVar(T::from_ranged_var(RangedVar {
                value,
                min_max: (min, max),
            })),
        );
    }

    pub fn add_button(&mut self, name: &str) {
        self.add_panel_widget(name, MicroWidgetType::Button(Button { pressed: false }));
    }

    pub fn add_enum_string_repr(&mut self, name: &str, value: &str, options: Vec<String>) {
        self.add_panel_widget(
            name,
            MicroWidgetType::EnumStringRepr(EnumStringRepr {
                value: value.to_string(),
                values: options,
            }),
        );
    }
}

#[derive(AppState)]
pub struct State {
    triangle: Triangle,
    pub layout: Option<IntensityImage>,
    shared_state: Shared<SharedContextState>,
    callback: Box<dyn Fn(Shared<SharedContextState>)>,

    pub update_receiver: std::sync::mpsc::Receiver<(String, MicroWidgetType)>,
    pub update_sender: std::sync::mpsc::Sender<(String, MicroWidgetType)>,
}

impl State {
    pub fn new(
        gfx: &mut Graphics,
        shared_state: Shared<SharedContextState>,
        callback: Box<dyn Fn(Shared<SharedContextState>)>,
    ) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();

        Self {
            triangle: Triangle::new(gfx, shared_state.clone()),
            layout: Option::None,
            shared_state,
            callback,
            update_receiver: receiver,
            update_sender: sender,
        }
    }
}

fn draw(gfx: &mut Graphics, plugins: &mut Plugins, state: &mut State) {
    let s = state.shared_state.clone();

    state.callback.as_mut()(s);

    let mut output = plugins.egui(|ctx| {
        egui::SidePanel::left("my_left_panel").show(ctx, |ui| {
            let mut s = state.shared_state.lock();
            for (name, widget) in s.panel_widgets.iter_mut() {
                widget.add_to_egui(ui, name.to_string(), &mut state.update_sender.clone());
            }
        });

        state.update_receiver.try_iter().for_each(|widget| {
            let mut s = state.shared_state.lock();
            // w.update(label,  s.panel_widgets);
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                let (rect, _response) =
                    ui.allocate_exact_size(egui::Vec2::splat(300.0), egui::Sense::drag());
                //state.angle += response.drag_delta().x * 0.01;

                // Pass as callback the triangle to be draw
                let triangle = state.triangle.clone();
                let vblock = state.shared_state.lock().vblock;
                let fblock = state.shared_state.lock().fblock;

                let cb = EguiCallbackFn::new(move |info, device| {
                    triangle.draw(device, info, vblock, fblock);
                });

                let callback = egui::PaintCallback {
                    rect,
                    callback: std::sync::Arc::new(cb),
                };

                ui.painter().add(callback);
            });

            ui.label("Drag to rotate!");
        });
    });

    output.clear_color(Color::BLACK);

    gfx.render(&output);
}

// --- TRIANGLE
//language=glsl
const VERT: ShaderSource = notan::vertex_shader! {
    r#"
    #version 450
    layout(location = 0) out vec2 pix;


    layout(set = 0, binding = 0) uniform VBlock {
        vec2 viewport_size;
    } block;


    const vec2 pos[4] = vec2[4](
        vec2( -1.0, +1.0), vec2( -1.0, -1.0),
        vec2( +1.0, +1.0), vec2( +1.0, -1.0)
      );

    void main() {
        vec2 ndc = pos[gl_VertexIndex];

        pix = (ndc + vec2(1.0))/2.0 * block.viewport_size;
        gl_Position = vec4(ndc, 0.0, 1.0);
    }
    "#
};

//language=glsl
const FRAG: ShaderSource = notan::fragment_shader! {
    r#"
    #version 450
    precision mediump float;

    layout(location = 0) in vec2 pix;

    layout(set = 0, binding = 0) uniform FBlock {
        vec4 color1;
        vec4 color2;
        int checksize;
    } block;
    
   

    layout(location = 0) out vec4 color;

    void main() {
        bool x = (int(pix.x)/block.checksize % 2) == 0;
        bool y = (int(pix.y)/block.checksize % 2) == 0;
        bool check = x ^^ y;
        color = check ? block.color1 : block.color2;
    }
    "#
};

struct Triangle {
    pipeline: Pipeline,
    vbo: Buffer,
    vblock_ubo: Buffer,
    hblock_ubo: Buffer,
    shared_state: Shared<SharedContextState>,
}

impl Clone for Triangle {
    fn clone(&self) -> Self {
        Self {
            pipeline: self.pipeline.clone(),
            vbo: self.vbo.clone(),
            vblock_ubo: self.vblock_ubo.clone(),
            hblock_ubo: self.hblock_ubo.clone(),
            shared_state: self.shared_state.clone(),
        }
    }
}

impl Triangle {
    fn new(gfx: &mut Graphics, shared_state: Shared<SharedContextState>) -> Self {
        let vertex_info = VertexInfo::new()
            .attr(0, VertexFormat::Float32x2)
            .attr(1, VertexFormat::Float32x4);

        let pipeline = gfx
            .create_pipeline()
            .from(&VERT, &FRAG)
            .with_vertex_info(&vertex_info)
            .build()
            .unwrap();

        #[rustfmt::skip]
        let vertices = [
            0.0, 1.0,   1.0, 0.2, 0.3,1.0,
            -1.0, -1.0,   0.1, 1.0, 0.3,1.0,
            1.0, -1.0,   0.1, 0.2, 1.0,1.0,
        ];

        let vbo = gfx
            .create_vertex_buffer()
            .with_info(&vertex_info)
            .with_data(&vertices)
            .build()
            .unwrap();
        print!("HEEAR");

        // let ubo = gfx
        //     .create_uniform_buffer(0, "Locals")
        //     .with_data(&[0.0])
        //     .build()
        //     .unwrap();

        let s = shared_state.lock();

        let vblock_ubo = gfx
            .create_uniform_buffer(0, "VBlock")
            .with_data(&s.vblock) // upload the transform to the gpu directly
            .build()
            .unwrap();

        let hblock_ubo = gfx
            .create_uniform_buffer(1, "HBlock")
            .with_data(&s.vblock) // upload the light object to thr gpu directly
            .build()
            .unwrap();

        Self {
            pipeline,
            vbo,
            vblock_ubo,
            hblock_ubo,
            shared_state: shared_state.clone(),
        }
    }

    fn draw(
        &self,
        device: &mut Device,
        info: egui::PaintCallbackInfo,
        vblock: VBlock,
        fblock: FBlock,
    ) {
        // update angle
        device.set_buffer_data(&self.vblock_ubo, &vblock);
        device.set_buffer_data(&self.hblock_ubo, &fblock);

        // create a new renderer
        let mut renderer = device.create_renderer();

        // set scissors using the clip_rect passed by egui
        renderer.set_scissors(
            info.clip_rect.min.x,
            info.clip_rect.min.y,
            info.clip_rect.width(),
            info.clip_rect.height(),
        );

        // start the pass
        renderer.begin(None);

        // set the viewport using the rect passed by egui
        renderer.set_viewport(
            info.viewport.min.x,
            info.viewport.min.y,
            info.viewport.width(),
            info.viewport.height(),
        );

        // draw the triangle
        renderer.set_pipeline(&self.pipeline);
        renderer.bind_buffer(&self.vbo);
        renderer.draw(0, 3);
        renderer.end();

        device.render(renderer.commands());
    }
}
