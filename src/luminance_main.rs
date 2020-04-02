// // use luminance_glfw::{GlfwSurface, Action, Key, Surface as _, WindowDim, WindowOpt, WindowEvent};
// use luminance_glfw::{Action, GlfwSurface, Key, Surface as _, WindowDim, WindowEvent, WindowOpt};
// use luminance::pipeline;
// use luminance::context::GraphicsContext as _;
// use std::process::exit;
// use std::time::Instant;
// use luminance::pipeline::PipelineState;  // For low precision (yet sufficient) time points for evolving display over time
//
// fn main() {
//   // our graphics surface
//   let surface = GlfwSurface::new(
//     WindowDim::Windowed(960, 540),
//     "Hello World!  I am potato!",
//     WindowOpt::default()
//   );
//
//   // If things worked properly, start the main loop, otherwise throw an error and exit
//   match surface {
//     Ok(surface) => {
//       eprintln!("Graphics surface created!");
//       main_loop(surface);
//     }
//     Err(e) => {
//       eprintln!("Cannot create graphics surface!");
//       exit(1);
//     }
//   }
// }
//
// /// The main loop for the graphics pipeline.  Keeps looping while program is running, checking if
// /// it needs to draw new things
// fn main_loop(mut surface: GlfwSurface) {
//   let start_t = Instant::now();  // Get start time
//   let back_buffer = surface.back_buffer().unwrap();  // Get back buffer, to which we can draw
//
//   'app: loop {  // Naming this outer loop `'app` allows us to break the outer loop within the inner for loop
//     for event in surface.poll_events() {
//       match event {
//         WindowEvent::Close | WindowEvent::Key(Key::Escape, _, Action::Release, _) => break 'app,
//         _ => ()
//       }
//     }
//
//     // If we want to do any rendering stuff, put that here
//     let t = start_t.elapsed().as_millis() as f32 * 1e-3;
//     let color =[t.cos(), t.sin(), 0.5, 1.];
//
//     // GraphicsContext::pipeline_builder(&mut surface).pipeline(
//     surface.pipeline_builder().pipeline(
//       &back_buffer,
//       &PipelineState::default().set_clear_color(color),
//       |_,_| (),
//     );
//
//     surface.swap_buffers();
//   }
// }
