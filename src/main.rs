use image::{ImageBuffer, Rgba};
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::format::{ClearValue, Format};
use vulkano::image::{Dimensions, StorageImage};
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::blend::AttachmentsBlend::Collective;
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline};
use vulkano::pipeline::viewport::Viewport;
use vulkano::sync::GpuFuture;
use vulkano::descriptor::descriptor::DescriptorType::StorageBuffer;
use vulkano::swapchain::{Swapchain, SurfaceTransform, PresentMode};
use vulkano_win::VkSurfaceBuild;
use winit::{EventsLoop, WindowBuilder, VirtualKeyCode};
use std::thread::sleep;
use winit::VirtualKeyCode::NoConvert;

mod images;
mod shader;
mod mandelbrot_set;
mod vertex_shader;
mod fragment_shader;

fn main() {
    // First, get instance.  This is the foundational Vulkan "thing"
    let instance = Instance::new(
        None,
        &InstanceExtensions::none(),
        None
    ).expect("failed to create instance");

    // Now, get physical devices
    let physical = PhysicalDevice::enumerate(&instance)
        .next().expect("No device available");

    for family in physical.queue_families() {
        println!(
            "Found queue family with {:?} queue(s)",
            family.queues_count()
        );
    }

    // Now, get a family of queues from the physical devices.  A `Queue` is a a place where commands can be submitted
    // for the GPU to process (usually transfer or compute queues)
    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("Couldn't find a graphical queue family");

    // Now, just pick one of the devices from the physical devices and the queue families.  This builds a new Vulkan device
    // for the given physical device.  It then uses the `queue_family` with priority `0.5`
    let (device, mut queues) = Device::new(
        physical,
        &Features::none(),
        &DeviceExtensions::none(),
        [(queue_family, 0.5)].iter().cloned(),
    ).expect("Failed to create device");

    // For now, just grab the first queue in the iterator
    let queue = queues.next().unwrap();

    // Now, create a buffer to be able to get data to and from GPU (without this, the GPU wouldn't really be able to do anything
    let data = 12;

    // Here, the parameters are telling which device to use (we're cloning a reference to the device, not a whole device
    // object itself), how we want to use it (important for optimizations), and lastly the data with which we want to
    // fill the buffer (note that in normal usage, we would almost always be putting a lot more than 4 bytes of data in
    // the buffer when creating it)
    let data_buffer = CpuAccessibleBuffer::from_data(
        device.clone(),
        BufferUsage::all(), data
    ).expect("Failed to create buffer!");

    // We can put pretty much any type of data in a buffer, but if it doesn't implement the `Send`, `Sync`, and `Copy`
    // traits, or if it isn't `'static`, the ways we can use the buffer will be limited
    struct MyStruct {
        a: u32,
        b: bool,
    }

    let my_struct = MyStruct { a: 5, b: true };
    let my_struct_buffer = CpuAccessibleBuffer::from_data(
        device.clone(),
        BufferUsage::all(),
        my_struct
    ).unwrap();

    // If the size of an array of data that we want to put into the buffer is not known at compile time, we can use the
    // `from_iter` method
    let iter = (0..18).map(|_| 5u8);
    let array_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        iter
    ).unwrap();

    // If we want to update the content of the buffer, we use `write()`, which basically gives us a write lock
    {
        let mut my_struct_content = my_struct_buffer.write().unwrap();
        my_struct_content.a *= 2;
        my_struct_content.b = false;

        // Note that the write lock must be out of scope to read from the buffer.  We do that by enclosing the content
        // writer in its own scope.  We could also just use the `drop()` method like below, if we preferred

        // drop(my_struct_content);
    }

    {
        let mut array_content = array_buffer.write().unwrap();
        array_content[12] = 83;
        array_content[3] = 32;
    }

    // Now, try reading
    let my_struct_reader = my_struct_buffer.read().unwrap();
    println!(
        "MyStruct.a: {:?},    MyStruct.b: {:?}",
        my_struct_reader.a, my_struct_reader.b
    );

    for val in array_buffer.read().unwrap().iter() {
        println!("Value: {:?}", val);
    }

    // Now, we'll learn how to transfer data between two different buffers
    let source_content = 0..64; // creates array with values from 0 to 63 (according to position in array)
    let source_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        source_content
    ).expect("Failed to get source buffer");

    let dest_content = (0..64).map(|_| 0); // creates array of size 64 filled with all 0s
    let dest_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        dest_content
    ).expect("Failed to get destination buffer");

    // Now, to start actually moving stuff from one buffer to another, we need to set up a command buffer.  Vulkan doesn't
    // let us just run individual commands one at a time because that would be way two inefficient, so we chain them
    // together into a command buffer.  Note that the `clone()`s here are also cloning references, so it's not too expensive
    let command_buffer = AutoCommandBufferBuilder::new(
        device.clone(),
        queue_family
    ).unwrap()
        .copy_buffer(
            source_buffer.clone(),
            dest_buffer.clone()
        ).unwrap()
        .build()
        .unwrap();

    // `finished` is "an object that represents the execution of the command buffer".  It does NOT mean that the commands
    // have been executed, just that they have been submitted.  It does not wait for the execution to complete (or even
    // start).  Thus, we cannot read from the destination buffer until we have ensured that the execution has been
    // completed.  We'll use this `finished` object to do that
    let finished = command_buffer.execute(queue.clone()).unwrap();

    // This basically blocks until the command queue (the transfer) is completed
    finished.then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let source_content = source_buffer.read().unwrap();
    let dest_content = dest_buffer.read().unwrap();
    assert_eq!(&*source_content, &*dest_content);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Actually doing stuff on the GPU
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Now we'll actually start performing computations on the GPU (in parallel).  We'll start by multiplying 65536 values
    // by 12
    let data_iter = 0..65536;
    let data_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        data_iter
    ).expect("Failed to obtain data buffer");

    // Call the shader in the `shader.rs` file.  The ` vulkano_shaders::shader!` macro constructs several methods and
    // structs, including the `Shader::load()` method we use here to load the compiled shader
    let shader = shader::cs::Shader::load(device.clone()).expect("Failed to load shader!");

    // This object actually holds the compute operation that we're going to perform
    let compute_pipeline = Arc::new(
        ComputePipeline::new(
            device.clone(),
            &shader.main_entry_point(),
            &()
        ).expect("Failed to create compute pipeline"),
    );

    // Now we create a descriptor set so that the CPU can actually give the data in the buffer to the GPU.  There are
    // multiple different types of descriptor sets; I'm not yet sure why we use a `PersistentDescriptorSet`.  Anyways,
    // we pass in a clone of the pipeline for which we're creating the descriptor set, and then we pass in the set for
    // which we're creating it (0, in this case)
    let descr_set = Arc::new(
        PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
            .add_buffer(data_buffer.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    // Now we must create another command buffer for performing the operation.  Remember, we must first build the command
    // buffer by passing in the device (which GPU to use) and the queue family (which set of work queues will handle the
    // work).  Then we "dispatch" the data to it, making sure to give it the proper dimensions (there'll need to be some
    // magic numbers here and in the shader, it seems), the pipeline (which holds the computation shader), and the
    // binding to the descriptor set containing the data to be used.
    let multiplier_buffer = AutoCommandBufferBuilder::new(
        device.clone(),
        queue_family
    ).unwrap()
        .dispatch(
            [1024, 1, 1],
            compute_pipeline.clone(),
            descr_set.clone(),
            (),
        )
        .unwrap()
        .build()
        .unwrap();

    // Schedule the operation on the GPU
    let finished = multiplier_buffer.execute(queue.clone()).unwrap();

    // Wait for it to complete
    finished.then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    // Now check that operation performed properly
    let buffer_contents = data_buffer.read().unwrap();
    for (n, val) in buffer_contents.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Images
    ////////////////////////////////////////////////////////////////////////////////////////////////

    let dim = Dimensions::Dim2d {
        width: 1024,
        height: 1024,
    };
    let format = Format::R8G8B8A8Unorm;
    // vulkan_images::images(device, dim);
    // images(&mut device.clone(), dim, format, &queue);

    // Create an image object that can be used to store an image (or some other data)
    let image = StorageImage::new(
        device.clone(),
        dim,
        format,
        Some(queue.family())
    ).unwrap();

    // Note that with images, we can't read or write directly to or from the memory.  We must ask the GPU to do it.
    // Note that the format of the `ClearValue` color depends on the format of the image.  Since we used the `Unorm`
    // type, which normalizes the 0-255 integer values to floating point numbers, we use the `ClearValue::Float` type
    let command_buffer = AutoCommandBufferBuilder::new(
        device.clone(),
        queue.family()
    ).unwrap()
        .clear_color_image(
            image.clone(),
            ClearValue::Float([0.0, 0.0, 1.0, 1.0])
        ).unwrap()
        .build()
        .unwrap();

    // To actually see the image, we must now create a buffer and have the GPU write the contents of the image to the
    // buffer.  Note that since the image is 1024 x 1024 pixels, and each pixel consists of four 8-bit values (RGBA),
    // our iter goes from 0 to 1024 * 1024 * 4
    let image_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        (0..1024 * 1024 * 4).map(|_| 0u8),
    ).expect("Failed to get memory for buffer");

    // Set up the command buffer to execute
    let image_command_buffer = AutoCommandBufferBuilder::new(
        device.clone(),
        queue.family()
    ).unwrap()
        .clear_color_image(
            image.clone(),
            ClearValue::Float([0.0, 0.0, 1.0, 1.0])
        ).unwrap()
        .copy_image_to_buffer(
            image.clone(),
            image_buffer.clone()
        )
        .unwrap()
        .build()
        .unwrap();

    // Put the command buffer in line to be executed
    let finished = image_command_buffer.execute(queue.clone()).unwrap();

    // Wait until it's done executing
    finished.then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    // Save the image in the buffer to a PNG file so we can view it
    let image_buffer_content = image_buffer.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, image_buffer_content).unwrap();
    image.save("image.png").unwrap();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Mandelbrot set!
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Here's where create the image to send to the shader
    let mandelbrot_img = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: 1024,
            height: 1024
        },
        Format::R8G8B8A8Unorm,
        Some(queue.family())
    ).unwrap();

    // Now we load the shader
    let mandelbrot_shader = mandelbrot_set::mandelbrot::Shader::load(device.clone()).expect("Failed to load shader!");

    // Set up the computation pipeline, telling it which device and shader we're using
    let compute_pipeline = Arc::new(
        ComputePipeline::new(
            device.clone(),
            &mandelbrot_shader.main_entry_point(),
            &()
        ).expect("Failed to obtain compute pipeline"),
    );

    // The descriptor set binds the `StorageImage` to the shader
    let desc_set = Arc::new(
        PersistentDescriptorSet::start(
            compute_pipeline.clone(),
            0
        ).add_image(mandelbrot_img.clone()).unwrap()
        .build().unwrap()
    );

    // Create the buffer to store the output from the GPU shader
    let mandelbrot_output = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        (0..1024 * 1024 * 4).map(|_| 0u8)
    ).expect("Failed to get buffer to store GPU output");

    // Build up the command buffer so that the GPU knows what it's actually supposed to do.  This says that we want to
    // dispatch 1024 / 8 = 128 threads.  The 1024 is from the width and height of the image.  The 8 is from the fact
    // that the shader is programmed to have each thread handle an 8x8 square of pixels.  And since we only have a 2D
    // image, the z-dimension is just one
    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .dispatch(
            [1024 / 8, 1024 / 8, 1],
            compute_pipeline.clone(),
            desc_set.clone(),
            ()
        ).unwrap()
        .copy_image_to_buffer(
            mandelbrot_img.clone(),
            mandelbrot_output.clone()
        ).unwrap()
        .build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let mand_buffer_content = mandelbrot_output.read().unwrap();
    let mand_image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &mand_buffer_content[..]).unwrap();
    mand_image.save("mandelbrot_set.png").unwrap();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Graphics pipeline!!
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // First an overview.  In Vulkan, the graphics pipeline involves four primary things:
    //      1. A graphics pipeline object (similar to the compute pipeline objects we've been using)
    //      2. One or more buffers containing the shapes of the things we want to draw
    //      3. A framebuffer object, which is a collection of images to write to
    //      4. Descriptor sets to pass the data between the CPU and GPU
    // The GPU begins by positioning the shapes on the screen by placing the vertices where they're supposed to go.
    // Then the fragment shader colors each pixel of all the shapes.  Note that the graphics pipeline object contains
    // both the vertex shader and the fragment shader and various options that can allow us to configure the GPU's
    // behavior

    // First we need to have objects to draw.  We'll only use triangles to construct objects, but things called
    // "tesselations" let us use other shapes as well, but we won't use these.  Vulkano expects us to create a struct
    // to hold the coordinates of each vertex
    #[derive(Default, Copy, Clone)]  // Apparently a vertex must implement these traits
    struct Vertatertot {  // The vertex struct can have whatever name we want to give it
        // Normally, this would be a three-dimensional array.  Note that this can also use whatever name we give it
        poop_position: [f32; 2],

        // There may optionally also be an array for color, such as this:
        // color: [f32; 4];
    }

    // THIS IS A CRUCIAL STEP!!!!!!!!!  This is what makes the connection between the struct and its component names
    // and the names of the inputs to the shaders.
    vulkano::impl_vertex!(Vertatertot, poop_position);

    // Note that in Vulkan, the screen's coordinate space is a 2x2x1 Euclidean grid, spanning from
    // x \in [-1, 1], y\in [-1, 1], and z \in [0, 1]

    // For now, we'll use a simple triangle to demonstrate how to render things
    let vert1 = Vertatertot { poop_position: [-0.5, -0.5 ] };
    let vert2 = Vertatertot { poop_position: [ 0.0,  0.5 ] };
    let vert3 = Vertatertot { poop_position: [ 0.5, -0.25] };

    // Now we need to put these vertices in a buffer so that we'll be able to pass them along to the GPU.  Such a buffer
    // is usually called a vertex buffer (with variable name `vertex_buffer`, but here we deviate from that since we can)
    let vert_bufferlo = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
        vec![vert1, vert2, vert3].into_iter()).unwrap();

    // All right, so there's a thing called a "render pass".  Basically, it's a mode that we enter when we need the GPU to
    // draw stuff (rather than just asking it to draw things whenver we get to them, i.e., the CPU asking the GPU to draw
    // one triangle at a time).  We must create a render pass object to handle this for us.  A render pass is made up of
    // "attachments" and "passes", neither of which are explained
    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color_of_poop: {
                    load: Clear,   // Tells GPU to clear image upon entering render mode (fill with single background color)
                    store: Store,  // Tells GPU to actually store the output of our draw commands to the image
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,    // No multisampling (used for antialiasing)
                }
            },
            pass: {
                color: [color_of_poop],
                depth_stencil: {}
            }
        ).unwrap());

    // Re-initialize the image buffer for this section for clarity.  We're using `StorageImage` because we're still
    // writing to a file, not rendering live to a window
    let image = StorageImage::new(device.clone(), Dimensions::Dim2d {width: 1024, height: 1024},
        Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

    // Create this buffer to hold the image output
    let image_output_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        (0 .. 1024 * 1024 * 4).map(|_| 0u8)
    ).expect("failed to create buffer");

    // The render pass describes the format we're using and how to store and load the image object we'll use.  Next,
    // we must create a frame buffer to hold a list of the actual attachments we'll use.  In general, for performance
    // reasons, it's usually a good idea to keep the frame buffer objects around between frames and reuse it
    let frame_buffer = Arc::new(
        Framebuffer::start(render_pass.clone())
            .add(image.clone()).unwrap()
        .build().unwrap()
    );

    // Now, we build a command buffer as usual.  Then, we call `begin_render_pass()` to, well, begin the render pass.
    // We tell it which frame buffer we want to use, a Boolean which we'll ignore for now, and the color to fill the
    // attachments with (the color we use here is blue, so we should ultimately be drawing a red triangle with a blue
    // background).  Since we have only one attachment in this instance, we need only one clear color
    AutoCommandBufferBuilder::primary_one_time_submit(
        device.clone(),
        queue.family()).unwrap()
        .begin_render_pass(
            frame_buffer.clone(),
            false,
            vec![[0.0, 0.0, 1.0, 1.0].into()])
        .unwrap()
        .end_render_pass()
        .unwrap();

    // Finally, let's put everything together!  First, load the shaders
    let vert_shader = vertex_shader::vertex_shader::Shader::load(device.clone()).expect("failed to create shader module");
    let frag_shader = fragment_shader::fragment_shader::Shader::load(device.clone()).expect("failed to create shader module");

    // Now, build the graphics pipeline
    let graphics_pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertatertot>()  // Specifies the type of vertex input (the struct) that will be used
            .vertex_shader(vert_shader.main_entry_point(), ())  // Tells it which vertex shader to use
            .viewports_dynamic_scissors_irrelevant(1)  //  See below
            .fragment_shader(frag_shader.main_entry_point(), ())  // Tells it which fragment shader to use
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())  // "This graphics pipeline object concerns the first pass of the render pass"
            .build(device.clone())  // Now that everything is specified, build it
            .unwrap()
    );

    // The `viewports_dynamic_scissors_irrelevant(1)` method call tells it that we want one viewport, that it should be
    // dynamic (this lets us change the viewport for each draw command without having to build a new pipeline object if,
    // for example, we wanted to resize the viewport).  I believe that the "scissors" part refers to the fact that
    // shapes outside the viewport will be "clipped" and not rendered, since they can't appear in the image anyways

    // Set up the dynamic state object to be able to draw things
    let dynamic_state = DynamicState {
        viewports: Some(vec![
            Viewport {
                origin:     [0.0, 0.0],
                dimensions: [1024.0, 1024.0],
                depth_range: 0.0..1.0,
            }
        ]),

        // This line is saying to fill the remaining values of the `DynamicState` object with the values of `DynamicState::none()`
        .. DynamicState::none()
    };

    // Now we set up the command buffer to tell Vulkan to draw the stuff
    let draw_command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
        device.clone(),
        queue.family()
    ).unwrap()
        .begin_render_pass(
            frame_buffer.clone(),
            false,
            vec![[0.0, 0.0, 1.0, 1.0].into()]
        ).unwrap()
        .draw(
            graphics_pipeline.clone(),
            &dynamic_state,
            vert_bufferlo.clone(),
            (),
            ()
        ).unwrap()
        .end_render_pass().unwrap()
        .copy_image_to_buffer(
            image.clone(),
            image_output_buffer.clone()
        ).unwrap()
        .build().unwrap();

    let finished = draw_command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let draw_buffer_content = image_output_buffer.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
        1024,
        1024,
        &draw_buffer_content[..]
    ).unwrap();
    image.save("triangle.png").unwrap();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Windowing!!
    ////////////////////////////////////////////////////////////////////////////////////////////////

    let extensions = vulkano_win::required_extensions();

    // Make a new instance with the required extensions
    let instance = Instance::new(
        None,
        &extensions,
        None
    ).expect("failed to create Vulkan instance");

    // Now, get physical devices
    let physical = PhysicalDevice::enumerate(&instance)
        .next().expect("No device available");

    // Now, get a family of queues from the physical devices.  A `Queue` is a a place where commands can be submitted
    // for the GPU to process (usually transfer or compute queues)
    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("Couldn't find a graphical queue family");

    // Now, just pick one of the devices from the physical devices and the queue families.  This builds a new Vulkan device
    // for the given physical device.  It then uses the `queue_family` with priority `0.5`
    let (device, mut queues) = Device::new(
        physical,
        &Features::none(),
        &DeviceExtensions::none(),
        [(queue_family, 0.5)].iter().cloned(),
    ).expect("Failed to create device");

    // For now, just grab the first queue in the iterator
    let queue = queues.next().unwrap();

    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();

    events_loop.run_forever(|event| {
        match event {
            winit::Event::WindowEvent {
                event: winit::WindowEvent::CloseRequested, ..
            } => {
                winit::ControlFlow::Break
            },
            winit::Event::WindowEvent {
                event: winit::WindowEvent::KeyboardInput { input, .. },
                window_id,
            } => {
                if window_id == surface.window().id() {
                    if input.virtual_keycode == Some(VirtualKeyCode::Escape) {
                        winit::ControlFlow::Break
                    } else {
                        winit::ControlFlow::Continue
                    }
                } else {
                    winit::ControlFlow::Continue
                }
            },
            _ => winit::ControlFlow::Continue,
        }
    });

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Swap chain (i.e., frame buffering, such as double or triple buffering)
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // First, get the capabilities of the surface we're using
    let caps = surface.capabilities(physical.clone()).expect("Failed to get surface capabitilities!");

    // Now we need use these capabilities to set the properties for the swap chain
    let dimensions = caps.current_extent.unwrap_or([1280, 1024]);
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let format = caps.supported_formats[0].0;

    // Now we may actually create the swap chain
    let (swap_chain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        caps.min_image_count,
        format,
        dimensions,
        1,
        caps.supported_usage_flags,
        &queue,
        SurfaceTransform::Identity,
        alpha,
        PresentMode::Fifo,
        true,
        None
    ).expect("Failed to get swap chain");
    // let (swapchain, images) = Swapchain::new(device.clone(), surface.clone(),
    //     caps.min_image_count, format, dimensions, 1, caps.supported_usage_flags, &queue,
    //     SurfaceTransform::Identity, alpha, PresentMode::Fifo, true, None)
    //     .expect("failed to create swapchain");

    println!("poop5");
    // Now we need to tell the device to be able to use this `winit` extension
    let device_ext = vulkano::device::DeviceExtensions {
        khr_swapchain: true,
        .. vulkano::device::DeviceExtensions::none()
    };
    println!("poop6");
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned()
    ).expect("Failed to create device");
}

// pub fn images(device: &mut Device, dim: Dimensions, format: Format, queue: &Queue) {
//     // Note that an "image" in Vulkan refers to a multidimensional array of pixels.  This does not have to
//     // correspond to a typical "image" (i.e., a matrix of pixels, each consisting of R, G, B,
//     // and sometimes alpha values).  Of course, Vulkan images can indeed be used to store typical images as well.
//     // Vulkan images may of course be two-dimensional, but may also be one- or three-dimensional.
//     let image = StorageImage::new(device, dim,
//                                   format, Some(queue.family())).unwrap();
// }
