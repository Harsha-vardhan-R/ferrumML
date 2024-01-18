use ash::{vk, Entry, LoadingError, Instance};

///! Contains the boiler plate code needed to use the Vulcan API,
///! like loading the library etc...


/// Creates a new Vulcan instance panics for any errors in creating and using the device.
pub fn initialize_device() -> Result<Instance, LoadingError>{

    // Loading the Vulkan library at runtime.
    let entry = unsafe { Entry::load()? };

    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 0, 0),
        ..Default::default()
    };
    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        ..Default::default()
    };

    //trying to create an instance.
    let instance = unsafe { entry.create_instance(&create_info, None).expect("Failed to create an Instance") };
    
    Ok(instance)
}


/// Creates and returns a buffer in the vram.
/// takes in the size of the buffer.
pub fn create_buffer(size : usize) {

    
}